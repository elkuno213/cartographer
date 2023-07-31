/*
 * Copyright 2018 The Cartographer Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cartographer/mapping/internal/imu_based_pose_extrapolator.h"

#include <algorithm>

#include "absl/memory/memory.h"
#include "cartographer/mapping/internal/3d/imu_integration.h"
#include "cartographer/mapping/internal/3d/rotation_parameterization.h"
#include "cartographer/mapping/internal/eigen_quaterniond_from_two_vectors.h"
#include "cartographer/mapping/internal/optimization/ceres_pose.h"
#include "cartographer/mapping/internal/optimization/cost_functions/acceleration_cost_function_3d.h"
#include "cartographer/mapping/internal/optimization/cost_functions/rotation_cost_function_3d.h"
#include "cartographer/mapping/internal/optimization/cost_functions/spa_cost_function_3d.h"
#include "cartographer/mapping/pose_graph_interface.h"
#include "cartographer/transform/transform.h"
#include "glog/logging.h"

namespace cartographer {
namespace mapping {

using ::cartographer::transform::TimestampedTransform;

ImuBasedPoseExtrapolator::ImuBasedPoseExtrapolator(
  const proto::ImuBasedPoseExtrapolatorOptions& options
)
  : options_(options)
  , solver_options_(common::CreateCeresSolverOptions(options_.solver_options()))
{}

ImuBasedPoseExtrapolator::~ImuBasedPoseExtrapolator() {
  LOG(INFO) << "Number of iterations for pose extrapolation:";
  LOG(INFO) << num_iterations_hist_.ToString(10);
}

std::unique_ptr<PoseExtrapolatorInterface>
ImuBasedPoseExtrapolator::InitializeWithImu(
  const proto::ImuBasedPoseExtrapolatorOptions& options,
  const std::vector<sensor::ImuData>& imu_data_vector,
  const std::vector<transform::TimestampedTransform>& initial_poses
) {
  CHECK(!imu_data_vector.empty());
  LOG(INFO) << options.DebugString();

  // Init extrapolator
  auto extrapolator = absl::make_unique<ImuBasedPoseExtrapolator>(options);
  // Copy all IMU data into cached deque
  std::copy(
    imu_data_vector.begin(),
    imu_data_vector.end(),
    std::back_inserter(extrapolator->imu_data_deque_)
  );
  // Add poses into extrapolator
  if (!initial_poses.empty()) {
    for (const auto& pose : initial_poses) {
      if (pose.time > imu_data_vector.front().time) {
        extrapolator->AddPose(pose.time, pose.transform);
      }
    }
  } else {
    // If poses is empty, then add pose (only rotation) calculated by linear
    // acceleration of the last IMU data and the unit Z vector.
    extrapolator->AddPose(
      imu_data_vector.back().time,
      transform::Rigid3d::Rotation(FromTwoVectors(
        imu_data_vector.back().linear_acceleration, Eigen::Vector3d::UnitZ()
      ))
    );
  }

  return extrapolator;
}

common::Time ImuBasedPoseExtrapolator::GetLastPoseTime() const {
  if (timed_pose_deque_.empty()) {
    return common::Time::min();
  }
  return timed_pose_deque_.back().time;
}

common::Time ImuBasedPoseExtrapolator::GetLastExtrapolatedTime() const {
  return last_extrapolated_time_;
}

void ImuBasedPoseExtrapolator::AddPose(
  const common::Time time, const transform::Rigid3d& pose
) {
  timed_pose_deque_.push_back(TimestampedTransform{time, pose});

  // Trim timed poses.
  while (
    // timed pose deque must have at least 3 poses.
    timed_pose_deque_.size() > 3
    // pop the old timed pose out of deque, considering queue duration.
    && timed_pose_deque_[1].time <= time - common::FromSeconds(options_.pose_queue_duration())
  ) {
    if (!previous_solution_deque_.empty()) {
      CHECK_EQ(
        timed_pose_deque_.front().time, previous_solution_deque_.front().time
      );
      previous_solution_deque_.pop_front();
    }
    timed_pose_deque_.pop_front();
  }

  TrimImuData();
}

void ImuBasedPoseExtrapolator::AddImuData(const sensor::ImuData& imu_data) {
  // Make sure that timestamp of the input IMU data must be newer than the
  // currently latest timed pose's one.
  CHECK(
    timed_pose_deque_.empty() || imu_data.time >= timed_pose_deque_.back().time
  );
  // Add IMU data
  imu_data_deque_.push_back(imu_data);

  TrimImuData();
}

void ImuBasedPoseExtrapolator::AddOdometryData(
  const sensor::OdometryData& odometry_data
) {
  // Make sure that timestamp of the input Odometry data must be newer than the
  // currently latest timed pose's one.
  CHECK(
    timed_pose_deque_.empty()
    || odometry_data.time >= timed_pose_deque_.back().time
  );
  // Add Odometry data
  odometry_data_deque_.push_back(odometry_data);

  TrimOdometryData();
}

ImuBasedPoseExtrapolator::ExtrapolationResult
ImuBasedPoseExtrapolator::ExtrapolatePosesWithGravity(
  const std::vector<common::Time>& times
) {
  const auto& time              = times.back();
  const auto& newest_timed_pose = timed_pose_deque_.back();
  CHECK_GE(time, newest_timed_pose.time);
  CHECK_GE(times.size(), 1);
  last_extrapolated_time_ = time;


  // ──────────────────────────────────────────────────────────────────────── //
  // ───────────────────────────── Early return ───────────────────────────── //
  // If there are only 2 timed pose in cache OR time as latest one of timed pose
  // deque, then return a list of the latest time poses as previous ones and
  // zero velocity.
  if (timed_pose_deque_.size() < 3 || common::ToSeconds(time - newest_timed_pose.time) < 1e-6) {
    return ExtrapolationResult{
      std::vector<transform::Rigid3f>(
        times.size() - 1, timed_pose_deque_.back().transform.cast<float>()
      ),
      timed_pose_deque_.back().transform,
      Eigen::Vector3d::Zero(),
      timed_pose_deque_.back().transform.rotation()
    };
  }


  // ──────────────────────────────────────────────────────────────────────── //
  // ───────────────────────────── Optimization ───────────────────────────── //
  // 1. Initialize
  ceres::Problem::Options problem_options;
  ceres::Problem problem(problem_options);
  // Track gravity alignment over time and use this as a frame here so that
  // we can estimate the gravity alignment of the current pose.
  optimization::CeresPose gravity_from_local(
    gravity_from_local_,
    nullptr,
    absl::make_unique<ceres::QuaternionParameterization>(),
    &problem
  );

  // 2. Formulate problem
  // Use deque for nodes so addresses stay constant during problem formulation.
  std::deque<optimization::CeresPose> nodes;
  std::vector<common::Time> node_times;
  // Add nodes from timed pose deque OR last optimized result.
  for (size_t i = 0; i < timed_pose_deque_.size(); i++) {
    const bool is_last     = (i == (timed_pose_deque_.size() - 1));
    const auto& timed_pose = timed_pose_deque_[i];
    node_times.push_back(timed_pose.time);

    transform::Rigid3d gravity_from_node;
    // Use the last scan match result (timed_pose_queue_.back()) for
    // initialization here instead of the last result from the optimization.
    // This keeps poses from slowly drifting apart due to lack of feedback
    // from the scan matching here.
    if (!previous_solution_deque_.empty() && !is_last) {
      CHECK_GT(previous_solution_deque_.size(), i);
      CHECK_EQ(timed_pose.time, previous_solution_deque_[i].time);
      gravity_from_node = previous_solution_deque_[i].transform;
    } else {
      gravity_from_node = gravity_from_local_ * timed_pose.transform;
    }

    if (is_last) {
      nodes.emplace_back(
        gravity_from_node,
        nullptr,
        absl::make_unique<ceres::AutoDiffLocalParameterization<ConstantYawQuaternionPlus, 4, 2>>(),
        &problem
      );
      problem.SetParameterBlockConstant(nodes.back().translation()); // TODO(VuHoi): why?
    } else {
      nodes.emplace_back(
        gravity_from_node,
        nullptr,
        absl::make_unique<ceres::QuaternionParameterization>(),
        &problem
      );
    }
  }
  // Correct gravity constant if option is available.
  double gravity_constant = 9.8;
  bool fix_gravity        = false;
  if (options_.gravity_constant() > 0) {
    fix_gravity      = true;
    gravity_constant = options_.gravity_constant();
  }
  // Get initial estimate from double-previous IMU data and gravity from tracking
  // - Double-previous IMU data is the one older than the latest timed pose.
  // TODO(VuHoi): maybe use std::lower_bound.
  auto imu_it_prev_prev = imu_data_deque_.begin();
  while (
    imu_it_prev_prev != std::prev(imu_data_deque_.end()) // Not the lastest IMU data
    && std::next(imu_it_prev_prev)->time <= timed_pose_deque_.back().time
  ) {
    ++imu_it_prev_prev;
  }
  // - Add the last node (initial estimate) based on gravity constant.
  const TimestampedTransform prev_gravity_from_tracking
    = TimestampedTransform{node_times.back(), nodes.back().ToRigid()};
  const TimestampedTransform prev_prev_gravity_from_tracking
    = TimestampedTransform{
        node_times.at(node_times.size() - 2),
        nodes.at(nodes.size() - 2).ToRigid()
      };
  const transform::Rigid3d initial_estimate
    = ExtrapolatePoseWithImu<double>(
      prev_gravity_from_tracking.transform,
      prev_gravity_from_tracking.time,
      prev_prev_gravity_from_tracking.transform,
      prev_prev_gravity_from_tracking.time,
      gravity_constant * Eigen::Vector3d::UnitZ(),
      time,
      imu_data_deque_,
      &imu_it_prev_prev
    ).pose;
  nodes.emplace_back(
    initial_estimate,
    nullptr,
    absl::make_unique<ceres::QuaternionParameterization>(),
    &problem
  );
  node_times.push_back(time);
  // Add cost functions for node constraints.
  for (size_t i = 0; i < timed_pose_deque_.size(); i++) {
    const auto& timed_pose = timed_pose_deque_[i];
    problem.AddResidualBlock(
      optimization::SpaCostFunction3D::CreateAutoDiffCostFunction(
        PoseGraphInterface::Constraint::Pose{
          timed_pose.transform,
          options_.pose_translation_weight(),
          options_.pose_rotation_weight()}
      ),
      nullptr /* loss function */,
      // TODO(VuHoi): rotation() and translation() return 1st pointer to data of
      // std::array. Why only first element is considered?
      gravity_from_local.rotation(),
      gravity_from_local.translation(),
      nodes.at(i).rotation(),
      nodes.at(i).translation()
    );
  }
  // Check time of IMU data deque and timed pose deque.
  // TODO(VuHoi): Why do not do this before optimization.
  CHECK(!imu_data_deque_.empty());
  CHECK_LE(imu_data_deque_.front().time, timed_pose_deque_.front().time);
  // Add IMU calibration data and parameters into problem.
  std::array<double, 4> imu_calibration{{1., 0., 0., 0.}};
  problem.AddParameterBlock(
    imu_calibration.data(), 4, new ceres::QuaternionParameterization()
  );
  problem.SetParameterBlockConstant(imu_calibration.data());
  // Check size and time of IMU data deque.
  // TODO(VuHoi): Why do not do this before optimization.
  auto imu_it = imu_data_deque_.begin();
  CHECK(
    imu_data_deque_.size() == 1
    || std::next(imu_it)->time > timed_pose_deque_.front().time
  );
  // Integrate Odometry data into problem based on IMU data.
  transform::Rigid3d last_node_odometry;
  common::Time last_node_odometry_time;
  for (size_t i = 1; i < nodes.size(); i++) {
    const common::Time first_time  = node_times[i - 1];
    const common::Time second_time = node_times[i];

    auto imu_it2 = imu_it;
    const IntegrateImuResult<double> result
      = IntegrateImu(imu_data_deque_, first_time, second_time, &imu_it);
    if ((i + 1) < nodes.size()) {
      const common::Time third_time          = node_times[i + 1];
      const common::Duration first_duration  = second_time - first_time;
      const common::Duration second_duration = third_time - second_time;
      const common::Time first_center        = first_time + first_duration / 2;
      const common::Time second_center       = second_time + second_duration / 2;
      const IntegrateImuResult<double> result_to_first_center
        = IntegrateImu(imu_data_deque_, first_time, first_center, &imu_it2);
      const IntegrateImuResult<double> result_center_to_center
        = IntegrateImu(imu_data_deque_, first_center, second_center, &imu_it2);
      // 'delta_velocity' is the change in velocity from the point in time
      // halfway between the first and second poses to halfway between
      // second and third pose. It is computed from IMU data and still
      // contains a delta due to gravity. The orientation of this vector is
      // in the IMU frame at the second pose.
      const Eigen::Vector3d delta_velocity
        = (result.delta_rotation.inverse()
        * result_to_first_center.delta_rotation)
        * result_center_to_center.delta_velocity;
      problem.AddResidualBlock(
        AccelerationCostFunction3D::CreateAutoDiffCostFunction(
          options_.imu_acceleration_weight(),
          delta_velocity,
          common::ToSeconds(first_duration),
          common::ToSeconds(second_duration)
        ),
        nullptr /* loss function */,
        nodes.at(i).rotation(),
        nodes.at(i - 1).translation(),
        nodes.at(i).translation(),
        nodes.at(i + 1).translation(),
        &gravity_constant,
        imu_calibration.data()
      );
      // TODO(danielsievers): Fix gravity in CostFunction.
      if (fix_gravity) {
        problem.SetParameterBlockConstant(&gravity_constant);
      } else {
        // Force gravity constant to be positive.
        problem.SetParameterLowerBound(&gravity_constant, 0, 0.0);
      }
    }
    problem.AddResidualBlock(
      RotationCostFunction3D::CreateAutoDiffCostFunction(
        options_.imu_rotation_weight(), result.delta_rotation
      ),
      nullptr /* loss function */,
      nodes.at(i - 1).rotation(),
      nodes.at(i).rotation(),
      imu_calibration.data()
    );

    // Add a relative pose constraint based on the odometry (if available).
    if (HasOdometryDataForTime(first_time) && HasOdometryDataForTime(second_time)) {
      // Here keep track of last node odometry to avoid double computation.
      // Do this if first loop, or if there were some missing odometry nodes
      // then recalculate.
      if (i == 1 || last_node_odometry_time != first_time) {
        last_node_odometry      = InterpolateOdometry(first_time);
        last_node_odometry_time = first_time;
      }
      const transform::Rigid3d current_node_odometry
        = InterpolateOdometry(second_time);
      const transform::Rigid3d relative_odometry
        = CalculateOdometryBetweenNodes(
          last_node_odometry, current_node_odometry
        );

      problem.AddResidualBlock(
        optimization::SpaCostFunction3D::CreateAutoDiffCostFunction(
          PoseGraphInterface::Constraint::Pose{
            relative_odometry,
            options_.odometry_translation_weight(),
            options_.odometry_rotation_weight()
          }
        ),
        nullptr /* loss function */,
        nodes.at(i - 1).rotation(),
        nodes.at(i - 1).translation(),
        nodes.at(i).rotation(),
        nodes.at(i).translation()
      );
      // Use the current node odometry in the next iteration
      last_node_odometry      = current_node_odometry;
      last_node_odometry_time = second_time;
    }
  }

  // 3. Solve.
  ceres::Solver::Summary summary;
  ceres::Solve(solver_options_, &problem, &summary);
  LOG_IF_EVERY_N(INFO, !fix_gravity, 20) << "Gravity was: " << gravity_constant;

  // 4. Retrieve results.
  // Update estimated gravity, last pose and extrapolated pose.
  const auto gravity_estimate  = nodes.back().ToRigid().rotation();
  const auto& last_pose        = timed_pose_deque_.back();
  const auto extrapolated_pose = TimestampedTransform{
    time,
    last_pose.transform * nodes.at(nodes.size() - 2).ToRigid().inverse() * nodes.back().ToRigid()
  };
  // Update histogram of number of iterations.
  num_iterations_hist_.Add(summary.iterations.size());
  // Update gravity transform.
  gravity_from_local_ = gravity_from_local.ToRigid();
  // Update previous solution.
  previous_solution_deque_.clear();
  for (size_t i = 0; i < nodes.size(); ++i) {
    previous_solution_deque_.push_back(
      TimestampedTransform{node_times.at(i), nodes.at(i).ToRigid()}
    );
  }
  // Update current velocity.
  const Eigen::Vector3d current_velocity
    = (extrapolated_pose.transform.translation()
    - last_pose.transform.translation()) / common::ToSeconds(time - last_pose.time);


  return ExtrapolationResult{
    InterpolatePoses(last_pose, extrapolated_pose, times.begin(), std::prev(times.end())),
    extrapolated_pose.transform,
    current_velocity,
    gravity_estimate
  };
}

std::vector<transform::Rigid3f> ImuBasedPoseExtrapolator::InterpolatePoses(
  const TimestampedTransform& start,
  const TimestampedTransform& end,
  const std::vector<common::Time>::const_iterator times_begin,
  const std::vector<common::Time>::const_iterator times_end
) {
  std::vector<transform::Rigid3f> poses;
  poses.reserve(std::distance(times_begin, times_end));
  const float duration_scale = 1. / common::ToSeconds(end.time - start.time);

  const Eigen::Quaternionf start_rotation
    = Eigen::Quaternionf(start.transform.rotation());
  const Eigen::Quaternionf end_rotation
    = Eigen::Quaternionf(end.transform.rotation());
  const Eigen::Vector3f start_translation
    = start.transform.translation().cast<float>();
  const Eigen::Vector3f end_translation
    = end.transform.translation().cast<float>();

  for (auto it = times_begin; it != times_end; ++it) {
    const float factor = common::ToSeconds(*it - start.time) * duration_scale;
    const Eigen::Vector3f origin
      = start_translation + (end_translation - start_translation) * factor;
    const Eigen::Quaternionf rotation
      = start_rotation.slerp(factor, end_rotation);
    poses.emplace_back(origin, rotation);
  }
  return poses;
}

transform::Rigid3d ImuBasedPoseExtrapolator::ExtrapolatePose(
  const common::Time time
) {
  return ExtrapolatePosesWithGravity(std::vector<common::Time>{time}).current_pose;
}

Eigen::Quaterniond ImuBasedPoseExtrapolator::EstimateGravityOrientation(
  const common::Time time
) {
  return ExtrapolatePosesWithGravity(std::vector<common::Time>{time}).gravity_from_tracking;
}

template <typename T>
void ImuBasedPoseExtrapolator::TrimDequeData(std::deque<T>* data) {
  while (
    data->size() > 1 // At least 2 elements to trim
    && !timed_pose_deque_.empty() // timed pose deque must NOT be empty for time comparison
    && data->at(1).time <= timed_pose_deque_.front().time // check time
  ) {
    data->pop_front();
  }
}

void ImuBasedPoseExtrapolator::TrimImuData() {
  TrimDequeData<sensor::ImuData>(&imu_data_deque_);
}

void ImuBasedPoseExtrapolator::TrimOdometryData() {
  TrimDequeData<sensor::OdometryData>(&odometry_data_deque_);
}

// Odometry methods
bool ImuBasedPoseExtrapolator::HasOdometryData() const {
  return odometry_data_deque_.size() >= 2;
}

bool ImuBasedPoseExtrapolator::HasOdometryDataForTime(const common::Time& time
) const {
  return HasOdometryData()
         && odometry_data_deque_.front().time < time
         && time < odometry_data_deque_.back().time;
}

transform::Rigid3d ImuBasedPoseExtrapolator::InterpolateOdometry(
  const common::Time& time
) const {
  // Only interpolate if time is within odometry data range.
  CHECK(HasOdometryDataForTime(time)) << "Odometry data range does not include time " << time;

  std::deque<sensor::OdometryData>::const_iterator data_it = std::upper_bound(
    odometry_data_deque_.begin(),
    odometry_data_deque_.end(),
    time,
    [](const common::Time& time, const sensor::OdometryData& odometry_data) {
      return time < odometry_data.time;
    }
  );
  const TimestampedTransform first{std::prev(data_it)->time, std::prev(data_it)->pose};
  const TimestampedTransform second{data_it->time, data_it->pose};

  return Interpolate(first, second, time).transform;
}

transform::Rigid3d ImuBasedPoseExtrapolator::CalculateOdometryBetweenNodes(
  const transform::Rigid3d& first_node_odometry,
  const transform::Rigid3d& second_node_odometry
) const {
  return first_node_odometry.inverse() * second_node_odometry;
}

} // namespace mapping
} // namespace cartographer
