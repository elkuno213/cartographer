# Initialize variables
if (CAIRO_ABSOLUTE_LIBRARIES)
  set(CAIRO_ABSOLUTE_LIBRARIES "")
endif()

# Find cairo by pkgconfig
find_package(PkgConfig REQUIRED)
pkg_search_module(CAIRO REQUIRED cairo>=1.12.16)

# Find the full paths to the Cairo libraries
foreach(LIB ${CAIRO_LIBRARIES})
  find_library(FOUND_${LIB} NAMES ${LIB} PATHS ${CAIRO_LIBRARY_DIRS})
  list(APPEND CAIRO_ABSOLUTE_LIBRARIES ${FOUND_${LIB}})
endforeach()

message(STATUS "CAIRO_INCLUDE_DIRS=${CAIRO_INCLUDE_DIRS}")
message(STATUS "CAIRO_ABSOLUTE_LIBRARIES=${CAIRO_ABSOLUTE_LIBRARIES}")

# Find pthread used by cairo
find_library(PTHREAD_LIB pthread)

# Set variables
set(CAIRO_INCLUDE_DIRS ${CAIRO_INCLUDE_DIRS} CACHE PATH "Cairo include directories" FORCE)
set(CAIRO_ABSOLUTE_LIBRARIES ${CAIRO_ABSOLUTE_LIBRARIES} ${PTHREAD_LIB} CACHE FILEPATH "Cairo absolute libraries" FORCE)
