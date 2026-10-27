cmake_minimum_required(VERSION 3.20)
project(timegeo_port LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)
# conda-forge's exported h3::h3 target references a build-machine libm path; link by file instead
find_library(H3_LIBRARY NAMES h3 REQUIRED)
find_path(H3_INCLUDE_DIR h3/h3api.h REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Snappy CONFIG)

# unmodified upstream module (environment proof)
add_executable(module_2_3_1_upstream module_2_3_1.cpp)
target_include_directories(module_2_3_1_upstream PRIVATE ${H3_INCLUDE_DIR})
target_link_libraries(module_2_3_1_upstream PRIVATE Arrow::arrow_shared Parquet::parquet_shared ${H3_LIBRARY} OpenMP::OpenMP_CXX)

# ported module
add_executable(module_2_3_1_port module_2_3_1_port.cpp)
target_link_libraries(module_2_3_1_port PRIVATE Arrow::arrow_shared Parquet::parquet_shared)

if(Snappy_FOUND)
  target_link_libraries(module_2_3_1_upstream PRIVATE Snappy::snappy)
  target_link_libraries(module_2_3_1_port PRIVATE Snappy::snappy)
endif()
