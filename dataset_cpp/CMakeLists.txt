cmake_minimum_required(VERSION 3.16)
project(build_dataset_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Искать Arrow/Parquet/OpenSSL в префиксе активированного env conda
if(DEFINED ENV{CONDA_PREFIX})
  list(PREPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
  set(OPENSSL_ROOT_DIR "$ENV{CONDA_PREFIX}" CACHE PATH "OpenSSL root (conda)")
endif()

find_package(Arrow REQUIRED CONFIG)
find_package(Parquet REQUIRED CONFIG)
find_package(OpenSSL REQUIRED)

add_executable(build_dataset build_dataset.cpp)
target_link_libraries(build_dataset PRIVATE Arrow::arrow_shared Parquet::parquet_shared OpenSSL::Crypto)
# std::filesystem (gcc often needs explicit link before gcc 9)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
  target_link_libraries(build_dataset PRIVATE stdc++fs)
endif()