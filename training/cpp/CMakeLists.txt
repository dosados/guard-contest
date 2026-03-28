cmake_minimum_required(VERSION 3.16)
project(parquet_to_catboost_tsv LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(DEFINED ENV{CONDA_PREFIX})
  list(PREPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
endif()

find_package(Arrow REQUIRED CONFIG)
find_package(Parquet REQUIRED CONFIG)
find_package(Threads REQUIRED)

add_executable(parquet_to_catboost_tsv parquet_to_catboost_tsv.cpp)
target_link_libraries(
  parquet_to_catboost_tsv
  PRIVATE Arrow::arrow_shared Parquet::parquet_shared Threads::Threads
)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
  target_link_libraries(parquet_to_catboost_tsv PRIVATE stdc++fs)
endif()
