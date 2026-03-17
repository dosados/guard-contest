cmake_minimum_required(VERSION 3.16)
project(guard_dataset_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Arrow REQUIRED CONFIG)
find_package(Parquet REQUIRED CONFIG)
find_package(OpenSSL REQUIRED)

add_executable(build_dataset
    build_dataset.cpp
)

target_link_libraries(build_dataset
    PRIVATE
        arrow_shared
        parquet_shared
        OpenSSL::Crypto
)

target_compile_definitions(build_dataset PRIVATE
    ARROW_PARQUET
)

