cmake_minimum_required(VERSION 3.15)
project(cpp4fun CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

find_package(ZLIB REQUIRED)



add_library(particles STATIC
  src/randomize.cpp
  src/randomize.h
  src/particle.cu
  src/particle.h
  src/v3.cu
  src/v3.h
  )

set_target_properties( particles
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                      )

add_executable(particle_test src/test.cu)

set_target_properties(particle_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(particle_test PRIVATE particles)


add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} ZLIB::ZLIB)

add_executable(tests tests/test.cpp)
target_link_libraries(tests GTest::gtest_main ZLIB::ZLIB)

include(GoogleTest)
enable_testing()

