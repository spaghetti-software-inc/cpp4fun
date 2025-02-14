project(cpp4fun CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(ZLIB REQUIRED)

# Recursively find all .cpp files in the tests directory and subdirectories
file(GLOB_RECURSE EXE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(${PROJECT_NAME} ${EXE_SOURCES})
target_link_libraries(${PROJECT_NAME} ZLIB::ZLIB)


