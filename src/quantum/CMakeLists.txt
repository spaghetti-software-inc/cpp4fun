project(quantum CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)


find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)

find_package(CUDA REQUIRED)


include_directories(
    ${OPENGL_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${GLFW_INCLUDE_DIRS}
)


# Recursively find all .cpp files in the tests directory and subdirectories
#file(GLOB_RECURSE EXE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)


# Source files
set(SOURCE_FILES
    main.cu
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME}
    ${OPENGL_LIBRARIES}    
    ${CUDA_LIBRARIES}
    glfw
    GLEW::GLEW
    glm::glm
    curand
)


set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
