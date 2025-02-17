project(cudaGL CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# Find OpenGL and GLUT
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(CUDA REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/Common
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/GL
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/UtilNPP
    ${OPENGL_INCLUDE_DIR}
    ${GLUT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

# Source files
set(SOURCE_FILES
    main.cu
)


# Add the executable
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

# Link CUDA library to the executable
target_link_libraries(${PROJECT_NAME}
    ${OPENGL_LIBRARIES}
    ${GLUT_LIBRARIES}
    ${CUDA_LIBRARIES}
)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
