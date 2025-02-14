project(particles CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


add_library(${PROJECT_NAME}_lib STATIC
  randomize.cpp
  randomize.h
  particle.cu
  particle.h
  v3.cu
  v3.h
  )

set_target_properties( ${PROJECT_NAME}_lib
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                      )

add_executable(${PROJECT_NAME} test.cu)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_NAME}_lib)




