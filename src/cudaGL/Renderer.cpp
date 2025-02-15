// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check

#include "Renderer.h"

Renderer::Renderer() {
    checkCudaErrors(cudaMalloc((void **)&_d_vbo_buffer, _mesh_width*_mesh_height*4*sizeof(float)));
}

Renderer::~Renderer() {
}

void Renderer::render() {
    glClearColor(0.529f, 0.808f, 0.922f, 1.0f); // Light sky blue
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}
