#ifndef RENDERER_H
#define RENDERER_H

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h


class Renderer {
private:
    const unsigned int _mesh_width    = 256;
    const unsigned int _mesh_height   = 256;

    // vbo variables
    GLuint _vbo;
    struct cudaGraphicsResource* _cuda_vbo_resource = nullptr;
    void* _d_vbo_buffer = nullptr;

    

public:
    Renderer();
    ~Renderer();

    void render();
};

#endif