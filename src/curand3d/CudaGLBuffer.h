#pragma once

#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <stdexcept>

class CudaGLBuffer {
public:
    CudaGLBuffer(GLsizeiptr sizeBytes, GLenum usage = GL_DYNAMIC_DRAW)
    : sizeBytes_(sizeBytes)
    {
        // Create and bind VBO
        glGenBuffers(1, &vbo_);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_);
        glBufferData(GL_ARRAY_BUFFER, sizeBytes_, nullptr, usage);

        // Register with CUDA
        cudaError_t err = cudaGraphicsGLRegisterBuffer(&cudaResource_, vbo_, cudaGraphicsMapFlagsWriteDiscard);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to register GL buffer with CUDA");
        }
    }

    ~CudaGLBuffer() {
        if (cudaResource_) {
            cudaGraphicsUnregisterResource(cudaResource_);
        }
        if (vbo_) {
            glDeleteBuffers(1, &vbo_);
        }
    }

    CudaGLBuffer(const CudaGLBuffer&) = delete;
    CudaGLBuffer& operator=(const CudaGLBuffer&) = delete;

    CudaGLBuffer(CudaGLBuffer&& other) noexcept {
        move(std::move(other));
    }
    CudaGLBuffer& operator=(CudaGLBuffer&& other) noexcept {
        if (this != &other) {
            cleanup();
            move(std::move(other));
        }
        return *this;
    }

    // Map the buffer so that you can write to it from a CUDA kernel
    float* map() {
        if (!cudaResource_) return nullptr;
        cudaGraphicsMapResources(1, &cudaResource_, 0);
        float* dPtr = nullptr;
        size_t dSize = 0;
        cudaGraphicsResourceGetMappedPointer((void**)&dPtr, &dSize, cudaResource_);
        return dPtr;
    }

    // Unmap the buffer when done
    void unmap() {
        if (cudaResource_) {
            cudaGraphicsUnmapResources(1, &cudaResource_, 0);
        }
    }

    GLuint vbo()   const { return vbo_; }
    size_t size()  const { return sizeBytes_; }

private:
    GLuint vbo_ = 0;
    size_t sizeBytes_ = 0;
    cudaGraphicsResource* cudaResource_ = nullptr;

    void cleanup() {
        if (cudaResource_) {
            cudaGraphicsUnregisterResource(cudaResource_);
            cudaResource_ = nullptr;
        }
        if (vbo_) {
            glDeleteBuffers(1, &vbo_);
            vbo_ = 0;
        }
    }

    void move(CudaGLBuffer&& other) {
        vbo_          = other.vbo_;
        sizeBytes_    = other.sizeBytes_;
        cudaResource_ = other.cudaResource_;

        other.vbo_          = 0;
        other.sizeBytes_    = 0;
        other.cudaResource_ = nullptr;
    }
};
