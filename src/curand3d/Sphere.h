#pragma once

#include "CudaGLBuffer.h"
#include <curand.h>
#include <curand_kernel.h>
#include <stdexcept>
#include <vector>
#include <memory>

static __global__ void finalizePoints(float* data, int numPoints)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;

    // Each point: data[4*idx + 0..3] => (x, y, z, rRandom)
    float x = data[4*idx + 0];
    float y = data[4*idx + 1];
    float z = data[4*idx + 2];
    float r = data[4*idx + 3];

    // Normalize (x, y, z)
    float len = sqrtf(x*x + y*y + z*z);
    if (len > 1e-7f) {
        x /= len; 
        y /= len; 
        z /= len;
    }

    // For example: radius = abs(1 + 0.2f * r)
    float radius = 1.0f + 0.2f * r;
    if (radius < 0.0f) {
        radius = -radius;
    }

    x *= radius;
    y *= radius;
    z *= radius;

    data[4*idx + 0] = x;
    data[4*idx + 1] = y;
    data[4*idx + 2] = z;
    data[4*idx + 3] = radius;
}

class Sphere {
public:
    Sphere(int numPoints, unsigned long long seed) 
    : numPoints_(numPoints),
      floatsPerVertex_(4)
    {
        // 1) Create the buffer
        buffer_ = std::make_unique<CudaGLBuffer>(numPoints_ * floatsPerVertex_ * sizeof(float));

        // 2) Generate random data in [buffer_] using cuRAND
        generateRandomData(seed);

        // 3) Create VAO for rendering
        setupVAO();
    }

    // For convenience, let move semantics happen
    Sphere(Sphere&&) = default;
    Sphere& operator=(Sphere&&) = default;

    // No copy
    Sphere(const Sphere&) = delete;
    Sphere& operator=(const Sphere&) = delete;

    ~Sphere() {
        // RAII: buffer_ and VAO are automatically cleaned up
        if (vao_) {
            glDeleteVertexArrays(1, &vao_);
        }
    }

    void draw() const {
        glBindVertexArray(vao_);
        glDrawArrays(GL_POINTS, 0, numPoints_);
        glBindVertexArray(0);
    }

private:
    int   numPoints_ = 0;
    int   floatsPerVertex_ = 4;
    GLuint vao_ = 0;
    std::unique_ptr<CudaGLBuffer> buffer_; 

    void generateRandomData(unsigned long long seed) {
        // Map the buffer for CUDA
        float* dPtr = buffer_->map();
        if(!dPtr) {
            throw std::runtime_error("Failed to map buffer for random data");
        }

        // Build a cuRAND generator
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, seed);

        // curandGenerateNormal needs an even number of floats
        size_t floatCount = size_t(numPoints_) * floatsPerVertex_;
        size_t alignedCount = (floatCount + 1) & ~1; // round up
        curandGenerateNormal(gen, dPtr, alignedCount, 0.0f, 1.0f);

        // Kernel: finalize direction + radius
        int blockSize = 256;
        int gridSize  = (numPoints_ + blockSize - 1) / blockSize;
        finalizePoints<<<gridSize, blockSize>>>(dPtr, numPoints_);
        cudaDeviceSynchronize(); // For simplicity

        // Cleanup
        curandDestroyGenerator(gen);
        buffer_->unmap();
    }

    void setupVAO() {
        glGenVertexArrays(1, &vao_);
        glBindVertexArray(vao_);

        // Bind the underlying buffer
        glBindBuffer(GL_ARRAY_BUFFER, buffer_->vbo());
        // Setup attribute layout: location 0 => 4 floats (x,y,z,r)
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4*sizeof(float), nullptr);

        glBindVertexArray(0);
    }
};
