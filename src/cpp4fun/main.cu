#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <curand.h>
#include <cuda_gl_interop.h>

int main() {
    // 1. OpenGL init
    glfwInit();
    GLFWwindow* window = glfwCreateWindow(800, 600, "CUDA-GL Scatter", nullptr, nullptr);
    glfwMakeContextCurrent(window);
    glewInit();

    // 2. Create VBO
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    size_t numPoints = 100000;
    size_t bufferSize = numPoints * 2 * sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, bufferSize, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // 3. Register VBO with CUDA
    cudaGraphicsResource* cudaVboResource;
    cudaGraphicsGLRegisterBuffer(&cudaVboResource, vbo, cudaGraphicsMapFlagsWriteDiscard);

    // Create a CURAND generator
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        // 4. Map resource
        cudaGraphicsMapResources(1, &cudaVboResource, 0);
        void* dPtr;
        size_t size;
        cudaGraphicsResourceGetMappedPointer(&dPtr, &size, cudaVboResource);

        // 5. Generate random data directly into the mapped pointer
        float mean = 0.0f;
        float stddev = 1.0f;
        curandGenerateNormal(gen, (float*)dPtr, 2 * numPoints, mean, stddev);

        // 6. Unmap
        cudaGraphicsUnmapResources(1, &cudaVboResource, 0);

        // 7. Render
        glClear(GL_COLOR_BUFFER_BIT);

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2*sizeof(float), (void*)0);
        glDrawArrays(GL_POINTS, 0, numPoints);
        glDisableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // Swap buffers
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    curandDestroyGenerator(gen);
    cudaGraphicsUnregisterResource(cudaVboResource);
    glDeleteBuffers(1, &vbo);
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
