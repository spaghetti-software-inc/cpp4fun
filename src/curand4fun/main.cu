#include <iostream>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <curand.h>
#include <cuda_gl_interop.h>


// Forward declaration of the resize callback
void framebuffer_size_callback(GLFWwindow* window, int width, int height);

int main() {
    // ------------------------------------------------------------------------
    // 1. Initialize GLFW and Create Window
    // ------------------------------------------------------------------------
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return -1;
    }

    // Set GLFW window hints here if needed (e.g., OpenGL version)

    GLFWwindow* window = glfwCreateWindow(800, 600, "CUDA-GL Scatter", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // Register the framebuffer size callback
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // ------------------------------------------------------------------------
    // 2. Initialize GLEW
    // ------------------------------------------------------------------------
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    // Set initial viewport
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    glViewport(0, 0, width, height);

    // ------------------------------------------------------------------------
    // 3. Create VBO
    // ------------------------------------------------------------------------
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    // Suppose we want `N` points, each with 2 floats (x, y).
    size_t numPoints = 100000;
    size_t bufferSize = numPoints * 2 * sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, bufferSize, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // ------------------------------------------------------------------------
    // 4. Register VBO with CUDA
    // ------------------------------------------------------------------------
    cudaGraphicsResource* cudaVboResource;
    cudaGraphicsGLRegisterBuffer(&cudaVboResource, vbo, cudaGraphicsMapFlagsWriteDiscard);

    // ------------------------------------------------------------------------
    // 5. Create a CURAND Generator
    // ------------------------------------------------------------------------
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    // Optional: set a fixed seed for reproducibility
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    // ------------------------------------------------------------------------
    // 6. Main Loop
    // ------------------------------------------------------------------------
    while (!glfwWindowShouldClose(window)) {
        // Map the buffer so CUDA can write to it
        cudaGraphicsMapResources(1, &cudaVboResource, 0);
        void* dPtr = nullptr;
        size_t dSize = 0;
        cudaGraphicsResourceGetMappedPointer(&dPtr, &dSize, cudaVboResource);

        // Generate normally distributed data directly into the buffer
        float mean = 0.0f;
        float stddev = 1.0f;
        curandGenerateNormal(gen, static_cast<float*>(dPtr), 2 * numPoints, mean, stddev);

        // Unmap so OpenGL can use it again
        cudaGraphicsUnmapResources(1, &cudaVboResource, 0);

        // Render
        glClear(GL_COLOR_BUFFER_BIT);

        // Bind the VBO
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);

        // Draw the points
        glDrawArrays(GL_POINTS, 0, numPoints);

        glDisableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // Swap front/back buffers
        glfwSwapBuffers(window);
        // Poll for events (like keypress, window resize, etc.)
        glfwPollEvents();
    }

    // ------------------------------------------------------------------------
    // 7. Cleanup
    // ------------------------------------------------------------------------
    // Clean up CURAND
    curandDestroyGenerator(gen);

    // Unregister the buffer object from CUDA
    cudaGraphicsUnregisterResource(cudaVboResource);

    // Delete buffer
    glDeleteBuffers(1, &vbo);

    // Close window and terminate GLFW
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}

// ------------------------------------------------------------------------
// Callback: Update the viewport when the framebuffer is resized
// ------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}
