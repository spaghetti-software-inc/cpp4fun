/******************************************************************************
 *  DoubleSlit.cu
 *
 *  A minimal CUDA + OpenGL program demonstrating a naive double-slit
 *  interference pattern via Monte Carlo (rejection sampling).
 *
 *  - Press Escape or close the window to exit.
 *  - The code uses a single pass each frame to generate N new points
 *    according to the double-slit interference probability distribution.
 *  - The slits are aligned along the y-axis, and the screen is in the x-axis
 *    (we only display x from -xRange to +xRange on screen).
 ******************************************************************************/

 #include <iostream>
 #include <cmath>
 #include <GL/glew.h>
 #include <GLFW/glfw3.h>
 
 #include <cuda_runtime.h>
 #include <curand_kernel.h>
 #include <cuda_gl_interop.h>
 
 // ----------------------------------------------------------
 // Constants for the double-slit experiment (tweak as desired)
 // ----------------------------------------------------------
 static const float LAMBDA   = 0.5e-6f;   // Wavelength (m), e.g. 500 nm
 static const float SLIT_DIST= 1.0e-3f;   // Center-to-center slit separation (m)
 static const float SLIT_WIDTH=0.2e-3f;   // Slit width (m)
 static const float SCREEN_Z = 1.0f;      // Distance to screen (m)
 static const float XRANGE   = 0.02f;     // Max +/- x-range on the screen to sample (m)
 
 // Number of points ("photons") to draw each frame
 static const size_t NUM_POINTS = 100000;
 
 // GPU block size for convenience
 static const int BLOCK_SIZE = 256;
 
 // ----------------------------------------------------------
 // Simple double-slit intensity function (far-field approximation)
 // I(x) = I0 * [cos^2( (π·d·x)/(λ·z) )] * [sinc^2( (π·a·x)/(λ·z) )]
 // For simplicity: let I0 = 1.0 (we only need relative shape)
 // ----------------------------------------------------------
 
 // A small helper for the squared "sinc" in terms of sin(x)/x
 __device__ __inline__
 float sinc2f(float x)
 {
     // In many contexts, sinc(x) = sin(x)/x; handle x ~ 0 carefully
     if(fabsf(x) < 1.0e-7f) 
         return 1.0f; // limit of (sin x / x)^2 as x->0 is 1
     float val = sinf(x)/x;
     return val*val;
 }
 
 __device__ __inline__
 float doubleSlitIntensity(float x, float wavelength, float d, float a, float z)
 {
     // Convert x to "angle" approximation: theta ~ x/z, but let's do direct
     // in the Fraunhofer formula:
     // I ~ cos^2( (π·d·x)/(λ·z) ) * sinc^2( (π·a·x)/(λ·z) )
     
     float alpha   = M_PI * d * x / (wavelength * z);
     float beta    = M_PI * a * x / (wavelength * z);
     
     float cosTerm = cosf(alpha);
     float cos2    = cosTerm*cosTerm;
     float s2      = sinc2f(beta);
     return cos2 * s2; // I0 = 1
 }
 
 // ----------------------------------------------------------
 // Device kernel that, for each thread, generates exactly one "photon" x-pos
 // according to the double-slit distribution. A rejection-sampling approach:
 //    1. Propose x in [-XRANGE, +XRANGE], uniform
 //    2. Compute I(x), if a uniform draw < I(x)/Imax, accept. Else repeat
 // We'll store the final accepted x in the VBO, y=0 (or small random).
 // ----------------------------------------------------------
 
 __global__
 void setupCurandStates(curandState *states, unsigned long long seed, int n)
 {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (idx < n) {
         // Each state gets a unique seed sequence
         curand_init(seed, idx, 0, &states[idx]);
     }
 }
 
 __global__
 void generateDoubleSlitPhotons(
     float2* pos,          // output positions
     curandState* states,  // RNG states
     int n,                // number of points
     float wavelength,
     float slitDistance,
     float slitWidth,
     float screenZ,
     float xRange,
     float Imax            // maximum possible intensity for acceptance
 )
 {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (idx >= n) return;
 
     curandState localState = states[idx];
     
     float x, y, I;
     bool accepted = false;
     
     // We'll do a small upper bound on tries to avoid infinite loops
     // (this is rarely an issue for well-chosen Imax)
     for(int attempts=0; attempts < 1000 && !accepted; attempts++) {
         // Propose x in [-xRange, +xRange]
         x = -xRange + 2.0f * xRange * curand_uniform(&localState);
 
         // Evaluate intensity
         I = doubleSlitIntensity(x, wavelength, slitDistance, slitWidth, screenZ);
 
         // Draw a uniform random number for acceptance
         float testVal = curand_uniform(&localState);
         if(testVal < (I / Imax)) {
             accepted = true;
         }
     }
 
     // For demonstration, let’s put the photon at (x, random small Y) to see
     // a bit of vertical spread. Alternatively, just store y=0.0f if you like.
     // Or use another random approach to add "vertical scatter."
     y = -0.01f + 0.02f * curand_uniform(&localState);
 
     pos[idx] = make_float2(x, y);
 
     // Store back the updated RNG state
     states[idx] = localState;
 }
 
 // ----------------------------------------------------------
 // Resize Callback
 // ----------------------------------------------------------
 void framebuffer_size_callback(GLFWwindow* window, int width, int height)
 {
     glViewport(0, 0, width, height);
 }
 
 // ----------------------------------------------------------
 // Main
 // ----------------------------------------------------------
 int main()
 {
     // 1. Initialize GLFW
     if (!glfwInit()) {
         std::cerr << "Failed to initialize GLFW\n";
         return -1;
     }
     
     GLFWwindow* window = glfwCreateWindow(800, 600, "CUDA-GL Double Slit", nullptr, nullptr);
     if (!window) {
         std::cerr << "Failed to create GLFW window\n";
         glfwTerminate();
         return -1;
     }
     glfwMakeContextCurrent(window);
     glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
 
     // 2. Initialize GLEW
     if (glewInit() != GLEW_OK) {
         std::cerr << "Failed to initialize GLEW\n";
         glfwDestroyWindow(window);
         glfwTerminate();
         return -1;
     }
 
     // Set up viewport
     int width, height;
     glfwGetFramebufferSize(window, &width, &height);
     glViewport(0, 0, width, height);
 
     // 3. Create VBO
     GLuint vbo;
     glGenBuffers(1, &vbo);
     glBindBuffer(GL_ARRAY_BUFFER, vbo);
     size_t bufferSize = NUM_POINTS * 2 * sizeof(float);
     glBufferData(GL_ARRAY_BUFFER, bufferSize, nullptr, GL_DYNAMIC_DRAW);
     glBindBuffer(GL_ARRAY_BUFFER, 0);
 
     // 4. Register VBO with CUDA
     cudaGraphicsResource* cudaVboResource;
     cudaGraphicsGLRegisterBuffer(&cudaVboResource, vbo, cudaGraphicsMapFlagsWriteDiscard);
 
     // 5. Setup CURAND states on the GPU
     curandState* d_rngStates = nullptr;
     cudaMalloc((void**)&d_rngStates, NUM_POINTS * sizeof(curandState));
 
     // Choose a global seed
     unsigned long long seed = 1234ULL;
     int grid = (NUM_POINTS + BLOCK_SIZE - 1) / BLOCK_SIZE;
     setupCurandStates<<<grid, BLOCK_SIZE>>>(d_rngStates, seed, NUM_POINTS);
     cudaDeviceSynchronize();
 
     // Precompute the maximum intensity in the range [-XRANGE, +XRANGE].
     // For a well-known formula, the maximum is at x=0 for symmetrical double-slit
     // That max is I(0) = 1 (since cos^2(0)=1, sinc^2(0)=1). So Imax = 1.0
     float Imax = 1.0f;
 
     // 6. Main Loop
     while (!glfwWindowShouldClose(window))
     {
         // Map the buffer so CUDA can write to it
         cudaGraphicsMapResources(1, &cudaVboResource, 0);
         void* dPtr = nullptr;
         size_t dSize = 0;
         cudaGraphicsResourceGetMappedPointer(&dPtr, &dSize, cudaVboResource);
 
         // Fill the buffer with double-slit distributed points
         generateDoubleSlitPhotons<<<grid, BLOCK_SIZE>>>(
             (float2*)dPtr,
             d_rngStates,
             NUM_POINTS,
             LAMBDA,
             SLIT_DIST,
             SLIT_WIDTH,
             SCREEN_Z,
             XRANGE,
             Imax
         );
         cudaDeviceSynchronize();
 
         // Unmap so OpenGL can use it
         cudaGraphicsUnmapResources(1, &cudaVboResource, 0);
 
         // Render
         glClear(GL_COLOR_BUFFER_BIT);
 
         // For a simple 2D rendering, we might want to set up an orthographic projection
         // that covers about [-XRANGE, +XRANGE] in x. Let's do a quick raw glOrtho:
         glMatrixMode(GL_PROJECTION);
         glLoadIdentity();
         // A little margin in vertical as well
         glOrtho(-XRANGE*1.1, XRANGE*1.1, -0.05, 0.05, -1.0, 1.0);
 
         glMatrixMode(GL_MODELVIEW);
         glLoadIdentity();
 
         // Bind VBO and render
         glBindBuffer(GL_ARRAY_BUFFER, vbo);
         glEnableClientState(GL_VERTEX_ARRAY);
         glVertexPointer(2, GL_FLOAT, 0, 0);
 
         // Draw
         glDrawArrays(GL_POINTS, 0, NUM_POINTS);
 
         glDisableClientState(GL_VERTEX_ARRAY);
         glBindBuffer(GL_ARRAY_BUFFER, 0);
 
         glfwSwapBuffers(window);
         glfwPollEvents();
 
         // Optional: if you want to exit on Esc
         if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
             glfwSetWindowShouldClose(window, 1);
         }
     }
 
     // 7. Cleanup
     cudaFree(d_rngStates);
     cudaGraphicsUnregisterResource(cudaVboResource);
     glDeleteBuffers(1, &vbo);
 
     glfwDestroyWindow(window);
     glfwTerminate();
 
     return 0;
 }
 