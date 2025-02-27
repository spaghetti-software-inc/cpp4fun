/******************************************************************************
 *  DoubleSlitWithShaderPanZoom.cu
 *
 *  A CUDA + OpenGL program demonstrating a double-slit interference pattern with
 *  physically meaningful colors (via a GLSL shader) and interactive pan/zoom.
 *
 *  Controls:
 *    - Arrow keys: Pan the view.
 *    - Mouse wheel: Zoom in/out.
 *    - Escape: Exit.
 *
 *  Each frame generates NUM_POINTS "photons" via rejection sampling.
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
 static const float LAMBDA    = 0.5e-6f;   // Wavelength (m), e.g. 500 nm
 static const float SLIT_DIST = 1.0e-3f;   // Center-to-center slit separation (m)
 static const float SLIT_WIDTH= 0.2e-3f;    // Slit width (m)
 static const float SCREEN_Z  = 1.0f;       // Distance to screen (m)
 static const float XRANGE    = 0.02f;      // Max +/- x-range on the screen to sample (m)
 
 // Number of points ("photons") to draw each frame
 static const size_t NUM_POINTS = 100000;
 
 // GPU block size for convenience
 static const int BLOCK_SIZE = 256;
 
 // ----------------------------------------------------------
 // Global variables for pan/zoom functionality
 // ----------------------------------------------------------
 float panX = 0.0f;
 float panY = 0.0f;
 float zoom = 1.0f;  // A higher value zooms in (i.e. visible range shrinks)
 
 // ----------------------------------------------------------
 // Simple double-slit intensity function (far-field approximation)
 // I(x) = I0 * [cos^2( (π·d·x)/(λ·z) )] * [sinc^2( (π·a·x)/(λ·z) )]
 // For simplicity: let I0 = 1.0 (we only need relative shape)
 // ----------------------------------------------------------
 __device__ __inline__
 float sinc2f(float x)
 {
     if(fabsf(x) < 1.0e-7f)
         return 1.0f; // limit as x->0
     float val = sinf(x)/x;
     return val*val;
 }
 
 __device__ __inline__
 float doubleSlitIntensity(float x, float wavelength, float d, float a, float z)
 {
     float alpha = M_PI * d * x / (wavelength * z);
     float beta  = M_PI * a * x / (wavelength * z);
     return cosf(alpha)*cosf(alpha) * sinc2f(beta);
 }
 
 // ----------------------------------------------------------
 // Device kernels for Monte Carlo photon generation
 // ----------------------------------------------------------
 __global__
 void setupCurandStates(curandState *states, unsigned long long seed, int n)
 {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (idx < n)
         curand_init(seed, idx, 0, &states[idx]);
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
     for(int attempts=0; attempts < 1000 && !accepted; attempts++) {
         x = -xRange + 2.0f * xRange * curand_uniform(&localState);
         I = doubleSlitIntensity(x, wavelength, slitDistance, slitWidth, screenZ);
         float testVal = curand_uniform(&localState);
         if(testVal < (I / Imax))
             accepted = true;
     }
     y = -0.01f + 0.02f * curand_uniform(&localState);
     pos[idx] = make_float2(x, y);
     states[idx] = localState;
 }
 
 // ----------------------------------------------------------
 // Shader source strings (GLSL version 120 for compatibility)
 // ----------------------------------------------------------
 const char* vertexShaderSource = R"(
 #version 120
 attribute vec2 vertexPosition;
 varying vec2 fragPos;
 void main() {
     fragPos = vertexPosition;
     gl_Position = gl_ModelViewProjectionMatrix * vec4(vertexPosition, 0.0, 1.0);
 }
 )";
 
 const char* fragmentShaderSource = R"(
 #version 120
 varying vec2 fragPos;
 uniform float wavelength;   // in meters
 uniform float slitDistance;
 uniform float slitWidth;
 uniform float screenZ;
 uniform float Imax;
  
 // Compute sinc^2(x) safely
 float sinc2(float x) {
     if (abs(x) < 1e-7)
         return 1.0;
     float s = sin(x) / x;
     return s * s;
 }
  
 // Double-slit intensity function
 float doubleSlitIntensity(float x) {
     float alpha = 3.14159265359 * slitDistance * x / (wavelength * screenZ);
     float beta  = 3.14159265359 * slitWidth * x / (wavelength * screenZ);
     return cos(alpha)*cos(alpha) * sinc2(beta);
 }
  
 // Convert wavelength (in meters) to an approximate RGB value.
 vec3 wavelengthToRGB(float lambda) {
     float nm = lambda * 1e9; // convert to nanometers
     float R, G, B;
     if(nm >= 380.0 && nm < 440.0) {
         R = -(nm - 440.0) / (440.0 - 380.0);
         G = 0.0;
         B = 1.0;
     } else if(nm >= 440.0 && nm < 490.0) {
         R = 0.0;
         G = (nm - 440.0) / (490.0 - 440.0);
         B = 1.0;
     } else if(nm >= 490.0 && nm < 510.0) {
         R = 0.0;
         G = 1.0;
         B = -(nm - 510.0) / (510.0 - 490.0);
     } else if(nm >= 510.0 && nm < 580.0) {
         R = (nm - 510.0) / (580.0 - 510.0);
         G = 1.0;
         B = 0.0;
     } else if(nm >= 580.0 && nm < 645.0) {
         R = 1.0;
         G = -(nm - 645.0) / (645.0 - 580.0);
         B = 0.0;
     } else if(nm >= 645.0 && nm <= 780.0) {
         R = 1.0;
         G = 0.0;
         B = 0.0;
     } else {
         R = 0.0;
         G = 0.0;
         B = 0.0;
     }
     float factor;
     if(nm >= 380.0 && nm < 420.0)
         factor = 0.3 + 0.7*(nm - 380.0) / (420.0 - 380.0);
     else if(nm >= 420.0 && nm <= 700.0)
         factor = 1.0;
     else if(nm > 700.0 && nm <= 780.0)
         factor = 0.3 + 0.7*(780.0 - nm) / (780.0 - 700.0);
     else
         factor = 0.0;
     return vec3(R, G, B) * factor;
 }
  
 void main() {
     // Compute the normalized intensity based on the x-position
     float intensity = doubleSlitIntensity(fragPos.x) / Imax;
     // Get the base color corresponding to the light's wavelength
     vec3 baseColor = wavelengthToRGB(wavelength);
     // Output the final color, modulated by the interference intensity
     gl_FragColor = vec4(baseColor * intensity, 1.0);
 }
 )";
 
 // ----------------------------------------------------------
 // Shader helper functions
 // ----------------------------------------------------------
 GLuint compileShader(GLenum type, const char* source) {
     GLuint shader = glCreateShader(type);
     glShaderSource(shader, 1, &source, nullptr);
     glCompileShader(shader);
     GLint compiled;
     glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
     if (!compiled) {
         GLint len;
         glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
         char* log = new char[len];
         glGetShaderInfoLog(shader, len, &len, log);
         std::cerr << "Shader compilation error:\n" << log << std::endl;
         delete[] log;
         glDeleteShader(shader);
         return 0;
     }
     return shader;
 }
 
 GLuint createShaderProgram() {
     GLuint vs = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
     GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);
     if (!vs || !fs) return 0;
     GLuint program = glCreateProgram();
     glAttachShader(program, vs);
     glAttachShader(program, fs);
     // Bind attribute location for our vertex positions
     glBindAttribLocation(program, 0, "vertexPosition");
     glLinkProgram(program);
     GLint linked;
     glGetProgramiv(program, GL_LINK_STATUS, &linked);
     if (!linked) {
         GLint len;
         glGetProgramiv(program, GL_INFO_LOG_LENGTH, &len);
         char* log = new char[len];
         glGetProgramInfoLog(program, len, &len, log);
         std::cerr << "Shader linking error:\n" << log << std::endl;
         delete[] log;
         glDeleteProgram(program);
         return 0;
     }
     // Shaders can be deleted after linking
     glDeleteShader(vs);
     glDeleteShader(fs);
     return program;
 }
 
 // ----------------------------------------------------------
 // GLFW scroll callback for zooming
 // ----------------------------------------------------------
 void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
 {
     // Adjust zoom factor. Scrolling up zooms in.
     float zoomFactor = 1.1f;
     if (yoffset > 0)
         zoom *= zoomFactor;
     else if (yoffset < 0)
         zoom /= zoomFactor;
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
     
     GLFWwindow* window = glfwCreateWindow(800, 600, "CUDA-GL Double Slit with Shader (Pan/Zoom)", nullptr, nullptr);
     if (!window) {
         std::cerr << "Failed to create GLFW window\n";
         glfwTerminate();
         return -1;
     }
     glfwMakeContextCurrent(window);
     glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
     // Set the scroll callback to handle zooming
     glfwSetScrollCallback(window, scroll_callback);
  
     // 2. Initialize GLEW
     if (glewInit() != GLEW_OK) {
         std::cerr << "Failed to initialize GLEW\n";
         glfwDestroyWindow(window);
         glfwTerminate();
         return -1;
     }
  
     int width, height;
     glfwGetFramebufferSize(window, &width, &height);
     glViewport(0, 0, width, height);
  
     // 3. Create VBO for photon positions
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
  
     unsigned long long seed = 1234ULL;
     int grid = (NUM_POINTS + BLOCK_SIZE - 1) / BLOCK_SIZE;
     setupCurandStates<<<grid, BLOCK_SIZE>>>(d_rngStates, seed, NUM_POINTS);
     cudaDeviceSynchronize();
  
     // Imax is precomputed (maximum intensity at x = 0)
     float Imax = 1.0f;
  
     // 6. Create and set up the shader program
     GLuint shaderProgram = createShaderProgram();
     if (!shaderProgram) {
         std::cerr << "Failed to create shader program.\n";
         return -1;
     }
     
     // 7. Main Loop
     while (!glfwWindowShouldClose(window))
     {
         // Poll events (including key presses)
         glfwPollEvents();
  
         // --- Update pan based on keyboard input ---
         // Adjust pan speed relative to zoom for consistency.
         float panSpeed = 0.0005f / zoom;
         if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
             panX -= panSpeed;
         if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
             panX += panSpeed;
         if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
             panY += panSpeed;
         if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
             panY -= panSpeed;
  
         // Map VBO for CUDA access
         cudaGraphicsMapResources(1, &cudaVboResource, 0);
         void* dPtr = nullptr;
         size_t dSize = 0;
         cudaGraphicsResourceGetMappedPointer(&dPtr, &dSize, cudaVboResource);
  
         // Generate new photons
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
  
         // Unmap so OpenGL can use the buffer
         cudaGraphicsUnmapResources(1, &cudaVboResource, 0);
  
         // Clear the screen
         glClear(GL_COLOR_BUFFER_BIT);
  
         // Update the orthographic projection to incorporate pan and zoom.
         glMatrixMode(GL_PROJECTION);
         glLoadIdentity();
         // The visible region is scaled by 1/zoom and shifted by panX, panY.
         float left = panX - (XRANGE * 1.1f) / zoom;
         float right = panX + (XRANGE * 1.1f) / zoom;
         float bottom = panY - (0.05f) / zoom;
         float top = panY + (0.05f) / zoom;
         glOrtho(left, right, bottom, top, -1.0, 1.0);
  
         glMatrixMode(GL_MODELVIEW);
         glLoadIdentity();
  
         // Use our shader program
         glUseProgram(shaderProgram);
         // Set uniform values for the shader
         glUniform1f(glGetUniformLocation(shaderProgram, "wavelength"), LAMBDA);
         glUniform1f(glGetUniformLocation(shaderProgram, "slitDistance"), SLIT_DIST);
         glUniform1f(glGetUniformLocation(shaderProgram, "slitWidth"), SLIT_WIDTH);
         glUniform1f(glGetUniformLocation(shaderProgram, "screenZ"), SCREEN_Z);
         glUniform1f(glGetUniformLocation(shaderProgram, "Imax"), Imax);
  
         // Bind VBO and set attribute pointer for "vertexPosition"
         glBindBuffer(GL_ARRAY_BUFFER, vbo);
         glEnableVertexAttribArray(0); // attribute location 0 (bound above)
         glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
  
         // Draw the points (photons)
         glDrawArrays(GL_POINTS, 0, NUM_POINTS);
  
         glDisableVertexAttribArray(0);
         glBindBuffer(GL_ARRAY_BUFFER, 0);
  
         // Unbind shader
         glUseProgram(0);
  
         glfwSwapBuffers(window);
  
         // Exit on Esc key
         if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
             glfwSetWindowShouldClose(window, 1);
     }
  
     // 8. Cleanup
     cudaFree(d_rngStates);
     cudaGraphicsUnregisterResource(cudaVboResource);
     glDeleteBuffers(1, &vbo);
     glDeleteProgram(shaderProgram);
  
     glfwDestroyWindow(window);
     glfwTerminate();
  
     return 0;
 }
 