/******************************************************************************
 *  DoubleSlitWithShaderPanZoomIntensityOverlay.cu
 *
 *  A CUDA + OpenGL program demonstrating a double-slit interference pattern
 *  with physically meaningful colors (via a GLSL shader), interactive pan/zoom,
 *  adjustable intensity boost, and on-screen overlay instructions using
 *  stb_easy_font.
 *
 *  Controls:
 *    - Arrow keys or Mouse drag: Pan the view.
 *    - Mouse wheel: Zoom in/out.
 *    - I/K keys: Increase/Decrease intensity boost.
 *    - ESC: Exit.
 *
 *  Each frame generates NUM_POINTS "photons" via rejection sampling.
 ******************************************************************************/

 #include <iostream>
 #include <cmath>
 #include <cstdio>
 #include <GL/glew.h>
 #include <GLFW/glfw3.h>
 
 #include <cuda_runtime.h>
 #include <curand_kernel.h>
 #include <cuda_gl_interop.h>
 
 // Include stb_easy_font for text overlay
 #define STB_EASY_FONT_IMPLEMENTATION
 #include "stb_easy_font.h"
 
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
 // Global variables for pan/zoom, intensity boost, and window size
 // ----------------------------------------------------------
 float panX = 0.0f;
 float panY = 0.0f;
 float zoom = 1.0f;             // Higher zoom => closer view (smaller visible region)
 float intensityBoost = 1.0f;   // Multiplier to brighten dark areas
 
 int windowWidth = 800;
 int windowHeight = 600;
 
 // Variables for mouse dragging to pan
 bool mouseDragging = false;
 double lastMouseX = 0.0, lastMouseY = 0.0;
 
 // ----------------------------------------------------------
 // Device functions: double-slit intensity calculation
 // ----------------------------------------------------------
 __device__ __inline__
 float sinc2f(float x)
 {
     if (fabsf(x) < 1.0e-7f)
         return 1.0f;
     float val = sinf(x)/x;
     return val * val;
 }
 
 __device__ __inline__
 float doubleSlitIntensity(float x, float wavelength, float d, float a, float z)
 {
     float alpha = M_PI * d * x / (wavelength * z);
     float beta  = M_PI * a * x / (wavelength * z);
     return cosf(alpha) * cosf(alpha) * sinc2f(beta);
 }
 
 // ----------------------------------------------------------
 // CUDA Kernels for photon generation using rejection sampling
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
     if (idx >= n)
         return;
 
     curandState localState = states[idx];
     float x, y, I;
     bool accepted = false;
     for (int attempts = 0; attempts < 1000 && !accepted; attempts++) {
         x = -xRange + 2.0f * xRange * curand_uniform(&localState);
         I = doubleSlitIntensity(x, wavelength, slitDistance, slitWidth, screenZ);
         float testVal = curand_uniform(&localState);
         if (testVal < (I / Imax))
             accepted = true;
     }
     y = -0.01f + 0.02f * curand_uniform(&localState);
     pos[idx] = make_float2(x, y);
     states[idx] = localState;
 }
 
 // ----------------------------------------------------------
 // GLSL Shader Sources (version 120 for compatibility)
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
 uniform float intensityBoost;  // Multiplier for computed intensity
 
 float sinc2(float x) {
     if (abs(x) < 1e-7)
         return 1.0;
     float s = sin(x) / x;
     return s * s;
 }
  
 float doubleSlitIntensity(float x) {
     float alpha = 3.14159265359 * slitDistance * x / (wavelength * screenZ);
     float beta  = 3.14159265359 * slitWidth * x / (wavelength * screenZ);
     return cos(alpha)*cos(alpha) * sinc2(beta);
 }
  
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
     float intensity = doubleSlitIntensity(fragPos.x) / Imax;
     intensity = clamp(intensity * intensityBoost, 0.0, 1.0);
     vec3 baseColor = wavelengthToRGB(wavelength);
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
     if (!vs || !fs)
         return 0;
     GLuint program = glCreateProgram();
     glAttachShader(program, vs);
     glAttachShader(program, fs);
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
     glDeleteShader(vs);
     glDeleteShader(fs);
     return program;
 }
 
 // ----------------------------------------------------------
 // GLFW callbacks for interaction
 // ----------------------------------------------------------
 
 // Scroll callback for zooming
 void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
 {
     float zoomFactor = 1.1f;
     if (yoffset > 0)
         zoom *= zoomFactor;
     else if (yoffset < 0)
         zoom /= zoomFactor;
 }
 
 // Framebuffer resize callback updates viewport and window size globals
 void framebuffer_size_callback(GLFWwindow* window, int width, int height)
 {
     windowWidth = width;
     windowHeight = height;
     glViewport(0, 0, width, height);
 }
 
 // Mouse button callback to start/end dragging for panning
 void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
 {
     if (button == GLFW_MOUSE_BUTTON_LEFT) {
         if (action == GLFW_PRESS) {
             mouseDragging = true;
             glfwGetCursorPos(window, &lastMouseX, &lastMouseY);
         } else if (action == GLFW_RELEASE) {
             mouseDragging = false;
         }
     }
 }
 
 // Cursor position callback for mouse-drag panning
 void cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
 {
     if (mouseDragging) {
         double dx = xpos - lastMouseX;
         double dy = ypos - lastMouseY;
         lastMouseX = xpos;
         lastMouseY = ypos;
         // Compute world-space offset based on current zoom and window size.
         float worldWidth = (XRANGE * 1.1f * 2.0f) / zoom;
         float worldHeight = (0.05f * 2.0f) / zoom;
         panX -= dx * (worldWidth / windowWidth);
         panY += dy * (worldHeight / windowHeight);
     }
 }
 
 // ----------------------------------------------------------
 // Render overlay text using stb_easy_font
 // ----------------------------------------------------------
 void renderTextOverlay()
 {
     // Save current matrices and switch to orthographic projection in window space.
     glMatrixMode(GL_PROJECTION);
     glPushMatrix();
     glLoadIdentity();
     glOrtho(0, windowWidth, windowHeight, 0, -1, 1);
     glMatrixMode(GL_MODELVIEW);
     glPushMatrix();
     glLoadIdentity();
     
     // Prepare overlay text
     char info[256];
     sprintf(info, "Controls: Arrow keys / Mouse drag to pan, Mouse wheel to zoom, I/K to change intensity, ESC to exit.");
     
     char status[128];
     sprintf(status, "Zoom: %.2f  Intensity Boost: %.2f", zoom, intensityBoost);
     
     char overlay[512];
     snprintf(overlay, sizeof(overlay), "%s\n%s", info, status);
     
     // Use stb_easy_font to generate vertex data for the text.
     char buffer[99999];
     int num_quads = stb_easy_font_print(10, windowHeight - 20, overlay, NULL, buffer, sizeof(buffer));
     
     glColor3f(1.0f, 1.0f, 1.0f);
     glEnableClientState(GL_VERTEX_ARRAY);
     glVertexPointer(2, GL_FLOAT, 16, buffer);
     glDrawArrays(GL_QUADS, 0, num_quads * 4);
     glDisableClientState(GL_VERTEX_ARRAY);
     
     // Restore previous matrices.
     glPopMatrix();
     glMatrixMode(GL_PROJECTION);
     glPopMatrix();
     glMatrixMode(GL_MODELVIEW);
 }
 
 // ----------------------------------------------------------
 // Main function
 // ----------------------------------------------------------
 int main()
 {
     if (!glfwInit()) {
         std::cerr << "Failed to initialize GLFW\n";
         return -1;
     }
     
     GLFWwindow* window = glfwCreateWindow(800, 600, "Quantum Photonics Double Slit Experiment", nullptr, nullptr);
     if (!window) {
         std::cerr << "Failed to create GLFW window\n";
         glfwTerminate();
         return -1;
     }
     glfwMakeContextCurrent(window);
     
     glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
     glfwSetScrollCallback(window, scroll_callback);
     glfwSetMouseButtonCallback(window, mouse_button_callback);
     glfwSetCursorPosCallback(window, cursor_position_callback);
     
     if (glewInit() != GLEW_OK) {
         std::cerr << "Failed to initialize GLEW\n";
         glfwDestroyWindow(window);
         glfwTerminate();
         return -1;
     }
     
     glfwGetFramebufferSize(window, &windowWidth, &windowHeight);
     glViewport(0, 0, windowWidth, windowHeight);
     
     // Create a VBO for photon positions.
     GLuint vbo;
     glGenBuffers(1, &vbo);
     glBindBuffer(GL_ARRAY_BUFFER, vbo);
     size_t bufferSize = NUM_POINTS * 2 * sizeof(float);
     glBufferData(GL_ARRAY_BUFFER, bufferSize, nullptr, GL_DYNAMIC_DRAW);
     glBindBuffer(GL_ARRAY_BUFFER, 0);
     
     // Register the VBO with CUDA.
     cudaGraphicsResource* cudaVboResource;
     cudaGraphicsGLRegisterBuffer(&cudaVboResource, vbo, cudaGraphicsMapFlagsWriteDiscard);
     
     // Set up CURAND states on the GPU.
     curandState* d_rngStates = nullptr;
     cudaMalloc((void**)&d_rngStates, NUM_POINTS * sizeof(curandState));
     
     unsigned long long seed = 1234ULL;
     int grid = (NUM_POINTS + BLOCK_SIZE - 1) / BLOCK_SIZE;
     setupCurandStates<<<grid, BLOCK_SIZE>>>(d_rngStates, seed, NUM_POINTS);
     cudaDeviceSynchronize();
     
     float Imax = 1.0f; // Maximum intensity at x = 0
     
     GLuint shaderProgram = createShaderProgram();
     if (!shaderProgram) {
         std::cerr << "Failed to create shader program.\n";
         return -1;
     }
     
     // Main loop.
     while (!glfwWindowShouldClose(window))
     {
         glfwPollEvents();
         
         // Also allow keyboard pan.
         float panSpeed = 0.0005f / zoom;
         if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
             panX -= panSpeed;
         if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
             panX += panSpeed;
         if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
             panY += panSpeed;
         if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
             panY -= panSpeed;
         
         // Adjust intensity boost with I/K keys.
         if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS)
             intensityBoost += 0.01f;
         if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS)
             intensityBoost = (intensityBoost > 0.02f) ? intensityBoost - 0.01f : 0.01f;
         
         // Generate new photons.
         cudaGraphicsMapResources(1, &cudaVboResource, 0);
         void* dPtr = nullptr;
         size_t dSize = 0;
         cudaGraphicsResourceGetMappedPointer(&dPtr, &dSize, cudaVboResource);
         
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
         cudaGraphicsUnmapResources(1, &cudaVboResource, 0);
         
         glClear(GL_COLOR_BUFFER_BIT);
         
         // Set up the orthographic projection based on pan and zoom.
         glMatrixMode(GL_PROJECTION);
         glLoadIdentity();
         float left = panX - (XRANGE * 1.1f) / zoom;
         float right = panX + (XRANGE * 1.1f) / zoom;
         float bottom = panY - (0.05f) / zoom;
         float top = panY + (0.05f) / zoom;
         glOrtho(left, right, bottom, top, -1.0, 1.0);
         
         glMatrixMode(GL_MODELVIEW);
         glLoadIdentity();
         
         // Render using the shader.
         glUseProgram(shaderProgram);
         glUniform1f(glGetUniformLocation(shaderProgram, "wavelength"), LAMBDA);
         glUniform1f(glGetUniformLocation(shaderProgram, "slitDistance"), SLIT_DIST);
         glUniform1f(glGetUniformLocation(shaderProgram, "slitWidth"), SLIT_WIDTH);
         glUniform1f(glGetUniformLocation(shaderProgram, "screenZ"), SCREEN_Z);
         glUniform1f(glGetUniformLocation(shaderProgram, "Imax"), Imax);
         glUniform1f(glGetUniformLocation(shaderProgram, "intensityBoost"), intensityBoost);
         
         glBindBuffer(GL_ARRAY_BUFFER, vbo);
         glEnableVertexAttribArray(0);
         glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
         
         glDrawArrays(GL_POINTS, 0, NUM_POINTS);
         
         glDisableVertexAttribArray(0);
         glBindBuffer(GL_ARRAY_BUFFER, 0);
         glUseProgram(0);
         
         // Render the overlay text.
         renderTextOverlay();
         
         glfwSwapBuffers(window);
         
         if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
             glfwSetWindowShouldClose(window, 1);
     }
     
     // Cleanup.
     cudaFree(d_rngStates);
     cudaGraphicsUnregisterResource(cudaVboResource);
     glDeleteBuffers(1, &vbo);
     glDeleteProgram(shaderProgram);
     
     glfwDestroyWindow(window);
     glfwTerminate();
     
     return 0;
 }
 