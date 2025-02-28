/******************************************************************************
 *  DoubleSlit_Modern.cpp
 *
 *  Modern C++20 version of a C/CUDA/OpenGL standalone double-slit simulation.
 *  It demonstrates:
 *    - Encapsulation of global state in a class,
 *    - RAII wrappers for GLFW and OpenGL resources,
 *    - CUDA/OpenGL interop with CURAND for generating photons,
 *    - Dynamic accumulation that resets on pan/zoom.
 *
 *  Controls:
 *    - Mouse drag or Arrow keys: Pan (accumulation resets)
 *    - Mouse wheel: Zoom in/out (accumulation resets)
 *    - I/K: Increase/Decrease intensity boost
 *    - C: Clear accumulation manually
 *    - ESC: Exit
 ******************************************************************************/

 #include <iostream>
 #include <cmath>
 #include <cstdio>
 #include <stdexcept>
 #include <array>
 #include <string_view>
 #include <string>
 
 // OpenGL / GLFW
 #include <GL/glew.h>
 #include <GLFW/glfw3.h>
 
 // CUDA and CURAND
 #include <cuda_runtime.h>
 #include <curand_kernel.h>
 #include <cuda_gl_interop.h>
 
 // stb_easy_font for text overlay (header-only)
 #define STB_EASY_FONT_IMPLEMENTATION
 #include "stb_easy_font.h"
 
 // --------------------------------------------------------------------------
 // Simulation parameters (struct used inside the app)
 // --------------------------------------------------------------------------
 struct SimulationParameters {
     const float lambda    = 0.5e-6f;   // wavelength in meters (500nm)
     const float slitDist  = 1.0e-3f;   // distance between slits (1 mm)
     const float slitWidth = 0.2e-3f;   // width of each slit (0.2 mm)
     const float screenZ   = 1.0f;      // distance from slits to screen (1 m)
     const float xRange    = 0.02f;     // half-range for sampling (±2cm)
     static constexpr size_t numPoints = 100000; // photons per frame
     static constexpr int    blockSize = 256;    // CUDA block size
 };
 
 // --------------------------------------------------------------------------
 // Device-side helper functions and kernels
 // --------------------------------------------------------------------------
 __device__ __inline__
 float sinc2f(float x) {
     if (fabsf(x) < 1.0e-7f) return 1.0f;
     float val = sinf(x) / x;
     return val * val;
 }
 
 __device__ __inline__
 float doubleSlitIntensity(float x, float wavelength, float d, float a, float z) {
     float alpha = M_PI * d * x / (wavelength * z);
     float beta  = M_PI * a * x / (wavelength * z);
     return cosf(alpha) * cosf(alpha) * sinc2f(beta);
 }
 
 // CPU-side version used for computing maximum intensity Imax
 float doubleSlitIntensityCPU(float x, float wavelength, float d, float a, float z) {
     float alpha = M_PI * d * x / (wavelength * z);
     float beta  = M_PI * a * x / (wavelength * z);
     float c     = cosf(alpha);
     float val   = c * c;
     float denom = (fabsf(beta) < 1e-7f) ? 1.0f : beta;
     float s     = sinf(denom) / denom;
     return val * (s * s);
 }
 
 float computeMaxIntensity(int sampleCount, const SimulationParameters& params) {
     float maxI = 0.0f;
     for (int i = 0; i < sampleCount; ++i) {
         float frac = static_cast<float>(i) / (sampleCount - 1);
         float x    = -params.xRange + 2.0f * params.xRange * frac;
         float I    = doubleSlitIntensityCPU(x, params.lambda, params.slitDist, params.slitWidth, params.screenZ);
         if (I > maxI) maxI = I;
     }
     return maxI;
 }
 
 // Kernel to set up CURAND states (one per thread)
 __global__
 void setupCurandStates(curandState* states, unsigned long long seed, int n) {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (idx < n)
         curand_init(seed, idx, 0, &states[idx]);
 }
 
 // Kernel to generate double-slit photons via rejection sampling
 __global__
 void generateDoubleSlitPhotons(
     float2*      pos,           // Output: 2D photon positions
     curandState* states,        // CURAND states
     int          n,             // Number of photons to generate
     float        wavelength,    
     float        slitDistance,
     float        slitWidth,
     float        screenZ,
     float        xRange,
     float        Imax           // Maximum intensity for rejection test
 ) {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (idx >= n) return;
     curandState localState = states[idx];
     float x, y, I;
     bool accepted = false;
     for (int attempts = 0; attempts < 1000 && !accepted; ++attempts) {
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
 
 // --------------------------------------------------------------------------
 // Shader source strings (using std::string_view)
 // --------------------------------------------------------------------------
 constexpr std::string_view pointVertexShaderSrc = R"(
 #version 120
 attribute vec2 vertexPosition;
 varying vec2 fragPos;
 void main() {
     fragPos = vertexPosition;
     gl_Position = gl_ModelViewProjectionMatrix * vec4(vertexPosition, 0.0, 1.0);
 }
 )";
 
 constexpr std::string_view pointFragmentShaderSrc = R"(
 #version 120
 varying vec2 fragPos;
 uniform float wavelength;
 uniform float slitDistance;
 uniform float slitWidth;
 uniform float screenZ;
 uniform float Imax;
 uniform float intensityBoost;
 float sinc2(float x) {
     if (abs(x) < 1e-7) return 1.0;
     float s = sin(x)/x;
     return s*s;
 }
 float doubleSlitIntensity(float x) {
     float alpha = 3.14159265359 * slitDistance * x / (wavelength * screenZ);
     float beta  = 3.14159265359 * slitWidth * x / (wavelength * screenZ);
     return cos(alpha)*cos(alpha) * sinc2(beta);
 }
 vec3 wavelengthToRGB(float lambda) {
     float nm = lambda * 1e9;
     float R, G, B;
     if(nm >= 380.0 && nm < 440.0) {
         R = -(nm - 440.0) / (440.0 - 380.0);
         G = 0.0; B = 1.0;
     } else if(nm >= 440.0 && nm < 490.0) {
         R = 0.0; G = (nm - 440.0) / (490.0 - 440.0); B = 1.0;
     } else if(nm >= 490.0 && nm < 510.0) {
         R = 0.0; G = 1.0; B = -(nm - 510.0)/(510.0 - 490.0);
     } else if(nm >= 510.0 && nm < 580.0) {
         R = (nm - 510.0)/(580.0 - 510.0); G = 1.0; B = 0.0;
     } else if(nm >= 580.0 && nm < 645.0) {
         R = 1.0; G = -(nm - 645.0)/(645.0 - 580.0); B = 0.0;
     } else if(nm >= 645.0 && nm <= 780.0) {
         R = 1.0; G = 0.0; B = 0.0;
     } else {
         R = G = B = 0.0;
     }
     float factor;
     if(nm >= 380.0 && nm < 420.0)
         factor = 0.3 + 0.7*(nm - 380.0)/(420.0 - 380.0);
     else if(nm >= 420.0 && nm <= 700.0)
         factor = 1.0;
     else if(nm > 700.0 && nm <= 780.0)
         factor = 0.3 + 0.7*(780.0 - nm)/(780.0 - 700.0);
     else
         factor = 0.0;
     return vec3(R, G, B) * factor;
 }
 void main(){
     float localI = doubleSlitIntensity(fragPos.x) / Imax;
     float val = clamp(localI * intensityBoost, 0.0, 1.0);
     vec3 baseColor = wavelengthToRGB(wavelength);
     gl_FragColor = vec4(baseColor * val, 1.0);
 }
 )";
 
 constexpr std::string_view quadVertexShaderSrc = R"(
 #version 120
 attribute vec2 pos;
 varying vec2 uv;
 void main(){
     uv = (pos * 0.5) + 0.5;
     gl_Position = vec4(pos, 0.0, 1.0);
 }
 )";
 
 // Modified Quad Fragment Shader with HDR tone mapping and gamma correction.
 constexpr std::string_view quadFragmentShaderSrc = R"(
 #version 120
 uniform sampler2D accumTex;
 uniform float exposure;  // New uniform for HDR exposure control
 varying vec2 uv;
 void main(){
     // Sample the accumulated high dynamic range color
     vec4 color = texture2D(accumTex, uv);
     
     // Apply exponential tone mapping (HDR)
     vec3 hdrColor = vec3(1.0) - exp(-color.rgb * exposure);
     
     // Gamma correction (assumes a gamma of 2.2)
     vec3 gammaCorrected = pow(hdrColor, vec3(1.0/2.2));
     
     gl_FragColor = vec4(gammaCorrected, 1.0);
 }
 )";
 
 // --------------------------------------------------------------------------
 // Shader helper functions
 // --------------------------------------------------------------------------
 GLuint compileShader(GLenum type, std::string_view source) {
     GLuint shader = glCreateShader(type);
     const char* src = source.data();
     glShaderSource(shader, 1, &src, nullptr);
     glCompileShader(shader);
     GLint compiled = 0;
     glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
     if (!compiled) {
         GLint len;
         glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
         std::string log(len, '\0');
         glGetShaderInfoLog(shader, len, &len, log.data());
         glDeleteShader(shader);
         throw std::runtime_error("Shader compile error: " + log);
     }
     return shader;
 }
 
 GLuint createShaderProgram(std::string_view vsSource, std::string_view fsSource, const char* attrib0Name) {
     GLuint vs = compileShader(GL_VERTEX_SHADER, vsSource);
     GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsSource);
     GLuint program = glCreateProgram();
     glAttachShader(program, vs);
     glAttachShader(program, fs);
     glBindAttribLocation(program, 0, attrib0Name);
     glLinkProgram(program);
     GLint linked = 0;
     glGetProgramiv(program, GL_LINK_STATUS, &linked);
     if (!linked) {
         GLint len;
         glGetProgramiv(program, GL_INFO_LOG_LENGTH, &len);
         std::string log(len, '\0');
         glGetProgramInfoLog(program, len, &len, log.data());
         glDeleteProgram(program);
         throw std::runtime_error("Shader link error: " + log);
     }
     glDeleteShader(vs);
     glDeleteShader(fs);
     return program;
 }
 
 // --------------------------------------------------------------------------
 // RAII wrapper for GLFW initialization/termination
 // --------------------------------------------------------------------------
 class GLFWContext {
 public:
     GLFWContext() {
         if (!glfwInit())
             throw std::runtime_error("Failed to initialize GLFW");
     }
     ~GLFWContext() { glfwTerminate(); }
 };
 
 // --------------------------------------------------------------------------
 // Main application class
 // --------------------------------------------------------------------------
 class DoubleSlitApp {
 public:
     DoubleSlitApp() 
       : windowWidth(800), windowHeight(600),
         panX(0.0f), panY(0.0f), zoom(1.0f), intensityBoost(1.0f),
         lastPanX(0.0f), lastPanY(0.0f), lastZoom(1.0f),
         exposure(1.0f),  // Default exposure value for HDR
         params{}
     {
         createWindow();
         initGL();
         initCUDA();
         createResources();
         setupCallbacks();
     }
     
     ~DoubleSlitApp() {
         cleanup();
     }
     
     void run() {
         while (!glfwWindowShouldClose(window)) {
             glfwPollEvents();
             processInput();
             updateAccumulationReset();
             float Imax = computeMaxIntensity(2000, params);
             generatePhotons(Imax);
             renderFrame(Imax);
             glfwSwapBuffers(window);
         }
         cudaDeviceSynchronize();
     }
     
 private:
     // GLFW window and dimensions
     GLFWwindow* window = nullptr;
     int windowWidth, windowHeight;
     
     // Camera parameters
     float panX, panY, zoom, intensityBoost;
     float lastPanX, lastPanY, lastZoom;
     bool mouseDragging = false;
     double lastMouseX = 0.0, lastMouseY = 0.0;
     
     // HDR exposure parameter
     float exposure;
     
     // Simulation parameters
     SimulationParameters params;
     
     // OpenGL resource handles
     GLuint vbo = 0;
     GLuint pointShaderProgram = 0;
     GLuint quadShaderProgram = 0;
     GLuint accumFBO = 0;
     GLuint accumTex = 0;
     GLuint quadVAO = 0;
     GLuint quadVBO = 0;
     
     // CUDA resource handles
     cudaGraphicsResource* cudaVboResource = nullptr;
     curandState* d_rngStates = nullptr;
     
     // Create GLFW window
     void createWindow() {
         window = glfwCreateWindow(windowWidth, windowHeight, "Double-Slit (Modern C++20)", nullptr, nullptr);
         if (!window)
             throw std::runtime_error("Failed to create GLFW window");
         glfwMakeContextCurrent(window);
         glfwSetWindowUserPointer(window, this);
     }
     
     // Initialize GLEW and OpenGL viewport
     void initGL() {
         if (glewInit() != GLEW_OK)
             throw std::runtime_error("Failed to initialize GLEW");
         glfwGetFramebufferSize(window, &windowWidth, &windowHeight);
         glViewport(0, 0, windowWidth, windowHeight);
     }
     
     // Initialize CUDA resources and register the VBO with CUDA
     void initCUDA() {
         glGenBuffers(1, &vbo);
         glBindBuffer(GL_ARRAY_BUFFER, vbo);
         size_t bufferSize = params.numPoints * 2 * sizeof(float);
         glBufferData(GL_ARRAY_BUFFER, bufferSize, nullptr, GL_DYNAMIC_DRAW);
         glBindBuffer(GL_ARRAY_BUFFER, 0);
         cudaGraphicsGLRegisterBuffer(&cudaVboResource, vbo, cudaGraphicsMapFlagsWriteDiscard);
         cudaMalloc(reinterpret_cast<void**>(&d_rngStates), params.numPoints * sizeof(curandState));
         int grid = (params.numPoints + SimulationParameters::blockSize - 1) / SimulationParameters::blockSize;
         setupCurandStates<<<grid, SimulationParameters::blockSize>>>(d_rngStates, 1234ULL, params.numPoints);
         cudaDeviceSynchronize();
     }
     
     // Create shaders, FBO, texture, and quad geometry
     void createResources() {
         pointShaderProgram = createShaderProgram(pointVertexShaderSrc, pointFragmentShaderSrc, "vertexPosition");
         quadShaderProgram  = createShaderProgram(quadVertexShaderSrc, quadFragmentShaderSrc, "pos");
         
         glGenFramebuffers(1, &accumFBO);
         glBindFramebuffer(GL_FRAMEBUFFER, accumFBO);
         glGenTextures(1, &accumTex);
         glBindTexture(GL_TEXTURE_2D, accumTex);
         glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, windowWidth, windowHeight, 0,
                      GL_RGBA, GL_FLOAT, nullptr);
         glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
         glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
         glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, accumTex, 0);
         if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
             std::cerr << "Accumulation FBO incomplete!" << std::endl;
         glClearColor(0, 0, 0, 0);
         glClear(GL_COLOR_BUFFER_BIT);
         glBindFramebuffer(GL_FRAMEBUFFER, 0);
         
         glGenVertexArrays(1, &quadVAO);
         glBindVertexArray(quadVAO);
         std::array<GLfloat, 8> fsQuadVerts = { -1.f, -1.f, 1.f, -1.f, -1.f, 1.f, 1.f, 1.f };
         glGenBuffers(1, &quadVBO);
         glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
         glBufferData(GL_ARRAY_BUFFER, fsQuadVerts.size() * sizeof(GLfloat), fsQuadVerts.data(), GL_STATIC_DRAW);
         glEnableVertexAttribArray(0);
         glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
         glBindVertexArray(0);
     }
     
     // Set up GLFW callbacks (framebuffer size, scroll, mouse button, and cursor position)
     void setupCallbacks() {
         glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
         glfwSetScrollCallback(window, scroll_callback);
         glfwSetMouseButtonCallback(window, mouse_button_callback);
         glfwSetCursorPosCallback(window, cursor_position_callback);
     }
     
     // Process keyboard input: pan with arrow keys, adjust intensity, clear accumulation, exit.
     void processInput() {
         float panSpeed = 0.0005f / zoom;
         if (glfwGetKey(window, GLFW_KEY_LEFT)  == GLFW_PRESS) panX -= panSpeed;
         if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) panX += panSpeed;
         if (glfwGetKey(window, GLFW_KEY_UP)    == GLFW_PRESS) panY += panSpeed;
         if (glfwGetKey(window, GLFW_KEY_DOWN)  == GLFW_PRESS) panY -= panSpeed;
         if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS)
             intensityBoost += 0.01f;
         if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS)
             intensityBoost = (intensityBoost > 0.02f) ? (intensityBoost - 0.01f) : 0.01f;
         if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS) {
             glBindFramebuffer(GL_FRAMEBUFFER, accumFBO);
             glClear(GL_COLOR_BUFFER_BIT);
             glBindFramebuffer(GL_FRAMEBUFFER, 0);
         }
         if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
             glfwSetWindowShouldClose(window, GLFW_TRUE);
     }
     
     // Reset accumulation texture if pan or zoom has changed
     void updateAccumulationReset() {
         float panDeltaX = std::fabs(panX - lastPanX);
         float panDeltaY = std::fabs(panY - lastPanY);
         if (panDeltaX > 1e-7f || panDeltaY > 1e-7f) {
             glBindFramebuffer(GL_FRAMEBUFFER, accumFBO);
             glClear(GL_COLOR_BUFFER_BIT);
             glBindFramebuffer(GL_FRAMEBUFFER, 0);
             lastPanX = panX; lastPanY = panY;
         }
         if (std::fabs(zoom - lastZoom) > 1e-7f) {
             glBindFramebuffer(GL_FRAMEBUFFER, accumFBO);
             glClear(GL_COLOR_BUFFER_BIT);
             glBindFramebuffer(GL_FRAMEBUFFER, 0);
             lastZoom = zoom;
         }
     }
     
     // Map the VBO, launch the CUDA kernel to generate photons, and unmap
     void generatePhotons(float Imax) {
         int grid = (params.numPoints + SimulationParameters::blockSize - 1) / SimulationParameters::blockSize;
         cudaGraphicsMapResources(1, &cudaVboResource, 0);
         void* dPtr = nullptr;
         size_t dSize = 0;
         cudaGraphicsResourceGetMappedPointer(&dPtr, &dSize, cudaVboResource);
         generateDoubleSlitPhotons<<<grid, SimulationParameters::blockSize>>>(
             reinterpret_cast<float2*>(dPtr), d_rngStates, params.numPoints,
             params.lambda, params.slitDist, params.slitWidth, params.screenZ,
             params.xRange, Imax);
         cudaDeviceSynchronize();
         cudaGraphicsUnmapResources(1, &cudaVboResource, 0);
     }
     
     // Render the frame in two passes:
     //   1) Render points into the accumulation FBO using the point shader.
     //   2) Draw a fullscreen quad sampling the accumulation texture with HDR tone mapping.
     // Also overlay text using stb_easy_font.
     void renderFrame(float Imax) {
         // Pass 1: Render to accumulation FBO
         glBindFramebuffer(GL_FRAMEBUFFER, accumFBO);
         glViewport(0, 0, windowWidth, windowHeight);
         glEnable(GL_BLEND);
         glBlendFunc(GL_ONE, GL_ONE);
         glMatrixMode(GL_PROJECTION);
         glLoadIdentity();
         float left   = panX - (params.xRange * 1.1f) / zoom;
         float right  = panX + (params.xRange * 1.1f) / zoom;
         float bottom = panY - (0.05f) / zoom;
         float top    = panY + (0.05f) / zoom;
         glOrtho(left, right, bottom, top, -1.0, 1.0);
         glMatrixMode(GL_MODELVIEW);
         glLoadIdentity();
         glUseProgram(pointShaderProgram);
         glUniform1f(glGetUniformLocation(pointShaderProgram, "wavelength"), params.lambda);
         glUniform1f(glGetUniformLocation(pointShaderProgram, "slitDistance"), params.slitDist);
         glUniform1f(glGetUniformLocation(pointShaderProgram, "slitWidth"), params.slitWidth);
         glUniform1f(glGetUniformLocation(pointShaderProgram, "screenZ"), params.screenZ);
         glUniform1f(glGetUniformLocation(pointShaderProgram, "Imax"), Imax);
         glUniform1f(glGetUniformLocation(pointShaderProgram, "intensityBoost"), intensityBoost);
         glBindBuffer(GL_ARRAY_BUFFER, vbo);
         glEnableVertexAttribArray(0);
         glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
         glDrawArrays(GL_POINTS, 0, params.numPoints);
         glDisableVertexAttribArray(0);
         glBindBuffer(GL_ARRAY_BUFFER, 0);
         glUseProgram(0);
         glDisable(GL_BLEND);
         glBindFramebuffer(GL_FRAMEBUFFER, 0);
         
         // Pass 2: Render accumulation texture to screen via a fullscreen quad with HDR tone mapping and gamma correction
         glViewport(0, 0, windowWidth, windowHeight);
         glClear(GL_COLOR_BUFFER_BIT);
         glUseProgram(quadShaderProgram);
         glActiveTexture(GL_TEXTURE0);
         glBindTexture(GL_TEXTURE_2D, accumTex);
         glUniform1i(glGetUniformLocation(quadShaderProgram, "accumTex"), 0);
         glUniform1f(glGetUniformLocation(quadShaderProgram, "exposure"), exposure);
         glBindVertexArray(quadVAO);
         glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
         glBindVertexArray(0);
         glUseProgram(0);
         
         // Overlay text using stb_easy_font
         renderTextOverlay();
     }
     
     // Render overlay text showing controls and parameters
     void renderTextOverlay() {
         glMatrixMode(GL_PROJECTION);
         glPushMatrix();
         glLoadIdentity();
         glOrtho(0, windowWidth, windowHeight, 0, -1, 1);
         glMatrixMode(GL_MODELVIEW);
         glPushMatrix();
         glLoadIdentity();
         char info[512];
         std::snprintf(info, sizeof(info),
             "Controls:\n"
             " Pan: Arrow keys / Mouse drag => resets accumulation\n"
             " Zoom: Mouse wheel => resets accumulation\n"
             " I/K = intensity boost +/-\n"
             " C = clear accumulation\n"
             " ESC = Quit\n"
             "Params:\n"
             " Zoom: %.2f  IntBoost: %.2f\n"
             " Exposure: %.2f\n"
             " lambda=%.3g m, dist=%.3g m, width=%.3g m, screenZ=%.3g m\n"
             " panX=%.4f, panY=%.4f",
             zoom, intensityBoost, exposure,
             params.lambda, params.slitDist, params.slitWidth, params.screenZ, panX, panY);
         char buffer[99999];
         int num_quads = stb_easy_font_print(10, 10, info, nullptr, buffer, sizeof(buffer));
         glColor3f(1, 1, 1);
         glEnableClientState(GL_VERTEX_ARRAY);
         glVertexPointer(2, GL_FLOAT, 16, buffer);
         glDrawArrays(GL_QUADS, 0, num_quads * 4);
         glDisableClientState(GL_VERTEX_ARRAY);
         glPopMatrix();
         glMatrixMode(GL_PROJECTION);
         glPopMatrix();
         glMatrixMode(GL_MODELVIEW);
     }
     
     // Cleanup all allocated resources.
     void cleanup() {
         if (d_rngStates) cudaFree(d_rngStates);
         if (cudaVboResource) cudaGraphicsUnregisterResource(cudaVboResource);
         if (vbo) glDeleteBuffers(1, &vbo);
         if (pointShaderProgram) glDeleteProgram(pointShaderProgram);
         if (quadShaderProgram) glDeleteProgram(quadShaderProgram);
         if (accumTex) glDeleteTextures(1, &accumTex);
         if (accumFBO) glDeleteFramebuffers(1, &accumFBO);
         if (quadVAO) glDeleteVertexArrays(1, &quadVAO);
         if (quadVBO) glDeleteBuffers(1, &quadVBO);
         if (window) {
             glfwDestroyWindow(window);
             window = nullptr;
         }
     }
     
     // Static GLFW callback wrappers forward events to the instance
     static void framebuffer_size_callback(GLFWwindow* win, int width, int height) {
         auto app = static_cast<DoubleSlitApp*>(glfwGetWindowUserPointer(win));
         app->windowWidth = width;
         app->windowHeight = height;
         glViewport(0, 0, width, height);
     }
     
     static void scroll_callback(GLFWwindow* win, double, double yoffset) {
         auto app = static_cast<DoubleSlitApp*>(glfwGetWindowUserPointer(win));
         float zoomFactor = 1.1f;
         if (yoffset > 0) app->zoom *= zoomFactor;
         else if (yoffset < 0) app->zoom /= zoomFactor;
     }
     
     static void mouse_button_callback(GLFWwindow* win, int button, int action, int) {
         auto app = static_cast<DoubleSlitApp*>(glfwGetWindowUserPointer(win));
         if (button == GLFW_MOUSE_BUTTON_LEFT) {
             if (action == GLFW_PRESS) {
                 app->mouseDragging = true;
                 glfwGetCursorPos(win, &app->lastMouseX, &app->lastMouseY);
             } else if (action == GLFW_RELEASE) {
                 app->mouseDragging = false;
             }
         }
     }
     
     static void cursor_position_callback(GLFWwindow* win, double xpos, double ypos) {
         auto app = static_cast<DoubleSlitApp*>(glfwGetWindowUserPointer(win));
         if (app->mouseDragging) {
             double dx = xpos - app->lastMouseX;
             double dy = ypos - app->lastMouseY;
             app->lastMouseX = xpos;
             app->lastMouseY = ypos;
             float worldWidth  = (app->params.xRange * 1.1f * 2.0f) / app->zoom;
             float worldHeight = (0.05f * 2.0f) / app->zoom;
             app->panX -= dx * (worldWidth / app->windowWidth);
             app->panY += dy * (worldHeight / app->windowHeight);
         }
     }
 };
 
 int main() {
     try {
         GLFWContext glfwContext;
         DoubleSlitApp app;
         app.run();
     } catch (const std::exception& ex) {
         std::cerr << "Fatal error: " << ex.what() << std::endl;
         return -1;
     }
     return 0;
 }
 