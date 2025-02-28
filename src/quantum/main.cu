/******************************************************************************
 *  DoubleSlit.cu
 *
 *  A single-file program that demonstrates s double-slit simulation
 *  using modern C++/CUDA/OpenGL with clear separation of concerns:
 *
 *    - GLFWWindow:         Handles window creation and input events.
 *    - DoubleSlitSimulation:Contains domain-specific (physics) logic.
 *    - CudaSimulator:      Manages CUDA initialization and photon generation.
 *    - DoubleSlitRenderer: Handles OpenGL rendering (shaders, FBO, text overlay).
 *    - DoubleSlitApp:      Ties all subsystems together.
 *
 *  Build with nvcc, linking against OpenGL, GLFW, CUDA, and CURAND libraries.
 ******************************************************************************/

 #include <iostream>
 #include <stdexcept>
 #include <cmath>
 #include <string>
 #include <array>
 #include <sstream>
 #include <cstring>
 
 #include <GL/glew.h>
 #include <GLFW/glfw3.h>
 
 #include <cuda_runtime.h>
 #include <curand_kernel.h>
 #include <cuda_gl_interop.h>
 
 #define STB_EASY_FONT_IMPLEMENTATION
 #include "stb_easy_font.h"
 
 // --------------------------------------------------------------------------
 // Simulation parameters
 // --------------------------------------------------------------------------
 struct SimulationParameters {
     float lambda    = 0.5e-6f;   // wavelength (m)
     float slitDist  = 1.0e-3f;   // distance between slits (m)
     float slitWidth = 0.2e-3f;   // width of each slit (m)
     float screenZ   = 1.0f;      // distance from slits to screen (m)
     float xRange    = 0.02f;     // half-range for sampling (±2 cm)
     static constexpr size_t numPoints = 100000; // photons per frame
     static constexpr int    blockSize = 256;    // CUDA block size
 };
 
 // --------------------------------------------------------------------------
 // Domain Simulation: double-slit intensity functions and Imax calculation
 // --------------------------------------------------------------------------
 class DoubleSlitSimulation {
 public:
     explicit DoubleSlitSimulation(const SimulationParameters& p)
         : params(p) {}
 
     float computeMaxIntensity(int sampleCount) {
         float maxI = 0.0f;
         for (int i = 0; i < sampleCount; ++i) {
             float frac = static_cast<float>(i) / (sampleCount - 1);
             float x = -params.xRange + 2.0f * params.xRange * frac;
             float I = doubleSlitIntensityCPU(x);
             if (I > maxI) maxI = I;
         }
         return maxI;
     }
 
     float doubleSlitIntensityCPU(float x) const {
         float alpha = M_PI * params.slitDist * x / (params.lambda * params.screenZ);
         float beta  = M_PI * params.slitWidth * x / (params.lambda * params.screenZ);
         float c     = std::cos(alpha);
         float val   = c * c;
         float denom = (std::fabs(beta) < 1e-7f) ? 1.0f : beta;
         float s     = std::sin(denom) / denom;
         return val * (s * s);
     }
 
     const SimulationParameters& getParams() const { return params; }
 
 private:
     SimulationParameters params;
 };
 
 // --------------------------------------------------------------------------
 // CUDA Kernels (free functions)
 // --------------------------------------------------------------------------
 __global__
 void setupCurandStates(curandState* states, unsigned long long seed, int n) {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (idx < n) {
         curand_init(seed, idx, 0, &states[idx]);
     }
 }
 
 __device__
 inline float sinc2f(float x) {
     if (fabsf(x) < 1.0e-7f) return 1.0f;
     float val = sinf(x) / x;
     return val * val;
 }
 
 __device__
 inline float doubleSlitIntensity(float x, float wavelength, float d, float a, float z) {
     float alpha = M_PI * d * x / (wavelength * z);
     float beta  = M_PI * a * x / (wavelength * z);
     float c = cosf(alpha);
     return c * c * sinc2f(beta);
 }
 
 __global__
 void generateDoubleSlitPhotons(
     float2* pos,
     curandState* states,
     int n,
     float wavelength,
     float slitDistance,
     float slitWidth,
     float screenZ,
     float xRange,
     float Imax
 ) {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (idx >= n) return;
     curandState localState = states[idx];
     float x, y, I;
     bool accepted = false;
     for (int attempts = 0; attempts < 1000 && !accepted; ++attempts) {
         x = -xRange + 2.0f * xRange * curand_uniform(&localState);
         I = doubleSlitIntensity(x, wavelength, slitDistance, slitWidth, screenZ);
         if (curand_uniform(&localState) < (I / Imax)) {
             accepted = true;
         }
     }
     y = -0.01f + 0.02f * curand_uniform(&localState);
     pos[idx] = make_float2(x, y);
     states[idx] = localState;
 }
 
 // --------------------------------------------------------------------------
 // CUDA Simulator: manages CURAND state and photon generation via CUDA kernels
 // --------------------------------------------------------------------------
 class CudaSimulator {
 public:
     explicit CudaSimulator(const SimulationParameters& p)
         : params(p), d_rngStates(nullptr), cudaVboResource(nullptr) {}
 
     ~CudaSimulator() {
         cleanup();
     }
 
     void initCUDA(GLuint vboId) {
         cudaGraphicsGLRegisterBuffer(&cudaVboResource, vboId, cudaGraphicsMapFlagsWriteDiscard);
         cudaMalloc((void**)&d_rngStates, params.numPoints * sizeof(curandState));
         int grid = (params.numPoints + params.blockSize - 1) / params.blockSize;
         setupCurandStates<<<grid, params.blockSize>>>(d_rngStates, 1234ULL, params.numPoints);
         cudaDeviceSynchronize();
     }
 
     void generatePhotons(float Imax) {
         if (!cudaVboResource) return;
         cudaGraphicsMapResources(1, &cudaVboResource, 0);
         void* dPtr = nullptr;
         size_t dSize = 0;
         cudaGraphicsResourceGetMappedPointer(&dPtr, &dSize, cudaVboResource);
         int grid = (params.numPoints + params.blockSize - 1) / params.blockSize;
         generateDoubleSlitPhotons<<<grid, params.blockSize>>>(
             reinterpret_cast<float2*>(dPtr),
             d_rngStates,
             params.numPoints,
             params.lambda,
             params.slitDist,
             params.slitWidth,
             params.screenZ,
             params.xRange,
             Imax
         );
         cudaDeviceSynchronize();
         cudaGraphicsUnmapResources(1, &cudaVboResource, 0);
     }
 
 private:
     SimulationParameters params;
     curandState* d_rngStates;
     cudaGraphicsResource* cudaVboResource;
 
     void cleanup() {
         if (d_rngStates) {
             cudaFree(d_rngStates);
             d_rngStates = nullptr;
         }
         if (cudaVboResource) {
             cudaGraphicsUnregisterResource(cudaVboResource);
             cudaVboResource = nullptr;
         }
     }
 };
 
 // --------------------------------------------------------------------------
 // OpenGL Renderer: compiles shaders, sets up FBO/quads, and draws the scene
 // --------------------------------------------------------------------------
 class DoubleSlitRenderer {
 public:
     DoubleSlitRenderer()
         : pointShaderProgram(0), quadShaderProgram(0), vbo(0),
           accumFBO(0), accumTex(0), quadVAO(0), quadVBO(0)
     {}
 
     ~DoubleSlitRenderer() {
         cleanup();
     }
 
     void initGL(int width, int height, const SimulationParameters& params) {
         if (glewInit() != GLEW_OK) {
             throw std::runtime_error("Failed to initialize GLEW.");
         }
         glViewport(0, 0, width, height);
 
         // Create VBO for photon positions
         glGenBuffers(1, &vbo);
         glBindBuffer(GL_ARRAY_BUFFER, vbo);
         size_t bufferSize = SimulationParameters::numPoints * 2 * sizeof(float);
         glBufferData(GL_ARRAY_BUFFER, bufferSize, nullptr, GL_DYNAMIC_DRAW);
         glBindBuffer(GL_ARRAY_BUFFER, 0);
 
         // Compile shaders
         pointShaderProgram = createShaderProgram(pointVertexShaderSrc, pointFragmentShaderSrc, "vertexPosition");
         quadShaderProgram  = createShaderProgram(quadVertexShaderSrc, quadFragmentShaderSrc, "pos");
 
         // Create accumulation FBO and fullscreen quad
         createAccumFBO(width, height);
         createFullScreenQuad();
     }
 
     void recreateAccumFBO(int width, int height) {
         if (accumFBO) {
             glDeleteFramebuffers(1, &accumFBO);
             accumFBO = 0;
         }
         if (accumTex) {
             glDeleteTextures(1, &accumTex);
             accumTex = 0;
         }
         createAccumFBO(width, height);
     }
 
     GLuint getVBO() const { return vbo; }
 
     void renderPointsToFBO(
         int width, int height,
         float panX, float panY, float zoom,
         float Imax, float intensityBoost,
         const SimulationParameters& params
     ) {
         glBindFramebuffer(GL_FRAMEBUFFER, accumFBO);
         glViewport(0, 0, width, height);
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
         glDrawArrays(GL_POINTS, 0, SimulationParameters::numPoints);
         glDisableVertexAttribArray(0);
         glBindBuffer(GL_ARRAY_BUFFER, 0);
 
         glUseProgram(0);
         glDisable(GL_BLEND);
         glBindFramebuffer(GL_FRAMEBUFFER, 0);
     }
 
     void renderAccumToScreen(int width, int height, float exposure) {
         glViewport(0, 0, width, height);
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
     }
 
     void clearAccumBuffer() {
         glBindFramebuffer(GL_FRAMEBUFFER, accumFBO);
         glClear(GL_COLOR_BUFFER_BIT);
         glBindFramebuffer(GL_FRAMEBUFFER, 0);
     }
 
     // Renders overlay text using stb_easy_font.
     // Uses const_cast to pass the text (since stb_easy_font_print requires a non-const char*).
     void renderTextOverlay(int windowWidth, int windowHeight, const char* text) {
         glMatrixMode(GL_PROJECTION);
         glPushMatrix();
         glLoadIdentity();
         glOrtho(0, windowWidth, windowHeight, 0, -1, 1);
 
         glMatrixMode(GL_MODELVIEW);
         glPushMatrix();
         glLoadIdentity();
 
         static char buffer[99999];
         int num_quads = stb_easy_font_print(10, 10, const_cast<char*>(text), nullptr, buffer, sizeof(buffer));
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
 
 private:
     GLuint pointShaderProgram;
     GLuint quadShaderProgram;
     GLuint vbo;
     GLuint accumFBO;
     GLuint accumTex;
     GLuint quadVAO;
     GLuint quadVBO;
 
     // Shader source strings
     static constexpr const char* pointVertexShaderSrc = R"(
         #version 120
         attribute vec2 vertexPosition;
         varying vec2 fragPos;
         void main() {
             fragPos = vertexPosition;
             gl_Position = gl_ModelViewProjectionMatrix * vec4(vertexPosition, 0.0, 1.0);
         }
     )";
 
     static constexpr const char* pointFragmentShaderSrc = R"(
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
 
     static constexpr const char* quadVertexShaderSrc = R"(
         #version 120
         attribute vec2 pos;
         varying vec2 uv;
         void main(){
             uv = (pos * 0.5) + 0.5;
             gl_Position = vec4(pos, 0.0, 1.0);
         }
     )";
 
     static constexpr const char* quadFragmentShaderSrc = R"(
         #version 120
         uniform sampler2D accumTex;
         uniform float exposure;
         varying vec2 uv;
         void main(){
             vec4 color = texture2D(accumTex, uv);
             vec3 hdrColor = vec3(1.0) - exp(-color.rgb * exposure);
             vec3 gammaCorrected = pow(hdrColor, vec3(1.0/2.2));
             gl_FragColor = vec4(gammaCorrected, 1.0);
         }
     )";
 
     static GLuint compileShader(GLenum type, const char* source) {
         GLuint shader = glCreateShader(type);
         glShaderSource(shader, 1, &source, nullptr);
         glCompileShader(shader);
         GLint compiled = 0;
         glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
         if (!compiled) {
             GLint len;
             glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
             std::string log(len, '\0');
             glGetShaderInfoLog(shader, len, &len, &log[0]);
             glDeleteShader(shader);
             throw std::runtime_error("Shader compile error: " + log);
         }
         return shader;
     }
 
     static GLuint createShaderProgram(const char* vsSrc, const char* fsSrc, const char* attrib0Name) {
         GLuint vs = compileShader(GL_VERTEX_SHADER, vsSrc);
         GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsSrc);
         GLuint prog = glCreateProgram();
         glAttachShader(prog, vs);
         glAttachShader(prog, fs);
         glBindAttribLocation(prog, 0, attrib0Name);
         glLinkProgram(prog);
         GLint linked;
         glGetProgramiv(prog, GL_LINK_STATUS, &linked);
         if (!linked) {
             GLint len;
             glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &len);
             std::string log(len, '\0');
             glGetProgramInfoLog(prog, len, &len, &log[0]);
             glDeleteProgram(prog);
             throw std::runtime_error("Shader link error: " + log);
         }
         glDeleteShader(vs);
         glDeleteShader(fs);
         return prog;
     }
 
     void createAccumFBO(int width, int height) {
         glGenFramebuffers(1, &accumFBO);
         glBindFramebuffer(GL_FRAMEBUFFER, accumFBO);
         glGenTextures(1, &accumTex);
         glBindTexture(GL_TEXTURE_2D, accumTex);
         glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
         glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
         glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
         glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, accumTex, 0);
         if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
             std::cerr << "Warning: Accumulation FBO incomplete!" << std::endl;
         }
         glClearColor(0, 0, 0, 0);
         glClear(GL_COLOR_BUFFER_BIT);
         glBindFramebuffer(GL_FRAMEBUFFER, 0);
     }
 
     void createFullScreenQuad() {
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
 
     void cleanup() {
         if (pointShaderProgram) glDeleteProgram(pointShaderProgram);
         if (quadShaderProgram)  glDeleteProgram(quadShaderProgram);
         if (vbo) glDeleteBuffers(1, &vbo);
         if (quadVBO) glDeleteBuffers(1, &quadVBO);
         if (quadVAO) glDeleteVertexArrays(1, &quadVAO);
         if (accumTex) glDeleteTextures(1, &accumTex);
         if (accumFBO) glDeleteFramebuffers(1, &accumFBO);
     }
 };
 
 // --------------------------------------------------------------------------
 // GLFW Window Wrapper: manages window creation and input events via GLFW
 // --------------------------------------------------------------------------
 class GLFWWindow {
 public:
     GLFWWindow(int w, int h, const std::string& title)
         : width(w), height(h)
     {
         if (!glfwInit()) {
             throw std::runtime_error("Failed to initialize GLFW.");
         }
         window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
         if (!window) {
             glfwTerminate();
             throw std::runtime_error("Failed to create GLFW window.");
         }
         glfwMakeContextCurrent(window);
         glfwSetWindowUserPointer(window, this);
         glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
         glfwSetScrollCallback(window, scrollCallback);
         glfwSetMouseButtonCallback(window, mouseButtonCallback);
         glfwSetCursorPosCallback(window, cursorPositionCallback);
         glfwSetKeyCallback(window, keyCallback);
     }
 
     ~GLFWWindow() {
         if (window) {
             glfwDestroyWindow(window);
             window = nullptr;
         }
         glfwTerminate();
     }
 
     void update() {
         glfwPollEvents();
         glfwSwapBuffers(window);
     }
 
     bool shouldClose() const {
         return glfwWindowShouldClose(window);
     }
 
     GLFWwindow* getGLFWHandle() const { return window; }
     int getWidth() const  { return width; }
     int getHeight() const { return height; }
 
     // Public members for user input state.
     float panX   = 0.0f;
     float panY   = 0.0f;
     float zoom   = 1.0f;
     bool clearAccumRequest = false;
     bool exitRequest       = false;
     float intensityBoost   = 1.0f;
     float exposure         = 1.0f;
 
     bool mouseDragging = false;
     double lastMouseX = 0.0;
     double lastMouseY = 0.0;
 
 private:
     GLFWwindow* window;
     int width;
     int height;
 
     static void framebufferSizeCallback(GLFWwindow* win, int w, int h) {
         auto self = static_cast<GLFWWindow*>(glfwGetWindowUserPointer(win));
         self->width = w;
         self->height = h;
         glViewport(0, 0, w, h);
     }
 
     static void scrollCallback(GLFWwindow* win, double, double yoffset) {
         auto self = static_cast<GLFWWindow*>(glfwGetWindowUserPointer(win));
         float zoomFactor = 1.1f;
         if (yoffset > 0) {
             self->zoom *= zoomFactor;
         } else if (yoffset < 0) {
             self->zoom /= zoomFactor;
         }
     }
 
     static void mouseButtonCallback(GLFWwindow* win, int button, int action, int) {
         auto self = static_cast<GLFWWindow*>(glfwGetWindowUserPointer(win));
         if (button == GLFW_MOUSE_BUTTON_LEFT) {
             if (action == GLFW_PRESS) {
                 self->mouseDragging = true;
                 glfwGetCursorPos(win, &self->lastMouseX, &self->lastMouseY);
             } else if (action == GLFW_RELEASE) {
                 self->mouseDragging = false;
             }
         }
     }
 
     static void cursorPositionCallback(GLFWwindow* win, double xpos, double ypos) {
         auto self = static_cast<GLFWWindow*>(glfwGetWindowUserPointer(win));
         if (self->mouseDragging) {
             double dx = xpos - self->lastMouseX;
             double dy = ypos - self->lastMouseY;
             self->lastMouseX = xpos;
             self->lastMouseY = ypos;
             float worldWidth  = (0.02f * 1.1f * 2.0f) / self->zoom;
             float worldHeight = (0.05f * 2.0f) / self->zoom;
             self->panX -= dx * (worldWidth / self->width);
             self->panY += dy * (worldHeight / self->height);
         }
     }
 
     static void keyCallback(GLFWwindow* win, int key, int, int action, int) {
         auto self = static_cast<GLFWWindow*>(glfwGetWindowUserPointer(win));
         if (action == GLFW_PRESS || action == GLFW_REPEAT) {
             float panSpeed = 0.0005f / self->zoom;
             switch (key) {
                 case GLFW_KEY_LEFT:  self->panX -= panSpeed; break;
                 case GLFW_KEY_RIGHT: self->panX += panSpeed; break;
                 case GLFW_KEY_UP:    self->panY += panSpeed; break;
                 case GLFW_KEY_DOWN:  self->panY -= panSpeed; break;
                 case GLFW_KEY_I:
                     self->intensityBoost += 0.01f;
                     break;
                 case GLFW_KEY_K:
                     self->intensityBoost = (self->intensityBoost > 0.02f)
                                            ? (self->intensityBoost - 0.01f)
                                            : 0.01f;
                     break;
                 case GLFW_KEY_C:
                     self->clearAccumRequest = true;
                     break;
                 case GLFW_KEY_ESCAPE:
                     self->exitRequest = true;
                     glfwSetWindowShouldClose(win, GLFW_TRUE);
                     break;
                 default:
                     break;
             }
         }
     }
 };
 
 // --------------------------------------------------------------------------
 // Application: orchestrates window, simulation, CUDA, and rendering subsystems.
 // --------------------------------------------------------------------------
 class DoubleSlitApp {
 public:
     DoubleSlitApp(int width, int height)
         : window(width, height, "Quantum Photonics Double-Slit Simulation"),
           simulation(params),
           cudaSim(params)
     {
         window.panX = 0.0f;
         window.panY = 0.0f;
         window.zoom = 1.0f;
         renderer.initGL(width, height, params);
         cudaSim.initCUDA(renderer.getVBO());
         renderer.clearAccumBuffer();
     }
 
     void run() {
         while (!window.shouldClose()) {
             window.update();
             if (window.exitRequest) break;
             if (window.clearAccumRequest) {
                 renderer.clearAccumBuffer();
                 window.clearAccumRequest = false;
             }
             int w = window.getWidth();
             int h = window.getHeight();
             renderer.recreateAccumFBO(w, h);
             if (fabs(window.panX - lastPanX) > 1e-7f || fabs(window.panY - lastPanY) > 1e-7f) {
                 renderer.clearAccumBuffer();
                 lastPanX = window.panX;
                 lastPanY = window.panY;
             }
             if (fabs(window.zoom - lastZoom) > 1e-7f) {
                 renderer.clearAccumBuffer();
                 lastZoom = window.zoom;
             }
             float Imax = simulation.computeMaxIntensity(2000);
             cudaSim.generatePhotons(Imax);
             renderer.renderPointsToFBO(w, h, window.panX, window.panY, window.zoom, Imax, window.intensityBoost, params);
             renderer.renderAccumToScreen(w, h, window.exposure);
             
             std::string overlay =
                 "Controls:\n"
                 " Pan: Arrow keys / Mouse drag\n"
                 " Zoom: Mouse wheel\n"
                 " I/K: Intensity boost +/-\n"
                 " C: Clear accumulation\n"
                 " ESC: Quit\n"
                 "\n"
                 "Parameters:\n"
                 " Zoom: " + std::to_string(window.zoom) +
                 "  IntBoost: " + std::to_string(window.intensityBoost) +
                 "  Exposure: " + std::to_string(window.exposure) + "\n" +
                 " lambda=" + std::to_string(params.lambda) + " m, " +
                 " d=" + std::to_string(params.slitDist) + " m, " +
                 " width=" + std::to_string(params.slitWidth) + " m\n" +
                 " panX=" + std::to_string(window.panX) +
                 " panY=" + std::to_string(window.panY) + "\n";
             renderer.renderTextOverlay(w, h, overlay.c_str());
         }
         cudaDeviceSynchronize();
     }
 
 private:
     GLFWWindow window;
     SimulationParameters params;
     DoubleSlitSimulation simulation;
     CudaSimulator cudaSim;
     DoubleSlitRenderer renderer;
     float lastPanX = 0.0f;
     float lastPanY = 0.0f;
     float lastZoom = 1.0f;
 };
 
 // --------------------------------------------------------------------------
 // Main entry point
 // --------------------------------------------------------------------------
 int main() {
     try {
         DoubleSlitApp app(800, 600);
         app.run();
     } catch (const std::exception& ex) {
         std::cerr << "Fatal error: " << ex.what() << std::endl;
         return EXIT_FAILURE;
     }
     return EXIT_SUCCESS;
 }
 