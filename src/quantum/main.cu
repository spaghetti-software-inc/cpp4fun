/******************************************************************************
 *  DoubleSlit_ResetOnPanOrZoom.cu
 *
 *  Demonstrates a double-slit simulation with:
 *    - Time accumulation in a float texture,
 *    - Dynamic Imax computation each frame,
 *    - Resets accumulation when user pans (mouse drag or arrow keys) or zooms.
 *
 *  Controls:
 *    - Mouse drag or Arrow keys: Pan (resets accumulation on change)
 *    - Mouse wheel: Zoom in/out (resets accumulation)
 *    - I/K: Increase/Decrease intensity boost
 *    - C: Clear accumulation manually
 *    - ESC: Exit
 ******************************************************************************/

 #include <iostream>
 #include <cmath>
 #include <cstdio>
 #include <GL/glew.h>
 #include <GLFW/glfw3.h>
 
 #include <cuda_runtime.h>
 #include <curand_kernel.h>
 #include <cuda_gl_interop.h>
 
 // For text rendering
 #define STB_EASY_FONT_IMPLEMENTATION
 #include "stb_easy_font.h"
 
 // ----------------------------------------------------------
 // Simulation / Physical Parameters
 // ----------------------------------------------------------
 static const float DEFAULT_LAMBDA    = 0.5e-6f; // 500 nm
 static const float DEFAULT_SLIT_DIST = 1.0e-3f; // 1 mm
 static const float DEFAULT_SLIT_WIDTH= 0.2e-3f; // 0.2 mm
 static const float DEFAULT_SCREEN_Z  = 1.0f;    // 1 m
 static const float XRANGE           = 0.02f;    // +/- 2 cm
 
 // Number of photons per frame
 static const size_t NUM_POINTS = 100000;
 static const int    BLOCK_SIZE = 256;
 
 // ----------------------------------------------------------
 // Globals for camera/pan/zoom, intensity, window size
 // ----------------------------------------------------------
 float panX = 0.0f;
 float panY = 0.0f;
 float zoom = 1.0f;
 float intensityBoost = 1.0f;
 
 int windowWidth  = 800;
 int windowHeight = 600;
 
 bool   mouseDragging = false;
 double lastMouseX    = 0.0;
 double lastMouseY    = 0.0;
 
 // Track last pan & zoom so we can detect changes
 float lastPanX = 0.0f;
 float lastPanY = 0.0f;
 float lastZoom = 1.0f;
 
 // Slit parameters (could be made dynamic too)
 float LAMBDA    = DEFAULT_LAMBDA;
 float SLIT_DIST = DEFAULT_SLIT_DIST;
 float SLIT_WIDTH= DEFAULT_SLIT_WIDTH;
 float SCREEN_Z  = DEFAULT_SCREEN_Z;
 
 // ----------------------------------------------------------
 // Device-side double-slit intensity
 // ----------------------------------------------------------
 __device__ __inline__
 float sinc2f(float x)
 {
     if (fabsf(x) < 1.0e-7f) return 1.0f;
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
 // CPU version for scanning to find max intensity
 // ----------------------------------------------------------
 float doubleSlitIntensityCPU(float x, float wavelength, float d, float a, float z)
 {
     float alpha = (float)M_PI * d * x / (wavelength * z);
     float beta  = (float)M_PI * a * x / (wavelength * z);
 
     float c   = cosf(alpha);
     float val = c*c;
 
     // sinc^2:
     float denom = (fabsf(beta) < 1e-7f) ? 1.0f : beta;
     float s     = sinf(denom)/denom;
     val *= (s*s);
 
     return val;
 }
 
 float computeMaxIntensity(int sampleCount)
 {
     float maxI = 0.0f;
     for (int i = 0; i < sampleCount; ++i) {
         float frac = (float)i / (float)(sampleCount - 1);
         float x    = -XRANGE + 2.0f * XRANGE * frac;
         float I    = doubleSlitIntensityCPU(x, LAMBDA, SLIT_DIST, SLIT_WIDTH, SCREEN_Z);
         if (I > maxI) maxI = I;
     }
     return maxI;
 }
 
 // ----------------------------------------------------------
 // CURAND setup + Photon generation
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
     float2*      pos,
     curandState* states,
     int          n,
     float        wavelength,
     float        slitDistance,
     float        slitWidth,
     float        screenZ,
     float        xRange,
     float        Imax
 )
 {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (idx >= n) return;
 
     curandState localState = states[idx];
     float x, y, I;
     bool accepted = false;
     for (int attempts = 0; attempts < 1000 && !accepted; attempts++) {
         x = -xRange + 2.0f*xRange*curand_uniform(&localState);
         I = doubleSlitIntensity(x, wavelength, slitDistance, slitWidth, screenZ);
         float testVal = curand_uniform(&localState);
         if (testVal < (I / Imax)) {
             accepted = true;
         }
     }
     // Slight vertical spread
     y = -0.01f + 0.02f * curand_uniform(&localState);
     pos[idx] = make_float2(x, y);
 
     states[idx] = localState;
 }
 
 // ----------------------------------------------------------
 // Shaders
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
     float beta  = 3.14159265359 * slitWidth    * x / (wavelength * screenZ);
     return cos(alpha)*cos(alpha) * sinc2(beta);
 }
 
 // Simple approximate mapping from wavelength to RGB
 vec3 wavelengthToRGB(float lambda) {
     float nm = lambda * 1e9; // convert to nm
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
         B = -(nm - 510.0)/(510.0 - 490.0);
     } else if(nm >= 510.0 && nm < 580.0) {
         R = (nm - 510.0)/(580.0 - 510.0);
         G = 1.0;
         B = 0.0;
     } else if(nm >= 580.0 && nm < 645.0) {
         R = 1.0;
         G = -(nm - 645.0)/(645.0 - 580.0);
         B = 0.0;
     } else if(nm >= 645.0 && nm <= 780.0) {
         R = 1.0;
         G = 0.0;
         B = 0.0;
     } else {
         R = 0.0; G = 0.0; B = 0.0;
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
 
     return vec3(R, G, B)*factor;
 }
 
 void main() {
     float localI = doubleSlitIntensity(fragPos.x) / Imax;
     float val = clamp(localI * intensityBoost, 0.0, 1.0);
     vec3 baseColor = wavelengthToRGB(wavelength);
     gl_FragColor = vec4(baseColor * val, 1.0);
 }
 )";
 
 // Fullscreen quad to display accumTex
 const char* quadVertexShaderSource = R"(
 #version 120
 attribute vec2 pos;
 varying vec2 uv;
 void main() {
     uv = (pos*0.5) + 0.5;
     gl_Position = vec4(pos, 0.0, 1.0);
 }
 )";
 
 const char* quadFragmentShaderSource = R"(
 #version 120
 uniform sampler2D accumTex;
 varying vec2 uv;
 void main(){
     vec4 color = texture2D(accumTex, uv);
     gl_FragColor = color;
 }
 )";
 
 // ----------------------------------------------------------
 // Shader helpers
 // ----------------------------------------------------------
 GLuint compileShader(GLenum type, const char* source)
 {
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
         std::cerr << "Shader compile error:\n" << log << std::endl;
         delete[] log;
         glDeleteShader(shader);
         return 0;
     }
     return shader;
 }
 
 GLuint createShaderProgram(const char* vsSource, const char* fsSource, const char* attrib0Name)
 {
     GLuint vs = compileShader(GL_VERTEX_SHADER, vsSource);
     GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsSource);
     if (!vs || !fs) return 0;
 
     GLuint program = glCreateProgram();
     glAttachShader(program, vs);
     glAttachShader(program, fs);
     glBindAttribLocation(program, 0, attrib0Name);
     glLinkProgram(program);
 
     GLint linked;
     glGetProgramiv(program, GL_LINK_STATUS, &linked);
     if (!linked) {
         GLint len;
         glGetProgramiv(program, GL_INFO_LOG_LENGTH, &len);
         char* log = new char[len];
         glGetProgramInfoLog(program, len, &len, log);
         std::cerr << "Shader link error:\n" << log << std::endl;
         delete[] log;
         glDeleteProgram(program);
         return 0;
     }
     glDeleteShader(vs);
     glDeleteShader(fs);
     return program;
 }
 
 // ----------------------------------------------------------
 // GLFW callbacks
 // ----------------------------------------------------------
 void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
 {
     float zoomFactor = 1.1f;
     if (yoffset > 0) zoom *= zoomFactor;
     else if (yoffset < 0) zoom /= zoomFactor;
 }
 
 void framebuffer_size_callback(GLFWwindow* window, int width, int height)
 {
     windowWidth = width;
     windowHeight = height;
     glViewport(0, 0, width, height);
 }
 
 void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
 {
     if (button == GLFW_MOUSE_BUTTON_LEFT) {
         if (action == GLFW_PRESS) {
             mouseDragging = true;
             glfwGetCursorPos(window, &lastMouseX, &lastMouseY);
         }
         else if (action == GLFW_RELEASE) {
             mouseDragging = false;
         }
     }
 }
 
 void cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
 {
     if (mouseDragging) {
         double dx = xpos - lastMouseX;
         double dy = ypos - lastMouseY;
         lastMouseX = xpos;
         lastMouseY = ypos;
         float worldWidth  = (XRANGE*1.1f*2.0f)/zoom;
         float worldHeight = (0.05f*2.0f)/zoom;
         panX -= dx*(worldWidth / windowWidth);
         panY += dy*(worldHeight / windowHeight);
     }
 }
 
 // ----------------------------------------------------------
 // Text Overlay
 // ----------------------------------------------------------
 void renderTextOverlay()
 {
     glMatrixMode(GL_PROJECTION);
     glPushMatrix();
     glLoadIdentity();
     glOrtho(0, windowWidth, windowHeight, 0, -1, 1);
 
     glMatrixMode(GL_MODELVIEW);
     glPushMatrix();
     glLoadIdentity();
 
     char info[512];
     sprintf(info,
             "Controls:\n"
             " Pan: Arrow keys / Mouse drag => resets accumulation\n"
             " Zoom: Mouse wheel => resets accumulation\n"
             " I/K = intensity boost +/-\n"
             " C = clear accumulation\n"
             " ESC = Quit\n"
             "Params:\n"
             " Zoom: %.2f  IntBoost: %.2f\n"
             " lambda=%.3g m, dist=%.3g m, width=%.3g m, screenZ=%.3g m\n"
             " panX=%.4f, panY=%.4f",
             zoom, intensityBoost, LAMBDA, SLIT_DIST, SLIT_WIDTH, SCREEN_Z, panX, panY);
 
     char buffer[99999];
     int num_quads = stb_easy_font_print(10, 10, info, NULL, buffer, sizeof(buffer));
 
     glColor3f(1,1,1);
     glEnableClientState(GL_VERTEX_ARRAY);
     glVertexPointer(2, GL_FLOAT, 16, buffer);
     glDrawArrays(GL_QUADS, 0, num_quads*4);
     glDisableClientState(GL_VERTEX_ARRAY);
 
     glPopMatrix();
     glMatrixMode(GL_PROJECTION);
     glPopMatrix();
     glMatrixMode(GL_MODELVIEW);
 }
 
 // ----------------------------------------------------------
 // Main
 // ----------------------------------------------------------
 int main()
 {
     // Init GLFW
     if (!glfwInit()) {
         std::cerr << "Failed to init GLFW\n";
         return -1;
     }
 
     GLFWwindow* window = glfwCreateWindow(800, 600, "Double-Slit (Reset on Pan/Zoom)", nullptr, nullptr);
     if (!window) {
         std::cerr << "Failed to create window\n";
         glfwTerminate();
         return -1;
     }
     glfwMakeContextCurrent(window);
 
     // Set callbacks
     glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
     glfwSetScrollCallback(window, scroll_callback);
     glfwSetMouseButtonCallback(window, mouse_button_callback);
     glfwSetCursorPosCallback(window, cursor_position_callback);
 
     // GLEW
     if (glewInit() != GLEW_OK) {
         std::cerr << "Failed to init GLEW\n";
         glfwDestroyWindow(window);
         glfwTerminate();
         return -1;
     }
 
     // Query viewport
     glfwGetFramebufferSize(window, &windowWidth, &windowHeight);
     glViewport(0, 0, windowWidth, windowHeight);
 
     // --------------------------------------------------
     // Create VBO + register with CUDA
     // --------------------------------------------------
     GLuint vbo;
     glGenBuffers(1, &vbo);
     glBindBuffer(GL_ARRAY_BUFFER, vbo);
     size_t bufferSize = NUM_POINTS*2*sizeof(float);
     glBufferData(GL_ARRAY_BUFFER, bufferSize, nullptr, GL_DYNAMIC_DRAW);
     glBindBuffer(GL_ARRAY_BUFFER, 0);
 
     cudaGraphicsResource* cudaVboResource;
     cudaGraphicsGLRegisterBuffer(&cudaVboResource, vbo, cudaGraphicsMapFlagsWriteDiscard);
 
     curandState* d_rngStates = nullptr;
     cudaMalloc((void**)&d_rngStates, NUM_POINTS*sizeof(curandState));
 
     int grid = (NUM_POINTS + BLOCK_SIZE - 1) / BLOCK_SIZE;
     setupCurandStates<<<grid, BLOCK_SIZE>>>(d_rngStates, 1234ULL, NUM_POINTS);
     cudaDeviceSynchronize();
 
     // --------------------------------------------------
     // Create Shaders
     // --------------------------------------------------
     GLuint pointShaderProgram = createShaderProgram(vertexShaderSource, fragmentShaderSource, "vertexPosition");
     GLuint quadShaderProgram  = createShaderProgram(quadVertexShaderSource, quadFragmentShaderSource, "pos");
     if (!pointShaderProgram || !quadShaderProgram) {
         std::cerr << "Shader program creation failed\n";
         return -1;
     }
 
     // --------------------------------------------------
     // Create accumulation FBO/Texture
     // --------------------------------------------------
     GLuint accumFBO;
     glGenFramebuffers(1, &accumFBO);
     glBindFramebuffer(GL_FRAMEBUFFER, accumFBO);
 
     GLuint accumTex;
     glGenTextures(1, &accumTex);
     glBindTexture(GL_TEXTURE_2D, accumTex);
     glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, windowWidth, windowHeight, 0,
                  GL_RGBA, GL_FLOAT, nullptr);
     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
 
     glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, accumTex, 0);
 
     if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
         std::cerr << "accumFBO incomplete!\n";
 
     glClearColor(0,0,0,0);
     glClear(GL_COLOR_BUFFER_BIT);
     glBindFramebuffer(GL_FRAMEBUFFER, 0);
 
     // --------------------------------------------------
     // Fullscreen quad to display accumTex
     // --------------------------------------------------
     GLuint quadVAO, quadVBO;
     glGenVertexArrays(1, &quadVAO);
     glBindVertexArray(quadVAO);
 
     static const GLfloat fsQuadVerts[] = {
         -1.f, -1.f,
          1.f, -1.f,
         -1.f,  1.f,
          1.f,  1.f
     };
     glGenBuffers(1, &quadVBO);
     glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
     glBufferData(GL_ARRAY_BUFFER, sizeof(fsQuadVerts), fsQuadVerts, GL_STATIC_DRAW);
 
     glEnableVertexAttribArray(0);
     glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
 
     glBindVertexArray(0);
 
     // Initialize trackers
     lastPanX = panX;
     lastPanY = panY;
     lastZoom = zoom;
 
     float Imax = 1.0f;
 
     // Main loop
     while (!glfwWindowShouldClose(window))
     {
         glfwPollEvents();
 
         // Arrow keys => Pan
         float panSpeed = 0.0005f / zoom;
         if (glfwGetKey(window, GLFW_KEY_LEFT)  == GLFW_PRESS) panX -= panSpeed;
         if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) panX += panSpeed;
         if (glfwGetKey(window, GLFW_KEY_UP)    == GLFW_PRESS) panY += panSpeed;
         if (glfwGetKey(window, GLFW_KEY_DOWN)  == GLFW_PRESS) panY -= panSpeed;
 
         // I/K => adjust intensityBoost
         if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS)
             intensityBoost += 0.01f;
         if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS)
             intensityBoost = (intensityBoost > 0.02f) ? (intensityBoost - 0.01f) : 0.01f;
 
         // C => clear accumulation
         if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS) {
             glBindFramebuffer(GL_FRAMEBUFFER, accumFBO);
             glClear(GL_COLOR_BUFFER_BIT);
             glBindFramebuffer(GL_FRAMEBUFFER, 0);
         }
 
         // -------------------------------
         // Reset on pan changes
         // -------------------------------
         float panDeltaX = fabsf(panX - lastPanX);
         float panDeltaY = fabsf(panY - lastPanY);
         if (panDeltaX > 1e-7f || panDeltaY > 1e-7f) {
             glBindFramebuffer(GL_FRAMEBUFFER, accumFBO);
             glClear(GL_COLOR_BUFFER_BIT);
             glBindFramebuffer(GL_FRAMEBUFFER, 0);
 
             // Update trackers
             lastPanX = panX;
             lastPanY = panY;
         }
 
         // -------------------------------
         // Reset on zoom changes
         // -------------------------------
         if (fabsf(zoom - lastZoom) > 1e-7f) {
             glBindFramebuffer(GL_FRAMEBUFFER, accumFBO);
             glClear(GL_COLOR_BUFFER_BIT);
             glBindFramebuffer(GL_FRAMEBUFFER, 0);
 
             lastZoom = zoom;
         }
 
         // 1) Compute dynamic Imax
         Imax = computeMaxIntensity(2000);
 
         // 2) Generate new photons
         cudaGraphicsMapResources(1, &cudaVboResource, 0);
         void* dPtr = nullptr;
         size_t dSize = 0;
         cudaGraphicsResourceGetMappedPointer(&dPtr, &dSize, cudaVboResource);
 
         generateDoubleSlitPhotons<<<grid, BLOCK_SIZE>>>(
             (float2*)dPtr, d_rngStates, NUM_POINTS,
             LAMBDA, SLIT_DIST, SLIT_WIDTH, SCREEN_Z, XRANGE,
             Imax
         );
         cudaDeviceSynchronize();
 
         cudaGraphicsUnmapResources(1, &cudaVboResource, 0);
 
         // PASS 1: render photons => accumFBO
         glBindFramebuffer(GL_FRAMEBUFFER, accumFBO);
         glViewport(0, 0, windowWidth, windowHeight);
 
         glEnable(GL_BLEND);
         glBlendFunc(GL_ONE, GL_ONE);
 
         // Orthographic setup
         glMatrixMode(GL_PROJECTION);
         glLoadIdentity();
         float left   = panX - (XRANGE*1.1f)/zoom;
         float right  = panX + (XRANGE*1.1f)/zoom;
         float bottom = panY - (0.05f)/zoom;
         float top    = panY + (0.05f)/zoom;
         glOrtho(left, right, bottom, top, -1.0, 1.0);
 
         glMatrixMode(GL_MODELVIEW);
         glLoadIdentity();
 
         glUseProgram(pointShaderProgram);
         glUniform1f(glGetUniformLocation(pointShaderProgram, "wavelength"),    LAMBDA);
         glUniform1f(glGetUniformLocation(pointShaderProgram, "slitDistance"), SLIT_DIST);
         glUniform1f(glGetUniformLocation(pointShaderProgram, "slitWidth"),    SLIT_WIDTH);
         glUniform1f(glGetUniformLocation(pointShaderProgram, "screenZ"),      SCREEN_Z);
         glUniform1f(glGetUniformLocation(pointShaderProgram, "Imax"),         Imax);
         glUniform1f(glGetUniformLocation(pointShaderProgram, "intensityBoost"), intensityBoost);
 
         glBindBuffer(GL_ARRAY_BUFFER, vbo);
         glEnableVertexAttribArray(0);
         glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
 
         glDrawArrays(GL_POINTS, 0, NUM_POINTS);
 
         glDisableVertexAttribArray(0);
         glBindBuffer(GL_ARRAY_BUFFER, 0);
         glUseProgram(0);
 
         glDisable(GL_BLEND);
         glBindFramebuffer(GL_FRAMEBUFFER, 0);
 
         // PASS 2: draw accumTex => screen
         glViewport(0, 0, windowWidth, windowHeight);
         glClear(GL_COLOR_BUFFER_BIT);
 
         glUseProgram(quadShaderProgram);
         glActiveTexture(GL_TEXTURE0);
         glBindTexture(GL_TEXTURE_2D, accumTex);
         GLint loc = glGetUniformLocation(quadShaderProgram, "accumTex");
         glUniform1i(loc, 0);
 
         glBindVertexArray(quadVAO);
         glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
         glBindVertexArray(0);
 
         glUseProgram(0);
 
         // Overlay text
         renderTextOverlay();
 
         glfwSwapBuffers(window);
 
         // ESC => exit
         if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
             glfwSetWindowShouldClose(window, 1);
     }
 
     // Cleanup
     cudaFree(d_rngStates);
     cudaGraphicsUnregisterResource(cudaVboResource);
     glDeleteBuffers(1, &vbo);
     glDeleteProgram(pointShaderProgram);
     glDeleteProgram(quadShaderProgram);
     glDeleteTextures(1, &accumTex);
     glDeleteFramebuffers(1, &accumFBO);
     glDeleteVertexArrays(1, &quadVAO);
     glDeleteBuffers(1, &quadVBO);
 
     glfwDestroyWindow(window);
     glfwTerminate();
     return 0;
 }
 