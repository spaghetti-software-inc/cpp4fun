#include <iostream>
#include <cmath>
#include <vector>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>

// GLM for matrix math
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// -----------------------------------------------------------------------------
// Shader sources
// -----------------------------------------------------------------------------
static const char* vsSource = R"(
#version 330 core
layout (location = 0) in vec4 inPosRad; // (x, y, z, radiusForColor)

uniform mat4 u_mvp;

out vec4 vColor;

void main()
{
    // Each star's position
    gl_Position = u_mvp * vec4(inPosRad.xyz, 1.0);
    gl_PointSize = 2.0;

    // Color by radius (in [0,1] or so). We'll just clamp it for safety.
    float r = clamp(inPosRad.w, 0.0, 1.0);

    // Simple gradient: from white (center) to bluish (outer edge)
    vColor = vec4(1.0 - r, 1.0 - r, 1.0, 1.0); // center=white, outer=blueish
}
)";

static const char* fsSource = R"(
#version 330 core
in vec4 vColor;
out vec4 FragColor;
void main()
{
    FragColor = vColor;
}
)";

// -----------------------------------------------------------------------------
// Globals for camera control
// -----------------------------------------------------------------------------
static int   g_width = 800, g_height = 600;
static float g_yaw = 0.f, g_pitch = 0.f, g_dist = 3.f;
static bool  g_lmb = false, g_rmb = false;
static double g_lastX=0.f, g_lastY=0.f;

// -----------------------------------------------------------------------------
void framebufferCB(GLFWwindow* window, int width, int height)
{
    g_width  = width;
    g_height = height;
    glViewport(0, 0, width, height);
}

void mouseButtonCB(GLFWwindow* w, int button, int action, int mods)
{
    if(button == GLFW_MOUSE_BUTTON_LEFT) {
        g_lmb = (action == GLFW_PRESS);
        glfwGetCursorPos(w, &g_lastX, &g_lastY);
    }
    else if(button == GLFW_MOUSE_BUTTON_RIGHT) {
        g_rmb = (action == GLFW_PRESS);
        glfwGetCursorPos(w, &g_lastX, &g_lastY);
    }
}

void cursorPosCB(GLFWwindow* w, double x, double y)
{
    float dx = float(x - g_lastX);
    float dy = float(y - g_lastY);
    g_lastX  = x;
    g_lastY  = y;

    if(g_lmb) {
        g_yaw   += dx * 0.3f;
        g_pitch += dy * 0.3f;
    }
    if(g_rmb) {
        g_dist += dy * 0.01f;
        if(g_dist < 0.1f) g_dist = 0.1f;
    }
}

void scrollCB(GLFWwindow* w, double xoff, double yoff)
{
    g_dist -= float(yoff)*0.2f;
    if(g_dist < 0.1f) g_dist = 0.1f;
}

// -----------------------------------------------------------------------------
// Simple shader utility
// -----------------------------------------------------------------------------
static GLuint compileShader(GLenum type, const char* src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint success;
    glGetShaderiv(s, GL_COMPILE_STATUS, &success);
    if(!success) {
        char log[512];
        glGetShaderInfoLog(s, 512, nullptr, log);
        std::cerr << "Shader compile error:\n" << log << std::endl;
    }
    return s;
}

static GLuint createShaderProgram(const char* vtx, const char* frg)
{
    GLuint vs = compileShader(GL_VERTEX_SHADER,   vtx);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, frg);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);

    GLint success;
    glGetProgramiv(prog, GL_LINK_STATUS, &success);
    if(!success) {
        char log[512];
        glGetProgramInfoLog(prog, 512, nullptr, log);
        std::cerr << "Program link error:\n" << log << std::endl;
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

// -----------------------------------------------------------------------------
// CUDA kernel data layout
//
// Each star is stored in an interleaved manner of 8 floats:
//   starData[8*i + 0] = posX
//   starData[8*i + 1] = posY
//   starData[8*i + 2] = posZ
//   starData[8*i + 3] = radiusForColor
//   starData[8*i + 4] = orbitRadius
//   starData[8*i + 5] = angle
//   starData[8*i + 6] = zOffset
//   starData[8*i + 7] = angularVelocity
//
// The vertex shader only reads the first 4 floats (posX, posY, posZ, radiusForColor).
// We update these first 4 floats each frame based on the orbit parameters (4..7).
// -----------------------------------------------------------------------------

__global__
void initGalaxy(float* starData, int numStars, unsigned int seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numStars) return;

    // Use a simple hash-based PRNG or a built-in CURAND. For demonstration,
    // let's do a tiny LCG or manipulations of idx + seed.

    // For more randomness, you could use curand in a separate step, but let's do a quick approach here:
    unsigned int h = idx ^ seed; 
    h ^= (h << 13); h ^= (h >> 17); h ^= (h << 5);
    float r1 = (h & 0xFFFF) / 65536.f;  // [0,1)
    float r2 = ((h >> 16) & 0xFFFF) / 65536.f;

    // Random radius in [0..1]
    float orbitRadius = powf(r1, 0.5f); // denser near center
    // random angle in [0..2pi]
    float angle = r2 * 6.2831853f;

    // small random z offset for a bit of thickness
    float zOffset = 0.02f * (r1 - 0.5f);

    // simplistic velocity: ~1 / sqrt(r + someEpsilon)
    float eps = 0.05f;
    float angularVelocity = 0.4f / sqrtf(orbitRadius + eps);

    // Store orbit parameters
    int base = 8*idx;
    starData[base + 4] = orbitRadius;
    starData[base + 5] = angle;
    starData[base + 6] = zOffset;
    starData[base + 7] = angularVelocity;

    // Compute initial position
    float x = orbitRadius * cosf(angle);
    float y = orbitRadius * sinf(angle);
    float z = zOffset;

    starData[base + 0] = x;
    starData[base + 1] = y;
    starData[base + 2] = z;
    // We'll use radiusForColor in the w-component => [0..1], just store orbitRadius
    starData[base + 3] = orbitRadius;
}

__global__
void updateGalaxy(float* starData, int numStars, float dt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numStars) return;

    int base = 8*idx;

    float orbitRadius    = starData[base + 4];
    float angle          = starData[base + 5];
    float zOffset        = starData[base + 6];
    float angularVel     = starData[base + 7];

    // Update angle
    angle += angularVel * dt;

    // Write it back
    starData[base + 5] = angle;

    // Compute new position
    float x = orbitRadius * cosf(angle);
    float y = orbitRadius * sinf(angle);
    float z = zOffset;

    starData[base + 0] = x;
    starData[base + 1] = y;
    starData[base + 2] = z;

    // The w-component is used for color interpolation => store orbitRadius there
    starData[base + 3] = orbitRadius;
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main()
{
    // 1) Initialize GLFW
    if(!glfwInit()) {
        std::cerr << "GLFW init failed\n";
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(g_width, g_height, "Galaxy Simulation (CUDA + OpenGL)", nullptr, nullptr);
    if(!window) {
        std::cerr << "Window creation failed\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    glfwSetFramebufferSizeCallback(window, framebufferCB);
    glfwSetMouseButtonCallback(window, mouseButtonCB);
    glfwSetCursorPosCallback(window, cursorPosCB);
    glfwSetScrollCallback(window, scrollCB);

    // 2) Initialize GLEW
    if( glewInit() != GLEW_OK ) {
        std::cerr << "GLEW init failed\n";
        return -1;
    }

    // 3) Create VBO + VAO
    int numStars = 50000;
    size_t floatsPerStar = 8; // see layout above
    size_t totalFloats = floatsPerStar * numStars;
    size_t bufferSize = totalFloats * sizeof(float);

    GLuint vbo;
    glGenBuffers(1, &vbo);

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, bufferSize, nullptr, GL_DYNAMIC_DRAW);

    // 4) Register VBO with CUDA
    cudaGraphicsResource* cudaRes;
    cudaGraphicsGLRegisterBuffer(&cudaRes, vbo, cudaGraphicsMapFlagsWriteDiscard);

    // 5) Initialize galaxy data (CUDA kernel)
    {
        // Map buffer
        cudaGraphicsMapResources(1, &cudaRes, 0);
        float* dPtr = nullptr;
        size_t dSize = 0;
        cudaGraphicsResourceGetMappedPointer((void**)&dPtr, &dSize, cudaRes);

        // Launch init kernel
        int blockSize = 256;
        int gridSize  = (numStars + blockSize - 1) / blockSize;
        initGalaxy<<<gridSize, blockSize>>>(dPtr, numStars, 12345u);
        cudaDeviceSynchronize();

        // Unmap
        cudaGraphicsUnmapResources(1, &cudaRes, 0);
    }

    // 6) Set up the vertex attribute pointers
    //    We only feed the first 4 floats (posX,posY,posZ,radiusForColor) to the vertex shader.
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, floatsPerStar*sizeof(float), (void*)(0));

    glBindVertexArray(0);

    // 7) Create shaders
    GLuint prog = createShaderProgram(vsSource, fsSource);
    glUseProgram(prog);
    GLint mvpLoc = glGetUniformLocation(prog, "u_mvp");

    // Some OpenGL states
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_DEPTH_TEST);

    // We'll track time to do a time-step for the simulation
    float lastTime = (float)glfwGetTime();

    // Main Loop
    while(!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        float currentTime = (float)glfwGetTime();
        float dt = currentTime - lastTime;
        lastTime = currentTime;
        if(dt > 0.1f) dt = 0.1f; // clamp for safety in case of large frame time

        // 1) Update GPU simulation
        {
            cudaGraphicsMapResources(1, &cudaRes, 0);
            float* dPtr = nullptr;
            size_t dSize = 0;
            cudaGraphicsResourceGetMappedPointer((void**)&dPtr, &dSize, cudaRes);

            int blockSize = 256;
            int gridSize  = (numStars + blockSize - 1) / blockSize;
            updateGalaxy<<<gridSize, blockSize>>>(dPtr, numStars, dt);
            cudaDeviceSynchronize();

            cudaGraphicsUnmapResources(1, &cudaRes, 0);
        }

        // 2) Render
        glClearColor(0.02f, 0.02f, 0.06f, 1.f); // a dark background
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Build MVP (camera)
        float aspect = float(g_width)/float(g_height);
        glm::mat4 proj  = glm::perspective(glm::radians(45.f), aspect, 0.1f, 100.f);
        glm::mat4 view  = glm::translate(glm::mat4(1.f), glm::vec3(0, 0, -g_dist));
        view            = glm::rotate(view, glm::radians(g_pitch), glm::vec3(1, 0, 0));
        view            = glm::rotate(view, glm::radians(g_yaw),   glm::vec3(0, 1, 0));
        glm::mat4 model = glm::mat4(1.f);
        glm::mat4 mvp   = proj * view * model;

        glUseProgram(prog);
        glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, glm::value_ptr(mvp));

        glBindVertexArray(vao);
        glDrawArrays(GL_POINTS, 0, numStars);
        glBindVertexArray(0);

        glfwSwapBuffers(window);
    }

    // Cleanup
    cudaGraphicsUnregisterResource(cudaRes);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(prog);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
