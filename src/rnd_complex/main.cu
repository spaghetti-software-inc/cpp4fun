#include <iostream>
#include <cmath>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <curand_kernel.h>

// Optionally for matrix math
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// -----------------------------------------------------------------------------
// Camera / Global
// -----------------------------------------------------------------------------
static int   g_width  = 800;
static int   g_height = 600;

// Pan/zoom in the complex plane
static float g_centerX = 0.f, g_centerY = 0.f;
static float g_scale   = 1.f;

static bool  g_lmb=false, g_rmb=false;
static double g_lastX=0., g_lastY=0.;

// # of random points
static int   g_numPoints = 50000;

// -----------------------------------------------------------------------------
// Vertex + Fragment Shaders
//   We'll pass (x,y) from the VBO, and compute color in the vertex shader
//   by converting (x,y) -> polar (r, theta) => hue = theta, brightness = f(r).
// -----------------------------------------------------------------------------
static const char* vsSource = R"(
#version 330 core
layout(location = 0) in vec2 inPos;

uniform mat4 u_mvp; // for 2D or 3D transform

out vec4 vColor;

// Simple HSV->RGB
vec3 hsv2rgb(float h, float s, float v) {
    h = fract(h);
    float r, g, b;
    float hd = h*6.0;
    int i = int(hd);
    float f = hd - float(i);
    float p = v*(1.0 - s);
    float q = v*(1.0 - f*s);
    float t = v*(1.0 - (1.0 - f)*s);
    if(i == 0) { r=v; g=t; b=p; }
    else if(i==1){ r=q; g=v; b=p; }
    else if(i==2){ r=p; g=v; b=t; }
    else if(i==3){ r=p; g=q; b=v; }
    else if(i==4){ r=t; g=p; b=v; }
    else { r=v; g=p; b=q; }
    return vec3(r,g,b);
}

void main()
{
    // Transform position
    gl_PointSize = 3.0;
    gl_Position = u_mvp * vec4(inPos, 0.0, 1.0);

    float x = inPos.x;
    float y = inPos.y;
    float r = sqrt(x*x + y*y);
    float theta = atan(y,x); // [-pi, pi]

    // Map angle to hue [0..1]
    float hue = (theta + 3.1415926535) / (2.0*3.1415926535);
    float saturation = 1.0;
    // Example brightness: logistic function so we see detail near 0
    float brightness = 1.0 - (1.0 / (1.0 + r*0.5));

    // Convert to RGB
    vec3 rgb = hsv2rgb(hue, saturation, brightness);
    vColor = vec4(rgb, 1.0);
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
// Shader utility
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
// CUDA kernel to generate random points (x,y) in the plane
// Here we use a normal distribution ~ N(0,1), but you can adapt as needed
// -----------------------------------------------------------------------------
__global__ void setupCurandStates(curandState *states, unsigned int seed)
{
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    // Each thread gets a state
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void randomPointsKernel(curandState *states, float2* outPoints, int n)
{
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if(idx >= n) return;

    curandState &st = states[idx];

    // Generate real, imag from e.g. normal distribution
    float x = curand_normal(&st); // N(0,1)
    float y = curand_normal(&st); // N(0,1)

    // Optionally scale or shift
    // x *= 0.5f;  y *= 0.5f;

    outPoints[idx] = make_float2(x, y);
}

// -----------------------------------------------------------------------------
// Global for CUDA-OpenGL resources
// -----------------------------------------------------------------------------
static GLuint              g_vbo = 0;
static cudaGraphicsResource* g_cudaRes = nullptr;
static curandState*        g_curandStates = nullptr;

// -----------------------------------------------------------------------------
// Camera / Mouse Handling
// -----------------------------------------------------------------------------
void framebufferCB(GLFWwindow* win, int width, int height)
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
    } else if(button == GLFW_MOUSE_BUTTON_RIGHT) {
        g_rmb = (action == GLFW_PRESS);
        glfwGetCursorPos(w, &g_lastX, &g_lastY);
    }
}

void cursorPosCB(GLFWwindow* w, double x, double y)
{
    double dx = x - g_lastX;
    double dy = y - g_lastY;
    g_lastX = x;
    g_lastY = y;

    // Right-drag => pan
    if(g_rmb) {
        float factor = 1.0f/g_scale * 0.002f;
        // aspect ratio compensation
        g_centerX -= (float)dx * factor * ((float)g_width/(float)g_height);
        g_centerY += (float)dy * factor;
    }
}

void scrollCB(GLFWwindow* w, double xoff, double yoff)
{
    float zoomFactor = powf(1.1f, (float)yoff);
    g_scale *= zoomFactor;
    if(g_scale < 1e-5f) g_scale = 1e-5f;
    if(g_scale > 1e5f)  g_scale = 1e5f;
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main()
{
    // 1) GLFW init
    if(!glfwInit()) {
        std::cerr << "GLFW init failed\n";
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(g_width, g_height, "Random Complex Numbers (CUDA+OpenGL)", nullptr, nullptr);
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

    // 2) GLEW init
    glewExperimental = GL_TRUE;
    if( glewInit() != GLEW_OK ) {
        std::cerr << "GLEW init failed\n";
        return -1;
    }

    // 3) Create VBO
    glGenBuffers(1, &g_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
    // We'll store 2 floats per point => total size
    size_t bufferSize = g_numPoints * sizeof(float2);
    glBufferData(GL_ARRAY_BUFFER, bufferSize, nullptr, GL_DYNAMIC_DRAW);

    // Register with CUDA
    cudaGraphicsGLRegisterBuffer(&g_cudaRes, g_vbo, cudaGraphicsMapFlagsWriteDiscard);

    // 4) Setup curand states
    cudaMalloc(&g_curandStates, g_numPoints*sizeof(curandState));
    {
        int blockSize = 256;
        int gridSize  = (g_numPoints + blockSize -1)/blockSize;
        setupCurandStates<<<gridSize, blockSize>>>(g_curandStates, 1234u);
        cudaDeviceSynchronize();
    }

    // 5) Create VAO & set up attribute
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    // bind the VBO
    glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
    glEnableVertexAttribArray(0);
    // each vertex => 2 floats => (x,y)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float2), (void*)0);

    glBindVertexArray(0);

    // 6) Create shader
    GLuint prog = createShaderProgram(vsSource, fsSource);
    glUseProgram(prog);
    GLint mvpLoc = glGetUniformLocation(prog, "u_mvp");

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);

    // Main Loop
    while(!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        // 1) Generate new random points on GPU each frame (or do it once if you prefer static)
        {
            cudaGraphicsMapResources(1, &g_cudaRes, 0);
            float2* dPtr = nullptr;
            size_t   size=0;
            cudaGraphicsResourceGetMappedPointer((void**)&dPtr, &size, g_cudaRes);

            int blockSize=256;
            int gridSize =(g_numPoints + blockSize-1)/blockSize;
            randomPointsKernel<<<gridSize, blockSize>>>(g_curandStates, dPtr, g_numPoints);
            cudaDeviceSynchronize();

            cudaGraphicsUnmapResources(1, &g_cudaRes, 0);
        }

        // 2) Clear screen
        glClearColor(0.0f, 0.0f, 0.0f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // 3) Build MVP for 2D
        //    We'll do an orthographic projection with pan+zoom
        float aspect = (float)g_width/(float)g_height;
        // We'll use glm to create an orthographic projection
        // that covers [-aspect..aspect] x [-1..1], scaled by g_scale,
        // and then translated by g_centerX, g_centerY.
        // Or we can do a simpler approach, but let's do a nice transform:
        glm::mat4 proj = glm::ortho(-aspect, aspect, -1.f, 1.f, -1.f, 1.f);
        // scale
        proj = glm::scale(proj, glm::vec3(1.f/g_scale, 1.f/g_scale, 1.f));
        // translate
        proj = glm::translate(proj, glm::vec3(-g_centerX, -g_centerY, 0.f));

        glUseProgram(prog);
        glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, glm::value_ptr(proj));

        // 4) Draw
        glBindVertexArray(vao);
        glDrawArrays(GL_POINTS, 0, g_numPoints);
        glBindVertexArray(0);

        glfwSwapBuffers(window);
    }

    // Cleanup
    cudaGraphicsUnregisterResource(g_cudaRes);
    cudaFree(g_curandStates);

    glDeleteBuffers(1, &g_vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(prog);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
