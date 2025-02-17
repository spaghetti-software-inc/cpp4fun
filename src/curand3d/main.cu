#include <iostream>
#include <cmath>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <curand.h>

// GLM
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

// -----------------------------------------------------------------------------
// Kernel: Convert (x,y,z,rRandom) into a point on a sphere with radius ~ N(1,0.2)
// -----------------------------------------------------------------------------
__global__ void finalizePoints(float* data, int numPoints)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;

    // Each point: data[4*idx + 0..3] => (x, y, z, rRandom)
    float x = data[4*idx + 0];
    float y = data[4*idx + 1];
    float z = data[4*idx + 2];
    float r = data[4*idx + 3];

    // Normalize (x,y,z) => direction on unit sphere
    float len = sqrtf(x*x + y*y + z*z);
    if (len > 1e-7f) {
        x /= len;
        y /= len;
        z /= len;
    }

    // Turn the random r into an actual radius. For example:
    //   radius = abs( 1.0 + 0.2 * r )
    // That yields a distribution centered around ~1.0 with stdev ~0.2, 
    // clipped to be non-negative.
    float radius = 1.0f + 0.2f * r;
    if (radius < 0.0f) {
        radius = -radius;  // or clamp to 0 if you prefer
    }

    // Scale direction by radius
    x *= radius;
    y *= radius;
    z *= radius;

    // Write back final position + store the final radius in w
    data[4*idx + 0] = x;
    data[4*idx + 1] = y;
    data[4*idx + 2] = z;
    data[4*idx + 3] = radius;
}

// -----------------------------------------------------------------------------
// GLSL Shaders
//    - Vertex shader uses (x,y,z,r) for position and color generation
//    - Fragment shader takes the color from the vertex shader
// -----------------------------------------------------------------------------
static const char* vsSource = R"(
#version 330 core
layout (location = 0) in vec4 inPosRad; // (x, y, z, radius)

uniform mat4 u_mvp;

out vec4 vColor; // pass color to fragment shader

void main()
{
    // Position
    gl_PointSize = 5.0;
    gl_Position  = u_mvp * vec4(inPosRad.xyz, 1.0);

    // Let's color by radius: gradient from blue to red
    float r = inPosRad.w;
    // We'll map r in [0, 2] => clamp to avoid crazy high or negative
    float t = clamp(r / 2.0, 0.0, 1.0);
    // simple linear blend: (1.0 - t, 0.0, t)
    vColor = vec4(1.0 - t, 0.0, t, 1.0);
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
// Shader utilities
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
// Camera + Mouse
// -----------------------------------------------------------------------------
static int   g_width=800, g_height=600;
static float g_yaw=0.f, g_pitch=0.f, g_dist=4.f;
static bool  g_lmb=false, g_rmb=false;
static double g_lastX=0.f, g_lastY=0.f;

void framebufferCB(GLFWwindow* w, int width, int height)
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
    float dx = float(x) - g_lastX;
    float dy = float(y) - g_lastY;
    g_lastX  = float(x);
    g_lastY  = float(y);

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
// Main
// -----------------------------------------------------------------------------
int main()
{
    // 1) Init GLFW
    if(!glfwInit()) {
        std::cerr << "GLFW init failed\n";
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(g_width, g_height, "Random Sphere w/Radius Colors", nullptr, nullptr);
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

    // 2) Init GLEW
    if( glewInit() != GLEW_OK ) {
        std::cerr << "GLEW init failed\n";
        return -1;
    }

    // 3) Create VBO + VAO
    int numPoints  = 50000;               // # of points
    size_t floatsPerVertex = 4;           // (x, y, z, radius)
    size_t floatCount      = floatsPerVertex * numPoints;
    size_t bufferSize      = floatCount*sizeof(float);

    GLuint vbo;
    glGenBuffers(1, &vbo);

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao); // bind the VAO

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, bufferSize, nullptr, GL_DYNAMIC_DRAW);

    // 4) Register with CUDA
    cudaGraphicsResource* cudaRes;
    cudaGraphicsGLRegisterBuffer(&cudaRes, vbo, cudaGraphicsMapFlagsWriteDiscard);

    // 5) Generate random data with CURAND, then finalize in kernel
    {
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, 12345ULL);

        // Map
        cudaGraphicsMapResources(1, &cudaRes, 0);
        float* dPtr = nullptr;
        size_t dSize = 0;
        cudaGraphicsResourceGetMappedPointer((void**)&dPtr, &dSize, cudaRes);

        // curandGenerateNormal needs an even number of floats
        size_t alignedCount = (floatCount + 1) & ~1; // round up
        curandGenerateNormal(gen, dPtr, alignedCount, 0.0f, 1.0f);

        // Kernel: finalize points => direction + radius
        int blockSize=256;
        int gridSize =(numPoints + blockSize-1)/blockSize;
        finalizePoints<<<gridSize, blockSize>>>(dPtr, numPoints);
        cudaDeviceSynchronize();

        cudaGraphicsUnmapResources(1, &cudaRes, 0);
        curandDestroyGenerator(gen);
    }

    // 6) Setup vertex attribute => (location=0) has 4 floats
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)0);

    glBindVertexArray(0);

    // 7) Create shader + enable program point size
    GLuint prog = createShaderProgram(vsSource, fsSource);
    glUseProgram(prog);
    GLint mvpLoc = glGetUniformLocation(prog, "u_mvp");

    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_DEPTH_TEST);

    // Main loop
    while(!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        glClearColor(0.1f,0.15f,0.2f,1.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Build MVP using GLM
        float aspect = float(g_width)/float(g_height);
        glm::mat4 proj  = glm::perspective(glm::radians(45.f), aspect, 0.1f, 100.f);
        glm::mat4 view  = glm::translate(glm::mat4(1.f), glm::vec3(0, 0, -g_dist));
        view            = glm::rotate(view, glm::radians(g_pitch), glm::vec3(1,0,0));
        view            = glm::rotate(view, glm::radians(g_yaw),   glm::vec3(0,1,0));
        glm::mat4 model = glm::mat4(1.f);

        glm::mat4 mvp   = proj * view * model;

        glUseProgram(prog);
        glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, glm::value_ptr(mvp));

        glBindVertexArray(vao);
        glDrawArrays(GL_POINTS, 0, numPoints);
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
