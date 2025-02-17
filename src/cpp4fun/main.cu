#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// -----------------------------------------------------------------------------
// CUDA kernel: generate a simple sphere with lat/lon
// -----------------------------------------------------------------------------
__global__ void generateSphere(float* positions, int numLat, int numLon, float radius)
{
    int idx         = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPoints = numLat * numLon;
    if (idx >= totalPoints) return;

    int i   = idx / numLon; // lat index
    int j   = idx % numLon; // lon index
    float PI    = 3.1415926535f;
    float lat   = (float)i / (numLat - 1) * PI;         // 0..PI
    float lon   = (float)j / (numLon) * 2.0f * PI;      // 0..2PI

    float x = radius * sinf(lat) * cosf(lon);
    float y = radius * cosf(lat);
    float z = radius * sinf(lat) * sinf(lon);

    positions[3 * idx + 0] = x;
    positions[3 * idx + 1] = y;
    positions[3 * idx + 2] = z;
}

// -----------------------------------------------------------------------------
// GLSL
// -----------------------------------------------------------------------------
static const char* vsSource = R"(
#version 330 core
layout (location = 0) in vec3 inPos;
uniform mat4 u_mvp;

void main()
{
    gl_PointSize = 5.0;
    gl_Position  = u_mvp * vec4(inPos, 1.0);
}
)";

static const char* fsSource = R"(
#version 330 core
out vec4 FragColor;
void main()
{
    FragColor = vec4(1, 1, 1, 1);
}
)";

// -----------------------------------------------------------------------------
// Compile & link shader
// -----------------------------------------------------------------------------
static GLuint compileShader(GLenum type, const char* src) {
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

static GLuint createShaderProgram(const char* vtx, const char* frg) {
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
// Globals for camera
// -----------------------------------------------------------------------------
static int   g_width=800, g_height=600;
static float g_yaw=0.f, g_pitch=0.f, g_dist=4.f;
static bool  g_lmb=false, g_rmb=false;
static double g_lastX=0.f, g_lastY=0.f;

void framebuffer_size_callback(GLFWwindow* w, int width, int height) {
    g_width  = width;
    g_height = height;
    glViewport(0, 0, width, height);
}

void mouseButtonCB(GLFWwindow* w, int button, int action, int mods) {
    if(button == GLFW_MOUSE_BUTTON_LEFT) {
        g_lmb = (action == GLFW_PRESS);
        glfwGetCursorPos(w, &g_lastX, &g_lastY);
    } else if(button == GLFW_MOUSE_BUTTON_RIGHT) {
        g_rmb = (action == GLFW_PRESS);
        glfwGetCursorPos(w, &g_lastX, &g_lastY);
    }
}

void cursorPosCB(GLFWwindow* w, double x, double y) {
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

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main() {
    // 1. Init GLFW
    if(!glfwInit()) {
        std::cerr << "GLFW init failed\n";
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(g_width, g_height, "Sphere", nullptr, nullptr);
    if(!window) {
        std::cerr << "Window creation failed\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // Callbacks
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetMouseButtonCallback(window, mouseButtonCB);
    glfwSetCursorPosCallback(window, cursorPosCB);

    // 2. Init GLEW
    if( glewInit() != GLEW_OK ) {
        std::cerr << "GLEW init failed\n";
        return -1;
    }

    // 3. Build VBO
    int numLat=64, numLon=128;
    int numPoints = numLat * numLon;
    size_t bufferSize = size_t(numPoints)*3*sizeof(float);

    GLuint vbo;
    glGenBuffers(1, &vbo);

    // 4. Create VAO (REQUIRED in core profile!)
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, bufferSize, nullptr, GL_DYNAMIC_DRAW);

    // 5. Register with CUDA
    cudaGraphicsResource* cudaRes;
    cudaGraphicsGLRegisterBuffer(&cudaRes, vbo, cudaGraphicsMapFlagsWriteDiscard);

    // 6. Use CUDA to fill buffer
    {
        cudaGraphicsMapResources(1, &cudaRes, 0);
        float* dPtr = nullptr;
        size_t sz   = 0;
        cudaGraphicsResourceGetMappedPointer((void**)&dPtr, &sz, cudaRes);

        int blockSize=256;
        int gridSize =(numPoints + blockSize-1)/blockSize;
        generateSphere<<<gridSize, blockSize>>>(dPtr, numLat, numLon, 1.f);
        cudaDeviceSynchronize();

        cudaGraphicsUnmapResources(1, &cudaRes, 0);
    }

    // 7. Setup Vertex Attribute
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), (void*)0);

    // (Unbind VAO to be safe)
    glBindVertexArray(0);

    // 8. Create/Use Shader
    GLuint prog = createShaderProgram(vsSource, fsSource);
    glUseProgram(prog);

    // Uniform location
    GLint mvpLoc = glGetUniformLocation(prog, "u_mvp");

    // 9. Optional: enable "programmable point size"
    glEnable(GL_PROGRAM_POINT_SIZE);

    // Depth test
    glEnable(GL_DEPTH_TEST);

    // Check errors up to this point
    {
        GLenum err;
        while((err=glGetError())!=GL_NO_ERROR) {
            std::cerr<<"OpenGL init error: "<<err<<"\n";
        }
    }

    // 10. Main Loop
    while(!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        glClearColor(0.1f,0.15f,0.2f,1.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Build MVP with GLM
        float aspect = float(g_width)/float(g_height);
        auto proj  = glm::perspective(glm::radians(45.f), aspect, 0.1f, 100.f);
        auto view  = glm::translate(glm::mat4(1.f), glm::vec3(0, 0, -g_dist));
        view       = glm::rotate(view, glm::radians(g_pitch), glm::vec3(1,0,0));
        view       = glm::rotate(view, glm::radians(g_yaw),   glm::vec3(0,1,0));
        auto model = glm::mat4(1.f);

        glm::mat4 mvp = proj * view * model;

        // Use shader
        glUseProgram(prog);
        glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, glm::value_ptr(mvp));

        // Draw
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
