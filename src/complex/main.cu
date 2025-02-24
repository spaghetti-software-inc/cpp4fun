#include <iostream>
#include <cmath>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// For matrix transforms (optional)
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// -----------------------------------------------------------------------------
// Global parameters for camera (pan/zoom/etc.)
// -----------------------------------------------------------------------------
static int   g_width  = 800;
static int   g_height = 600;

// We'll interpret the complex plane so that the center is (cx, cy)
// and each unit in screen is scaled by 'scale'.
static float g_centerX = 0.0f;
static float g_centerY = 0.0f;
static float g_scale   = 1.0f;

static bool  g_lmb=false, g_rmb=false;
static double g_lastX=0.0, g_lastY=0.0;

// Optional rotation around Z (like rotating the plane). 
static float g_angleZ = 0.0f;

// -----------------------------------------------------------------------------
// Simple vertex/fragment shader to draw a fullscreen quad with the PBO as a texture
// -----------------------------------------------------------------------------
static const char* vsSource = R"(
#version 330 core
layout(location = 0) in vec2 aPos;
out vec2 vTexCoord;

void main()
{
    // Fullscreen quad: we directly pass the position [-1..1],
    // and we set the texture coordinate in [0..1].
    vTexCoord = (aPos + vec2(1.0)) * 0.5;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)";

static const char* fsSource = R"(
#version 330 core
in vec2 vTexCoord;
out vec4 FragColor;

uniform sampler2D uTex;

void main()
{
    FragColor = texture(uTex, vTexCoord);
}
)";

// -----------------------------------------------------------------------------
// Utility: compile + link a simple shader program
// -----------------------------------------------------------------------------
GLuint createShaderProgram(const char* vtx, const char* frg)
{
    auto compile = [&](GLenum type, const char* src) {
        GLuint s = glCreateShader(type);
        glShaderSource(s, 1, &src, nullptr);
        glCompileShader(s);
        GLint success;
        glGetShaderiv(s, GL_COMPILE_STATUS, &success);
        if(!success){
            char log[512];
            glGetShaderInfoLog(s, 512, nullptr, log);
            std::cerr << "Shader compile error:\n" << log << std::endl;
        }
        return s;
    };

    GLuint vs = compile(GL_VERTEX_SHADER,   vtx);
    GLuint fs = compile(GL_FRAGMENT_SHADER, frg);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);

    GLint success;
    glGetProgramiv(prog, GL_LINK_STATUS, &success);
    if(!success) {
        char log[512];
        glGetProgramInfoLog(prog, 512, nullptr, log);
        std::cerr << "Shader link error:\n" << log << std::endl;
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

// -----------------------------------------------------------------------------
// CUDA kernel for domain coloring
// For each pixel (px, py):
//   1) Map to complex z in plane
//   2) Compute f(z) = z (or any other function you like)
//   3) Convert to color (H = arg, V = some function of magnitude, etc.)
//   4) Overdraw lines if near radial or angular grid
// -----------------------------------------------------------------------------

// Convert HSV to RGB
__device__ uchar4 hsv2rgb(float h, float s, float v)
{
    // Wrap hue to [0,1)
    h = fmodf(h, 1.0f);
    if(h < 0.0f) h += 1.0f;

    float r,g,b;
    int   i = int(h * 6);
    float f = h * 6 - i;
    float p = v*(1 - s);
    float q = v*(1 - f*s);
    float t = v*(1 - (1 - f)*s);

    switch(i % 6){
        case 0: r=v; g=t; b=p; break;
        case 1: r=q; g=v; b=p; break;
        case 2: r=p; g=v; b=t; break;
        case 3: r=p; g=q; b=v; break;
        case 4: r=t; g=p; b=v; break;
        case 5: r=v; g=p; b=q; break;
    }

    // Convert to 8-bit
    uchar4 c;
    c.x = (unsigned char)(__saturatef(r)*255.0f);
    c.y = (unsigned char)(__saturatef(g)*255.0f);
    c.z = (unsigned char)(__saturatef(b)*255.0f);
    c.w = 255;
    return c;
}

// Domain coloring kernel
__global__ void domainColorKernel(
    uchar4* outPixels, // 4-byte RGBA for each pixel
    int width, int height,
    float centerX, float centerY, 
    float scale,
    float angleZ
)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if(px >= width || py >= height) return;

    // Map pixel to [-1..1], then to the complex plane
    float u = (px + 0.5f) / width  * 2.0f - 1.0f;
    float v = (py + 0.5f) / height * 2.0f - 1.0f;
    // That gives u,v in [-1,1]; now scale to aspect ratio
    float aspect = (float)width / (float)height;
    u *= aspect;

    // (u, v) is the normalized coordinate for the "view".
    // Rotate by angleZ if desired:
    float ca = cosf(angleZ);
    float sa = sinf(angleZ);
    float rx =  u*ca - v*sa;
    float ry =  u*sa + v*ca;

    // Then scale + translate => complex z
    float z_real = rx/scale + centerX;
    float z_imag = ry/scale + centerY;

    // f(z) = z  =>  w_real=z_real, w_imag=z_imag
    float w_real = z_real;
    float w_imag = z_imag;

    // Compute magnitude + angle
    float mag = sqrtf(w_real*w_real + w_imag*w_imag);
    float arg = atan2f(w_imag, w_real); // in [-pi, pi]

    // Convert angle to hue in [0..1]
    float hue = (arg + M_PI)/(2.0f*M_PI); // shift + normalize
    float saturation = 1.0f;
    // Let's pick brightness from the logistic function so we see differences near 0
    float brightness = 1.0f - 1.0f/(1.0f + mag*0.5f);

    // Optionally overlay grid lines:
    // - radial lines if mag is near an integer
    // - angular lines if arg is near multiples of pi/6
    float lineFactor = 1.0f; 
    // radial: distance from integer radius
    float radialDist = fabsf(mag - roundf(mag));
    if(radialDist < 0.01f) {
        // darken color slightly
        lineFactor *= 0.5f; 
    }

    // angular: let's define multiples of pi/6 => arg in [-pi, pi]
    float mul = arg/(M_PI/6.0f); 
    float frac = fabsf(mul - roundf(mul));
    if(frac < 0.005f) {
        lineFactor *= 0.4f;
    }

    brightness *= lineFactor;

    // Convert to RGB
    uchar4 color = hsv2rgb(hue, saturation, brightness);

    // Store
    outPixels[py*width + px] = color;
}

// -----------------------------------------------------------------------------
// PBO variables
// -----------------------------------------------------------------------------
static GLuint          g_pbo       = 0;
static cudaGraphicsResource* g_cudaPbo = nullptr;
static GLuint          g_texture   = 0;

// -----------------------------------------------------------------------------
// GLFW Callbacks
// -----------------------------------------------------------------------------
void framebufferCB(GLFWwindow* win, int w, int h)
{
    g_width  = w;
    g_height = h;
    glViewport(0, 0, w, h);
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

    // Left drag => rotate the plane around Z
    if(g_lmb) {
        g_angleZ += (float)dx * 0.01f;
    }
    // Right drag => pan
    if(g_rmb) {
        // pan in screen coords => shift center
        float factor = 1.0f/g_scale * 0.002f; 
        g_centerX -= (float)dx * factor*( (float)g_width/(float)g_height );
        g_centerY += (float)dy * factor;
    }
}

void scrollCB(GLFWwindow* w, double xoff, double yoff)
{
    // Zoom in/out
    // if yoff>0 => zoom in
    // if yoff<0 => zoom out
    float zoomFactor = powf(1.1f, (float)yoff);
    g_scale *= zoomFactor;
    if(g_scale < 0.00001f) g_scale = 0.00001f;
    if(g_scale > 100000.0f) g_scale = 100000.0f;
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

    GLFWwindow* window = glfwCreateWindow(g_width, g_height, "Domain Coloring f(z)=z", nullptr, nullptr);
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
    glewExperimental = GL_TRUE;
    if( glewInit() != GLEW_OK ) {
        std::cerr << "GLEW init failed\n";
        return -1;
    }

    // 3) Create the PBO + register with CUDA
    glGenBuffers(1, &g_pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, g_pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, g_width*g_height*4, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    cudaGraphicsGLRegisterBuffer(&g_cudaPbo, g_pbo, cudaGraphicsMapFlagsWriteDiscard);

    // 4) Create a texture to draw the PBO onto a fullscreen quad
    glGenTextures(1, &g_texture);
    glBindTexture(GL_TEXTURE_2D, g_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // 5) Create a fullscreen quad VAO
    float quadVerts[8] = {
        -1.f, -1.f,
         1.f, -1.f,
        -1.f,  1.f,
         1.f,  1.f
    };
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVerts), quadVerts, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void*)(0));
    glBindVertexArray(0);

    // 6) Create our shader program
    GLuint prog = createShaderProgram(vsSource, fsSource);

    // Main loop
    while(!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        // Resize PBO if window size changed
        static int oldW = g_width, oldH = g_height;
        if(oldW != g_width || oldH != g_height) {
            oldW = g_width; 
            oldH = g_height;
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, g_pbo);
            glBufferData(GL_PIXEL_UNPACK_BUFFER, g_width*g_height*4, nullptr, GL_DYNAMIC_DRAW);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        }

        // 7) Run CUDA kernel to fill the PBO
        cudaGraphicsMapResources(1, &g_cudaPbo, 0);
        size_t pboSize=0;
        uchar4* dPtr = nullptr;
        cudaGraphicsResourceGetMappedPointer((void**)&dPtr, &pboSize, g_cudaPbo);

        dim3 block(16,16);
        dim3 grid((g_width+block.x-1)/block.x,
                  (g_height+block.y-1)/block.y);
        domainColorKernel<<<grid, block>>>(dPtr, g_width, g_height,
                                           g_centerX, g_centerY,
                                           g_scale, g_angleZ);
        cudaDeviceSynchronize();
        cudaGraphicsUnmapResources(1, &g_cudaPbo, 0);

        // 8) Update the texture from the PBO
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, g_pbo);
        glBindTexture(GL_TEXTURE_2D, g_texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, g_width, g_height, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // 9) Draw the fullscreen quad
        glClearColor(0,0,0,1);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(prog);
        glBindVertexArray(vao);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, g_texture);
        // If needed: glUniform1i(glGetUniformLocation(prog,"uTex"),0);

        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glBindVertexArray(0);

        glfwSwapBuffers(window);
    }

    // Cleanup
    cudaGraphicsUnregisterResource(g_cudaPbo);
    glDeleteBuffers(1, &g_pbo);
    glDeleteTextures(1, &g_texture);
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteProgram(prog);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
