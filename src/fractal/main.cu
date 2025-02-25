#include <iostream>
#include <cmath>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// For convenience, if you want matrix transforms (orthographic etc.):
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// -----------------------------------------------------------------------------
// Window / Camera parameters
// -----------------------------------------------------------------------------
static int   g_width  = 800;
static int   g_height = 600;

// Mandelbrot "camera"
static double g_centerX = -0.5;  // Typically a good default center is around (-0.5, 0)
static double g_centerY =  0.0; 
static double g_scale   =  3.0;  // The width of our view in the complex plane

static bool   g_rmb     = false;
static double g_lastX   = 0.0;
static double g_lastY   = 0.0;

// -----------------------------------------------------------------------------

// Forward declarations
static void  keyCB       (GLFWwindow*, int, int, int, int);
static void  framebufferCB(GLFWwindow*, int, int);
static void  mouseButtonCB(GLFWwindow*, int, int, int);
static void  cursorPosCB (GLFWwindow*, double, double);
static void  scrollCB    (GLFWwindow*, double, double);

// -----------------------------------------------------------------------------
// Simple passthrough shaders to draw a fullscreen quad
// -----------------------------------------------------------------------------
static const char* s_vsSource = R"(
#version 330 core
layout(location = 0) in vec2 aPos;
out vec2 vTexCoord;

void main()
{
    // Map [-1..1] clip coords to [0..1] texture coords
    vTexCoord = (aPos + vec2(1.0,1.0))*0.5;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)";

static const char* s_fsSource = R"(
#version 330 core
in vec2 vTexCoord;
out vec4 FragColor;

uniform sampler2D uTex;

void main()
{
    FragColor = texture(uTex, vTexCoord);
}
)";

// Shader compile/link utility
static GLuint createShaderProgram(const char* vsSrc, const char* fsSrc)
{
    auto compile = [&](GLenum type, const char* src) {
        GLuint s = glCreateShader(type);
        glShaderSource(s, 1, &src, nullptr);
        glCompileShader(s);

        GLint success=0;
        glGetShaderiv(s, GL_COMPILE_STATUS, &success);
        if(!success) {
            char log[512];
            glGetShaderInfoLog(s, 512, nullptr, log);
            std::cerr << ((type==GL_VERTEX_SHADER)?"Vertex":"Fragment") 
                      << " shader error:\n" << log << std::endl;
        }
        return s;
    };

    GLuint vs = compile(GL_VERTEX_SHADER, vsSrc);
    GLuint fs = compile(GL_FRAGMENT_SHADER, fsSrc);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);

    GLint success=0;
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
// CUDA kernel for the Mandelbrot Set
// -----------------------------------------------------------------------------

// A simple function to map the iteration count to an RGB color.
__device__ uchar4 iterationToColor(int iter, int maxIter)
{
    if(iter >= maxIter) {
        // A point inside the set (we'll color it black)
        return make_uchar4(0,0,0,255);
    } else {
        // Color mapping approach:
        // We'll interpret iteration as an integer in [0..maxIter).
        // A simple approach: hue = iter / maxIter. Let's use some
        // sine-based function or typical "smooth" coloring. But for now,
        // let's do a quick approach with a hue ramp.

        float t = float(iter)/float(maxIter); 
        // Use a simple gradient: color transitions from blue to yellow, for instance.

        // We'll do a 3-phase approach: from blue(0,0,1) to magenta(1,0,1) to yellow(1,1,0)
        // or you can do a full HSV approach.
        // For brevity, let's do an easy hue style with sin/cos:
        float r = 0.5f + 0.5f*cosf(6.2831853f * t);
        float g = 0.5f + 0.5f*cosf(6.2831853f * (t+0.3333f));
        float b = 0.5f + 0.5f*cosf(6.2831853f * (t+0.6667f));

        uchar4 c;
        c.x = (unsigned char)(__saturatef(r)*255.f);
        c.y = (unsigned char)(__saturatef(g)*255.f);
        c.z = (unsigned char)(__saturatef(b)*255.f);
        c.w = 255;
        return c;
    }
}

__global__
void mandelbrotKernel(uchar4* outColor, int width, int height,
                      double centerX, double centerY, double scale)
{
    // Each pixel => one thread
    int px = blockDim.x*blockIdx.x + threadIdx.x;
    int py = blockDim.y*blockIdx.y + threadIdx.y;
    if(px >= width || py >= height) return;

    // map pixel (px,py) -> complex c
    // range in x ~ [centerX - scale/2, centerX + scale/2]
    // range in y ~ [centerY - (scale*height/width)/2, centerY + ... ]
    double aspect = (double)width/(double)height;
    double halfW  = scale*0.5;
    double halfH  = (scale/aspect)*0.5;

    double x0 = centerX - halfW;
    double x1 = centerX + halfW;
    double y0 = centerY - halfH;
    double y1 = centerY + halfH;

    double u = (double)px/(double)(width -1); // [0..1]
    double v = (double)py/(double)(height-1); // [0..1]

    // complex c
    double cx = x0 + (x1 - x0)*u;
    double cy = y0 + (y1 - y0)*v;

    // typical mandelbrot iteration: z_{n+1} = z_n^2 + c, start z_0=0
    double zx = 0.0;
    double zy = 0.0;

    // iteration limit
    const int maxIter = 256;
    int iter=0;
    while(iter < maxIter) {
        // z^2 = (zx+ i zy)^2 => (zx^2 - zy^2) + i(2 zx zy)
        double zx2 = zx*zx - zy*zy;
        double zy2 = 2.0 * zx * zy;
        zx = zx2 + cx;
        zy = zy2 + cy;

        // check magnitude
        if(zx*zx + zy*zy > 4.0) {
            // diverge
            break;
        }
        iter++;
    }

    // color
    uchar4 c = iterationToColor(iter, maxIter);
    outColor[py*width + px] = c;
}

// -----------------------------------------------------------------------------
// Globals for the PBO + Texture
// -----------------------------------------------------------------------------
static GLuint              g_pbo      = 0;
static cudaGraphicsResource* g_pboCuda = nullptr;
static GLuint              g_texture  = 0;
static GLuint              g_prog     = 0;
static GLuint              g_vao      = 0;

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

    GLFWwindow* window = glfwCreateWindow(g_width, g_height, "Mandelbrot Fractal (CUDA+OpenGL)", nullptr, nullptr);
    if(!window) {
        std::cerr << "Window creation failed\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // callbacks
    glfwSetFramebufferSizeCallback(window, framebufferCB);
    glfwSetMouseButtonCallback(window, mouseButtonCB);
    glfwSetCursorPosCallback(window, cursorPosCB);
    glfwSetScrollCallback(window, scrollCB);
    glfwSetKeyCallback(window, keyCB);

    // 2) GLEW
    glewExperimental = GL_TRUE;
    if( glewInit() != GLEW_OK ) {
        std::cerr << "GLEW init failed\n";
        return -1;
    }

    // 3) Create PBO + register with CUDA
    glGenBuffers(1, &g_pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, g_pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, g_width*g_height*4, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    cudaGraphicsGLRegisterBuffer(&g_pboCuda, g_pbo, cudaGraphicsMapFlagsWriteDiscard);

    // 4) Create a texture
    glGenTextures(1, &g_texture);
    glBindTexture(GL_TEXTURE_2D, g_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // 5) Fullscreen quad VAO
    float quad[8] = {
        -1.f, -1.f,
         1.f, -1.f,
        -1.f,  1.f,
         1.f,  1.f
    };
    GLuint vbo;
    glGenVertexArrays(1, &g_vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(g_vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glBindVertexArray(0);

    // 6) Create our shader program
    g_prog = createShaderProgram(s_vsSource, s_fsSource);

    // main loop
    while(!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        // If window size changed, reallocate PBO
        static int oldW = g_width, oldH = g_height;
        if(oldW != g_width || oldH != g_height) {
            oldW = g_width; 
            oldH = g_height;
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, g_pbo);
            glBufferData(GL_PIXEL_UNPACK_BUFFER, g_width*g_height*4, nullptr, GL_DYNAMIC_DRAW);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        }

        // 7) Launch CUDA kernel to fill the PBO with the fractal
        {
            cudaGraphicsMapResources(1, &g_pboCuda, 0);
            uchar4* dPtr = nullptr;
            size_t size=0;
            cudaGraphicsResourceGetMappedPointer((void**)&dPtr, &size, g_pboCuda);

            dim3 block(16,16);
            dim3 grid((g_width+block.x-1)/block.x,
                      (g_height+block.y-1)/block.y);
            mandelbrotKernel<<<grid, block>>>(dPtr, g_width, g_height,
                                              g_centerX, g_centerY, g_scale);
            cudaDeviceSynchronize();

            cudaGraphicsUnmapResources(1, &g_pboCuda, 0);
        }

        // 8) Copy from PBO to the texture
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, g_pbo);
        glBindTexture(GL_TEXTURE_2D, g_texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, g_width, g_height, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // 9) Draw a fullscreen quad
        glClearColor(0,0,0,1);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(g_prog);
        glBindVertexArray(g_vao);
        // If needed: glUniform1i(glGetUniformLocation(g_prog,"uTex"),0);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glBindVertexArray(0);

        glfwSwapBuffers(window);
    }

    // Cleanup
    cudaGraphicsUnregisterResource(g_pboCuda);
    glDeleteBuffers(1, &g_pbo);
    glDeleteTextures(1, &g_texture);
    glDeleteProgram(g_prog);
    glDeleteVertexArrays(1, &g_vao);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

// -----------------------------------------------------------------------------
// Callbacks
// -----------------------------------------------------------------------------
static void keyCB(GLFWwindow* w, int key, int scancode, int action, int mods)
{
    if(action == GLFW_PRESS) {
        if(key == GLFW_KEY_ESCAPE) {
            glfwSetWindowShouldClose(w, true);
        }
        if(key == GLFW_KEY_R) {
            // Reset view
            g_centerX = -0.5;
            g_centerY = 0.0;
            g_scale   = 3.0;
        }
    }
}

static void framebufferCB(GLFWwindow* win, int width, int height)
{
    g_width  = width;
    g_height = height;
    glViewport(0, 0, width, height);
}

static void mouseButtonCB(GLFWwindow* w, int button, int action, int mods)
{
    if(button == GLFW_MOUSE_BUTTON_RIGHT) {
        g_rmb = (action == GLFW_PRESS);
        glfwGetCursorPos(w, &g_lastX, &g_lastY);
    }
}

static void cursorPosCB(GLFWwindow* w, double x, double y)
{
    // Right-drag => pan
    if(g_rmb) {
        double dx = x - g_lastX;
        double dy = y - g_lastY;
        g_lastX = x;
        g_lastY = y;

        // We'll map dx, dy to the scale in complex plane
        double aspect = (double)g_width / (double)g_height;
        double halfW  = g_scale * 0.5;
        double halfH  = (g_scale/aspect)*0.5;

        // Move center by fraction of the window
        g_centerX -= (dx / (double)g_width)  * (2.0*halfW);
        g_centerY += (dy / (double)g_height) * (2.0*halfH);
    }
}

static void scrollCB(GLFWwindow* w, double xoff, double yoff)
{
    // yoff > 0 => zoom in, < 0 => zoom out
    // We'll define a zoom factor
    double zoomFactor = pow(0.9, yoff); // or 1.1^( -yoff)
    g_scale *= zoomFactor;
    if(g_scale < 1e-14) g_scale = 1e-14; // avoid underflow
    if(g_scale > 1e6)   g_scale = 1e6;
}
