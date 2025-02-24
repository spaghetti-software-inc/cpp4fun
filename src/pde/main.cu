#include <iostream>
#include <cmath>
#include <vector>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// For convenience
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ----------------------------------------------------------------------------
// Grid dimensions for the PDE solver
// ----------------------------------------------------------------------------
static const int Nx = 200;  // Number of cells in X
static const int Ny = 200;  // Number of cells in Y

// Space step
static const float dx = 1.0f; 
// Thermal diffusivity
static const float alpha = 1.0f;  
// We pick a dt that satisfies the stability condition: dt < dx^2/(4*alpha)
// so let's pick something comfortably smaller:
static const float dt = 0.1f;  

// We'll store temperature in 2D arrays, allocated on the GPU.
static float* d_T       = nullptr; // current temperature
static float* d_Tnext   = nullptr; // next time step

// ----------------------------------------------------------------------------
// PBO / Texture for visualization
// ----------------------------------------------------------------------------
static GLuint               g_pbo     = 0;
static cudaGraphicsResource* g_pboRes = nullptr;
static GLuint               g_tex     = 0;

// Window / camera
static int   g_width  = 800;
static int   g_height = 800;
static float g_centerX= Nx*0.5f; // we want to center on domain's middle
static float g_centerY= Ny*0.5f;
static float g_zoom   = 1.0f;    // zoom factor

// Input
static bool  g_lmb = false;  
static double g_lastX=0.0, g_lastY=0.0;

// ----------------------------------------------------------------------------
// Boundary + Initial Condition
// We'll set T=0 on boundary, a "hot" circle in center at T=1.0
// ----------------------------------------------------------------------------
__global__
void initTemperature(float* T, int Nx, int Ny)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    if(i >= Nx || j >= Ny) return;

    // Distance from center
    float cx = Nx*0.5f;
    float cy = Ny*0.5f;
    float dx = i - cx;
    float dy = j - cy;
    float r2 = dx*dx + dy*dy;

    // If within radius ~ (Nx/4), set T=1, else 0
    float radius = Nx*0.25f;
    float val = (r2 < radius*radius) ? 1.0f : 0.0f;
    T[j*Nx + i] = val;
}

// ----------------------------------------------------------------------------
// Enforce Dirichlet BC: T=0 on edges
// ----------------------------------------------------------------------------
__global__
void applyBoundary(float* T, int Nx, int Ny)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    if(i >= Nx || j >= Ny) return;

    // If on boundary, set T=0
    if(i == 0 || i == Nx-1 || j == 0 || j == Ny-1) {
        T[j*Nx + i] = 0.0f;
    }
}

// ----------------------------------------------------------------------------
// Finite difference update Tnext = T + alpha * laplacian(T)
// Explicit method: 
//   Tnext(i,j) = T(i,j) + dt*alpha*( (T(i+1,j)-2T(i,j)+T(i-1,j))/dx^2
//                                  + (T(i,j+1)-2T(i,j)+T(i,j-1))/dx^2 )
// ----------------------------------------------------------------------------
__global__
void stepHeatEquation(const float* T, float* Tnext,
                      int Nx, int Ny, float alpha, float dx, float dt)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    if(i < 1 || i >= Nx-1 || j < 1 || j >= Ny-1) return;

    float Tij = T[j*Nx + i];
    float T_left  = T[j*Nx + (i-1)];
    float T_right = T[j*Nx + (i+1)];
    float T_down  = T[(j-1)*Nx + i];
    float T_up    = T[(j+1)*Nx + i];

    float laplacian = (T_left - 2.0f*Tij + T_right)/(dx*dx)
                    + (T_down - 2.0f*Tij + T_up)/(dx*dx);

    float val = Tij + dt*alpha*laplacian;
    Tnext[j*Nx + i] = val;
}

// ----------------------------------------------------------------------------
// Swap references (T <-> Tnext) without extra copying
// ----------------------------------------------------------------------------
static void swapTemps(float*& dA, float*& dB)
{
    float* tmp = dA;
    dA = dB;
    dB = tmp;
}

// ----------------------------------------------------------------------------
// Color mapping kernel: map [0..1..higher?] to RGBA
// We'll do a simple clamp between [0..1] and blend from blue to red
// ----------------------------------------------------------------------------
__device__ uchar4 tempToColor(float t)
{
    // clamp
    if(t < 0.0f) t=0.0f;
    if(t > 1.0f) t=1.0f;
    // simple gradient: t=0 => blue, t=1 => red
    float r = t;
    float g = 0.0f;
    float b = 1.0f - t;
    uchar4 c;
    c.x = (unsigned char)(r*255);
    c.y = (unsigned char)(g*255);
    c.z = (unsigned char)(b*255);
    c.w = 255;
    return c;
}

// This kernel copies T(i,j) into the PBO (with Nx,Ny => width,height)
__global__
void copyToPBO(uchar4* pbo, const float* T, int Nx, int Ny,
               float zoom, float centerX, float centerY, int outWidth, int outHeight)
{
    int px = blockIdx.x*blockDim.x + threadIdx.x;
    int py = blockIdx.y*blockDim.y + threadIdx.y;
    if(px >= outWidth || py >= outHeight) return;

    // We'll treat the screen as a subregion in [0..Nx, 0..Ny],
    // scaled by zoom and centered at (centerX, centerY).
    // So (px,py) => (i,j) in [0..Nx)
    float fx = (px - outWidth*0.5f)/(zoom) + centerX;
    float fy = (py - outHeight*0.5f)/(zoom) + centerY;

    // We'll do a simple nearest-neighbor lookup
    int i = (int)floorf(fx + 0.5f);
    int j = (int)floorf(fy + 0.5f);

    uchar4 color = make_uchar4(0,0,0,255); // default black
    if(i >=0 && i<Nx && j>=0 && j<Ny) {
        float val = T[j*Nx + i];
        color = tempToColor(val);
    }
    pbo[py*outWidth + px] = color;
}

// ----------------------------------------------------------------------------
// OpenGL/GLFW callbacks etc.
// ----------------------------------------------------------------------------
static void framebufferSizeCB(GLFWwindow* w, int width, int height)
{
    g_width  = width;
    g_height = height;
    glViewport(0,0,width,height);
}

static void mouseButtonCB(GLFWwindow* w, int button, int action, int mods)
{
    if(button == GLFW_MOUSE_BUTTON_LEFT) {
        g_lmb = (action == GLFW_PRESS);
        glfwGetCursorPos(w, &g_lastX, &g_lastY);
    }
}

static void cursorPosCB(GLFWwindow* w, double x, double y)
{
    if(!g_lmb) return;
    double dx = x - g_lastX;
    double dy = y - g_lastY;
    g_lastX = x;
    g_lastY = y;

    // Pan
    g_centerX -= (float)dx / g_zoom;
    g_centerY += (float)dy / g_zoom;
}

static void scrollCB(GLFWwindow* w, double xoff, double yoff)
{
    // Zoom in/out
    float factor = powf(1.1f, (float)yoff);
    g_zoom *= factor;
    if(g_zoom < 0.1f) g_zoom=0.1f;
}

static void keyCB(GLFWwindow* w, int key, int scancode, int action, int mods)
{
    if(action == GLFW_PRESS)
    {
        if(key == GLFW_KEY_ESCAPE) {
            glfwSetWindowShouldClose(w, true);
        }
        else if(key == GLFW_KEY_R) {
            // Re-init temperature
            dim3 block(16,16);
            dim3 grid( (Nx+block.x-1)/block.x,
                       (Ny+block.y-1)/block.y);
            initTemperature<<<grid,block>>>(d_T, Nx, Ny);
            cudaDeviceSynchronize();
            std::cout << "Reset simulation.\n";
        }
    }
}

// ----------------------------------------------------------------------------
// Simple fullscreen quad rendering (PBO -> texture -> quad)
// ----------------------------------------------------------------------------
static const char* vsSource = R"(
#version 330 core
layout(location=0) in vec2 aPos;
out vec2 vTexCoord;
void main()
{
    vTexCoord = (aPos+1.0)*0.5;
    gl_Position = vec4(aPos,0.0,1.0);
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

static GLuint compileShader(GLenum type, const char* src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s,1,&src,nullptr);
    glCompileShader(s);
    GLint success;
    glGetShaderiv(s, GL_COMPILE_STATUS, &success);
    if(!success) {
        char log[512]; 
        glGetShaderInfoLog(s,512,nullptr,log);
        std::cerr<<"Shader error:\n"<<log<<"\n";
    }
    return s;
}

static GLuint createProg(const char* vtx, const char* frg)
{
    GLuint vs = compileShader(GL_VERTEX_SHADER,vtx);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER,frg);
    GLuint p = glCreateProgram();
    glAttachShader(p,vs);
    glAttachShader(p,fs);
    glLinkProgram(p);
    glDeleteShader(vs);
    glDeleteShader(fs);
    return p;
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------
int main()
{
    // 1) Init GLFW
    if(!glfwInit()) {
        std::cerr<<"GLFW init failed\n";
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(g_width,g_height,"Heat Equation (CUDA+OpenGL)",nullptr,nullptr);
    if(!window){
        std::cerr<<"Window creation failed\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCB);
    glfwSetMouseButtonCallback(window, mouseButtonCB);
    glfwSetCursorPosCallback(window, cursorPosCB);
    glfwSetScrollCallback(window, scrollCB);
    glfwSetKeyCallback(window, keyCB);

    // 2) Init GLEW
    glewExperimental = GL_TRUE;
    if( glewInit()!=GLEW_OK ){
        std::cerr<<"GLEW init failed\n";
        return -1;
    }

    // 3) Allocate device arrays: T, Tnext
    cudaMalloc((void**)&d_T,     Nx*Ny*sizeof(float));
    cudaMalloc((void**)&d_Tnext, Nx*Ny*sizeof(float));

    // 4) Init temperature
    {
        dim3 block(16,16);
        dim3 grid((Nx+block.x-1)/block.x,(Ny+block.y-1)/block.y);
        initTemperature<<<grid,block>>>(d_T, Nx, Ny);
        cudaDeviceSynchronize();
    }

    // 5) Create PBO + register with CUDA
    glGenBuffers(1, &g_pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, g_pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, g_width*g_height*4, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    cudaGraphicsGLRegisterBuffer(&g_pboRes, g_pbo, cudaGraphicsMapFlagsWriteDiscard);

    // 6) Create texture
    glGenTextures(1, &g_tex);
    glBindTexture(GL_TEXTURE_2D, g_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // 7) Fullscreen quad setup
    float quadVerts[8] = {-1,-1,  1,-1, -1,1,  1,1};
    GLuint vao,vbo;
    glGenVertexArrays(1,&vao);
    glGenBuffers(1,&vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER,vbo);
    glBufferData(GL_ARRAY_BUFFER,sizeof(quadVerts),quadVerts,GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,0,(void*)0);
    glBindVertexArray(0);

    GLuint prog = createProg(vsSource,fsSource);

    // Main loop
    while(!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        // 1) Evolve PDE a few steps per frame for smoother animation
        //    We'll do e.g. 10 sub-steps each frame
        for(int sub=0; sub<10; sub++){
            // apply boundary condition
            dim3 block(16,16);
            dim3 grid((Nx+block.x-1)/block.x,(Ny+block.y-1)/block.y);
            applyBoundary<<<grid,block>>>(d_T, Nx, Ny);

            // do one step
            stepHeatEquation<<<grid,block>>>(d_T, d_Tnext, Nx, Ny, alpha, dx, dt);
            cudaDeviceSynchronize();

            // swap
            swapTemps(d_T, d_Tnext);
        }

        // 2) Update PBO with new temperature
        cudaGraphicsMapResources(1,&g_pboRes,0);
        uchar4* d_pboData=nullptr;
        size_t pboSize=0;
        cudaGraphicsResourceGetMappedPointer((void**)&d_pboData,&pboSize,g_pboRes);

        // fill PBO
        {
            dim3 block(16,16);
            dim3 grid((g_width+block.x-1)/block.x,
                      (g_height+block.y-1)/block.y);
            copyToPBO<<<grid,block>>>(d_pboData,d_T, Nx,Ny,
                                      g_zoom, g_centerX, g_centerY,
                                      g_width, g_height);
            cudaDeviceSynchronize();
        }

        cudaGraphicsUnmapResources(1,&g_pboRes,0);

        // 3) Upload PBO to texture
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, g_pbo);
        glBindTexture(GL_TEXTURE_2D, g_tex);
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,g_width,g_height,0,
                     GL_RGBA,GL_UNSIGNED_BYTE,(void*)0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER,0);

        // 4) Draw the fullscreen quad
        glClearColor(0,0,0,1);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(prog);
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLE_STRIP,0,4);

        glfwSwapBuffers(window);
    }

    // Cleanup
    cudaGraphicsUnregisterResource(g_pboRes);
    glDeleteBuffers(1,&g_pbo);
    glDeleteTextures(1,&g_tex);
    glDeleteProgram(prog);
    glDeleteBuffers(1,&vbo);
    glDeleteVertexArrays(1,&vao);

    cudaFree(d_T);
    cudaFree(d_Tnext);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
