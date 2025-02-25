#include <iostream>
#include <cmath>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Optional for matrix transforms
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// -----------------------------------------------------------------------------
// Window / "Camera" parameters
// -----------------------------------------------------------------------------
static int   g_width  = 800;
static int   g_height = 600;

// Fractal "camera" or complex-plane view
static double g_centerX = -0.5;  // typical center for mandelbrot
static double g_centerY =  0.0;
static double g_scale   =  3.0;  // the width of our view in the complex plane

// Mouse interactions
static bool   g_rmb   = false;
static double g_lastX = 0.0;
static double g_lastY = 0.0;

// Which fractal are we showing? (1=Mandelbrot, 2=Julia, 3=Burning Ship, 4=Tricorn, 5=Celtic, 6=Newton)
static int    g_fractalID = 1;

// We'll pick a default Julia parameter:
static double g_juliaCx   = -0.7;
static double g_juliaCy   =  0.27015;

// -----------------------------------------------------------------------------
// Shaders: fullscreen quad
// -----------------------------------------------------------------------------
static const char* s_vsSource = R"(
#version 330 core
layout(location=0) in vec2 aPos;
out vec2 vTexCoord;

void main()
{
    vTexCoord = (aPos + vec2(1.0)) * 0.5; // map [-1..1] to [0..1]
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

// -----------------------------------------------------------------------------
// Convert an HSV color to RGBA. We'll do a full 0..1 hue range for iteration.
// -----------------------------------------------------------------------------
__device__ uchar4 hsvToRGBA(float h, float s, float v)
{
    // h in [0, 1], s in [0, 1], v in [0, 1]
    float H = h * 360.0f; // Hue in degrees
    float C = v * s;
    float X = C * (1.0f - fabsf(fmodf(H / 60.0f, 2.0f) - 1.0f));
    float m = v - C;

    float r, g, b;
    if      (H < 60)  { r = C; g = X; b = 0;   }
    else if (H < 120) { r = X; g = C; b = 0;   }
    else if (H < 180) { r = 0;   g = C; b = X; }
    else if (H < 240) { r = 0;   g = X; b = C; }
    else if (H < 300) { r = X; g = 0;   b = C; }
    else              { r = C; g = 0;   b = X; }
    
    r += m; g += m; b += m;
    r = fminf(fmaxf(r, 0.0f), 1.0f);
    g = fminf(fmaxf(g, 0.0f), 1.0f);
    b = fminf(fmaxf(b, 0.0f), 1.0f);

    return make_uchar4(
        static_cast<unsigned char>(r * 255.0f),
        static_cast<unsigned char>(g * 255.0f),
        static_cast<unsigned char>(b * 255.0f),
        255
    );
}

// -----------------------------------------------------------------------------
// GPU color mapping for iteration counts: Full HSV rainbow
// -----------------------------------------------------------------------------
__device__ uchar4 iterationToColor(int iter, int maxIter)
{
    if (iter >= maxIter) {
        // Inside fractal => black
        return make_uchar4(0, 0, 0, 255);
    } else {
        // Map iteration [0..maxIter-1] into [0..1] for a full HSV rainbow
        float t = float(iter) / float(maxIter - 1);
        return hsvToRGBA(t, 1.0f, 1.0f); 
    }
}

// -----------------------------------------------------------------------------
// Multi-fractal device function
//   Returns the iteration count for (x,y), depending on fractalID
// -----------------------------------------------------------------------------
__device__ int fractalCompute(double x, double y, int fractalID,
                              double juliaCx, double juliaCy)
{
    const int maxIter = 256;

    // Some helper lambdas to keep code DRY:
    auto magnitudeSqr = [](double a, double b) {
        return a*a + b*b;
    };

    switch(fractalID)
    {
    case 1:
    {
        // Mandelbrot
        double zx = 0.0, zy = 0.0;
        int iter = 0;
        while(iter < maxIter)
        {
            double zx2 = zx*zx - zy*zy + x;
            double zy2 = 2.0*zx*zy + y;
            zx = zx2; 
            zy = zy2;
            if (magnitudeSqr(zx, zy) > 4.0) break;
            iter++;
        }
        return iter;
    }
    case 2:
    {
        // Julia
        double zx = x;
        double zy = y;
        int iter = 0;
        while(iter < maxIter)
        {
            double zx2 = zx*zx - zy*zy + juliaCx;
            double zy2 = 2.0*zx*zy + juliaCy;
            zx = zx2; 
            zy = zy2;
            if (magnitudeSqr(zx, zy) > 4.0) break;
            iter++;
        }
        return iter;
    }
    case 3:
    {
        // Burning Ship
        double zx = 0.0, zy = 0.0;
        int iter = 0;
        while(iter < maxIter)
        {
            double ax = fabs(zx);
            double ay = fabs(zy);
            double zx2 = ax*ax - ay*ay + x;
            double zy2 = 2.0*ax*ay + y;
            zx = zx2; 
            zy = zy2;
            if (magnitudeSqr(zx, zy) > 4.0) break;
            iter++;
        }
        return iter;
    }
    case 4:
    {
        // Tricorn: z_{n+1} = conj(z_n)^2 + c
        // conj(z) = (zx, -zy)
        double zx = 0.0, zy = 0.0;
        int iter = 0;
        while(iter < maxIter)
        {
            double zx2 = zx*zx - ( -zy )*( -zy ); // x^2 - (-y)^2 = x^2 - y^2
            double zy2 = 2.0*zx*(-zy);            // 2*x*(-y) = -2xy
            zx = zx2 + x;
            zy = zy2 + y;
            if (magnitudeSqr(zx, zy) > 4.0) break;
            iter++;
        }
        return iter;
    }
    case 5:
    {
        // Celtic: z_{n+1} = (|x^2 - y^2| + 2xy i) + c
        double zx = 0.0, zy = 0.0;
        int iter = 0;
        while(iter < maxIter)
        {
            double zx2 = fabs(zx*zx - zy*zy);
            double zy2 = 2.0 * zx * zy;
            zx = zx2 + x;
            zy = zy2 + y;
            if (magnitudeSqr(zx, zy) > 4.0) break;
            iter++;
        }
        return iter;
    }
    case 6:
    {
        // Newton's method for f(z)=z^3-1
        // z_{n+1} = z_n - f(z_n)/f'(z_n)
        // We treat (x,y) as the initial guess in the complex plane
        double zx = x, zy = y;
        const double EPS = 1e-12;
        for(int i = 0; i < maxIter; i++)
        {
            // f(z) = z^3 - 1
            // f'(z) = 3z^2
            // z^3 => (zx + i zy)^3
            double rx = zx*zx - zy*zy;    // (x^2 - y^2)
            double ry = 2.0*zx*zy;        // (2xy)
            double fx = rx*zx - ry*zy - 1.0; // real part of z^3 - 1
            double fy = rx*zy + ry*zx;    // imag part of z^3
            // derivative 3z^2 => 3*(rx + i ry)
            double dfx = 3.0*rx;
            double dfy = 3.0*ry;
            
            // complex division: (fx + i fy) / (dfx + i dfy)
            double denom = dfx*dfx + dfy*dfy + EPS;
            double nx = (fx*dfx + fy*dfy) / denom;
            double ny = (fy*dfx - fx*dfy) / denom;
            
            zx -= nx;  // new z = z - f(z)/f'(z)
            zy -= ny;
            
            // If close to any root of unity, break
            double mag = sqrt(magnitudeSqr(zx*zx - zy*zy, 2*zx*zy));
            if(fabs(mag - 1.0) < 1e-3)
            {
                return i;
            }
        }
        return maxIter - 1;
    }
    default:
    {
        // Fallback: treat as mandelbrot
        double zx=0.0, zy=0.0;
        int iter=0;
        while(iter<maxIter)
        {
            double zx2 = zx*zx - zy*zy + x;
            double zy2 = 2.0*zx*zy + y;
            zx = zx2; 
            zy = zy2;
            if (magnitudeSqr(zx, zy) > 4.0) break;
            iter++;
        }
        return iter;
    }
    }
}

// -----------------------------------------------------------------------------
// Kernel to compute the chosen fractal into the output pixel buffer
// -----------------------------------------------------------------------------
__global__
void fractalKernel(uchar4* outColor, int width, int height,
                   double centerX, double centerY, double scale,
                   int fractalID,
                   double juliaCx, double juliaCy)
{
    int px = blockDim.x*blockIdx.x + threadIdx.x;
    int py = blockDim.y*blockIdx.y + threadIdx.y;
    if(px >= width || py >= height) return;

    double aspect = (double)width / (double)height;
    double halfW  = scale * 0.5;
    double halfH  = (scale / aspect)*0.5;

    double x0 = centerX - halfW;
    double x1 = centerX + halfW;
    double y0 = centerY - halfH;
    double y1 = centerY + halfH;

    double u = (double)px / (double)(width -1);
    double v = (double)py / (double)(height-1);

    double x = x0 + (x1 - x0)*u;
    double y = y0 + (y1 - y0)*v;

    // compute iteration
    int iter = fractalCompute(x, y, fractalID, juliaCx, juliaCy);

    // map iteration to color
    outColor[py*width + px] = iterationToColor(iter, 256);
}

// -----------------------------------------------------------------------------
// OpenGL + CUDA interop
// -----------------------------------------------------------------------------
static GLuint               g_pbo       = 0;
static cudaGraphicsResource* g_pboCuda  = nullptr;
static GLuint               g_tex       = 0;
static GLuint               g_vao       = 0;
static GLuint               g_prog      = 0;

// -----------------------------------------------------------------------------
// Callbacks
// -----------------------------------------------------------------------------
static void keyCB(GLFWwindow* win, int key, int scancode, int action, int mods)
{
    if(action == GLFW_PRESS) {
        // number keys => set fractal ID
        if(key >= GLFW_KEY_1 && key <= GLFW_KEY_9) {
            g_fractalID = key - GLFW_KEY_0;
            std::cout<<"Switched to fractal ID = "<<g_fractalID<<"\n";
        }
        // R => reset view
        if(key == GLFW_KEY_R) {
            g_centerX = -0.5;
            g_centerY = 0.0;
            g_scale   = 3.0;
        }
        if(key == GLFW_KEY_ESCAPE) {
            glfwSetWindowShouldClose(win, true);
        }
    }
}

static void framebufferCB(GLFWwindow* w, int width, int height)
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
    if(g_rmb) {
        double dx = x - g_lastX;
        double dy = y - g_lastY;
        g_lastX   = x;
        g_lastY   = y;

        double aspect = (double)g_width/(double)g_height;
        double halfW  = g_scale*0.5;
        double halfH  = (g_scale/aspect)*0.5;

        g_centerX -= (dx/(double)g_width)* (2.0*halfW);
        g_centerY += (dy/(double)g_height)*(2.0*halfH);
    }
}
static void scrollCB(GLFWwindow* w, double xoff, double yoff)
{
    // zoom
    double factor = pow(0.9, yoff); 
    g_scale *= factor;
    if(g_scale<1e-15) g_scale=1e-15;
    if(g_scale>1e6)   g_scale=1e6;
}

// -----------------------------------------------------------------------------
// Shader utility
// -----------------------------------------------------------------------------
static GLuint createShaderProg(const char* vs, const char* fs)
{
    auto compile = [&](GLenum type, const char* src){
        GLuint s = glCreateShader(type);
        glShaderSource(s, 1, &src, nullptr);
        glCompileShader(s);
        GLint success=0; 
        glGetShaderiv(s, GL_COMPILE_STATUS, &success);
        if(!success){
            char log[512];
            glGetShaderInfoLog(s, 512, nullptr, log);
            std::cerr<<"Shader error:\n"<<log<<"\n";
        }
        return s;
    };
    GLuint vsO = compile(GL_VERTEX_SHADER, vs);
    GLuint fsO = compile(GL_FRAGMENT_SHADER, fs);
    GLuint p   = glCreateProgram();
    glAttachShader(p, vsO);
    glAttachShader(p, fsO);
    glLinkProgram(p);
    glDeleteShader(vsO);
    glDeleteShader(fsO);
    return p;
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main()
{
    // 1) Init GLFW
    if(!glfwInit()){
        std::cerr<<"GLFW init failed\n";
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(g_width, g_height, "Multi-Fractal (CUDA+OpenGL)", nullptr, nullptr);
    if(!window){
        std::cerr<<"Window creation failed\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    glfwSetFramebufferSizeCallback(window, framebufferCB);
    glfwSetMouseButtonCallback(window, mouseButtonCB);
    glfwSetCursorPosCallback(window, cursorPosCB);
    glfwSetScrollCallback(window, scrollCB);
    glfwSetKeyCallback(window, keyCB);

    // 2) GLEW
    glewExperimental = GL_TRUE;
    if(glewInit() != GLEW_OK){
        std::cerr<<"GLEW init failed\n";
        return -1;
    }

    // 3) Create PBO + register with CUDA
    glGenBuffers(1, &g_pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, g_pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, g_width*g_height*4, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    cudaGraphicsGLRegisterBuffer(&g_pboCuda, g_pbo, cudaGraphicsMapFlagsWriteDiscard);

    // 4) Create a texture to display the PBO
    glGenTextures(1, &g_tex);
    glBindTexture(GL_TEXTURE_2D, g_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // 5) Fullscreen quad
    float quad[8] = {
       -1.f, -1.f,
        1.f, -1.f,
       -1.f,  1.f,
        1.f,  1.f
    };
    glGenVertexArrays(1, &g_vao);
    GLuint vbo;
    glGenBuffers(1, &vbo);

    glBindVertexArray(g_vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glBindVertexArray(0);

    // 6) Create shader
    g_prog = createShaderProg(s_vsSource, s_fsSource);

    // main loop
    while(!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        // handle window resizing
        static int oldW=g_width, oldH=g_height;
        if(oldW!=g_width || oldH!=g_height){
            oldW = g_width; 
            oldH = g_height;
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, g_pbo);
            glBufferData(GL_PIXEL_UNPACK_BUFFER, g_width*g_height*4, nullptr, GL_DYNAMIC_DRAW);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        }

        // 7) Launch CUDA kernel
        {
            cudaGraphicsMapResources(1, &g_pboCuda, 0);
            uchar4* dPtr = nullptr;
            size_t  size=0;
            cudaGraphicsResourceGetMappedPointer((void**)&dPtr, &size, g_pboCuda);

            dim3 block(16,16);
            dim3 grid((g_width+block.x-1)/block.x,
                      (g_height+block.y-1)/block.y);

            fractalKernel<<<grid, block>>>(
                dPtr, g_width, g_height,
                g_centerX, g_centerY, g_scale,
                g_fractalID,
                g_juliaCx, g_juliaCy
            );
            cudaDeviceSynchronize();

            cudaGraphicsUnmapResources(1, &g_pboCuda, 0);
        }

        // 8) Update texture from PBO
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, g_pbo);
        glBindTexture(GL_TEXTURE_2D, g_tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, g_width, g_height, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // 9) Render fullscreen quad
        glClearColor(0,0,0,1);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(g_prog);
        glBindVertexArray(g_vao);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glBindVertexArray(0);

        glfwSwapBuffers(window);
    }

    // cleanup
    cudaGraphicsUnregisterResource(g_pboCuda);
    glDeleteBuffers(1, &g_pbo);
    glDeleteTextures(1, &g_tex);
    glDeleteProgram(g_prog);
    glDeleteVertexArrays(1, &g_vao);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
