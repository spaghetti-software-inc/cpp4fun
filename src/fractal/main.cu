// fractal_demo.cpp

#include <iostream>
#include <cmath>
#include <vector>
#include <cstring>
#include <cstdio>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// stb_easy_font for quick text overlay (download from stb repository)
#define STB_EASY_FONT_IMPLEMENTATION
#include "stb_easy_font.h"

// -----------------------------------------------------------------------------
// Global Parameters & State
// -----------------------------------------------------------------------------
static int    g_width   = 800;
static int    g_height  = 600;
static double g_centerX = -0.5;
static double g_centerY =  0.0;
static double g_scale   =  3.0;  // view width in complex plane

static bool   g_rmb     = false;
static double g_lastX   = 0.0;
static double g_lastY   = 0.0;

// Fractal type (1: Mandelbrot, 2: Julia, 3: Burning Ship, 4: Tricorn, 5: Celtic, 6: Newton)
static int    g_fractalID = 1;

// Default Julia parameters
static double g_juliaCx   = -0.7;
static double g_juliaCy   =  0.27015;

// Maximum iterations (adjustable via keyboard)
static int    g_maxIter   = 256;

// OpenGL/CUDA interop objects
static GLuint               g_pbo      = 0;
static cudaGraphicsResource* g_pboCuda = nullptr;
static GLuint               g_tex      = 0;
static GLuint               g_vao      = 0;
static GLuint               g_prog     = 0;

// -----------------------------------------------------------------------------
// Shader Sources for Fullscreen Quad (fractal) and Text Overlay
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

// Text overlay shader sources
static const char* text_vsSource = R"(
#version 330 core
layout(location = 0) in vec2 aPos;
uniform mat4 projection;
void main(){
    gl_Position = projection * vec4(aPos, 0.0, 1.0);
}
)";

static const char* text_fsSource = R"(
#version 330 core
out vec4 FragColor;
uniform vec3 textColor;
void main(){
    FragColor = vec4(textColor, 1.0);
}
)";

// -----------------------------------------------------------------------------
// CUDA: Utility Color Mapping
// -----------------------------------------------------------------------------
__device__ uchar4 hsvToRGBA(float h, float s, float v)
{
    float H = h * 360.0f;
    float C = v * s;
    float X = C * (1.0f - fabsf(fmodf(H / 60.0f, 2.0f) - 1.0f));
    float m = v - C;
    float r, g, b;
    if      (H < 60)  { r = C; g = X; b = 0; }
    else if (H < 120) { r = X; g = C; b = 0; }
    else if (H < 180) { r = 0; g = C; b = X; }
    else if (H < 240) { r = 0; g = X; b = C; }
    else if (H < 300) { r = X; g = 0; b = C; }
    else              { r = C; g = 0; b = X; }
    r += m; g += m; b += m;
    return make_uchar4(static_cast<unsigned char>(r * 255.0f),
                       static_cast<unsigned char>(g * 255.0f),
                       static_cast<unsigned char>(b * 255.0f),
                       255);
}

__device__ uchar4 iterationToColor(int iter, int maxIter)
{
    if (iter >= maxIter)
        return make_uchar4(0, 0, 0, 255);
    else {
        float t = float(iter) / float(maxIter - 1);
        return hsvToRGBA(t, 1.0f, 1.0f);
    }
}

// -----------------------------------------------------------------------------
// CUDA: Fractal Compute (uses maxIter passed as parameter)
// -----------------------------------------------------------------------------
__device__ int fractalCompute(double x, double y, int fractalID,
                              double juliaCx, double juliaCy, int maxIter)
{
    auto magnitudeSqr = [](double a, double b) { return a*a + b*b; };
    switch(fractalID)
    {
    case 1: // Mandelbrot
    {
        double zx = 0.0, zy = 0.0;
        int iter = 0;
        while(iter < maxIter)
        {
            double zx2 = zx*zx - zy*zy + x;
            double zy2 = 2.0*zx*zy + y;
            zx = zx2; zy = zy2;
            if (magnitudeSqr(zx, zy) > 4.0) break;
            iter++;
        }
        return iter;
    }
    case 2: // Julia
    {
        double zx = x, zy = y;
        int iter = 0;
        while(iter < maxIter)
        {
            double zx2 = zx*zx - zy*zy + juliaCx;
            double zy2 = 2.0*zx*zy + juliaCy;
            zx = zx2; zy = zy2;
            if (magnitudeSqr(zx, zy) > 4.0) break;
            iter++;
        }
        return iter;
    }
    case 3: // Burning Ship
    {
        double zx = 0.0, zy = 0.0;
        int iter = 0;
        while(iter < maxIter)
        {
            double ax = fabs(zx), ay = fabs(zy);
            double zx2 = ax*ax - ay*ay + x;
            double zy2 = 2.0*ax*ay + y;
            zx = zx2; zy = zy2;
            if (magnitudeSqr(zx, zy) > 4.0) break;
            iter++;
        }
        return iter;
    }
    case 4: // Tricorn
    {
        double zx = 0.0, zy = 0.0;
        int iter = 0;
        while(iter < maxIter)
        {
            double zx2 = zx*zx - ((-zy)*(-zy));
            double zy2 = 2.0*zx*(-zy);
            zx = zx2 + x;
            zy = zy2 + y;
            if (magnitudeSqr(zx, zy) > 4.0) break;
            iter++;
        }
        return iter;
    }
    case 5: // Celtic
    {
        double zx = 0.0, zy = 0.0;
        int iter = 0;
        while(iter < maxIter)
        {
            double zx2 = fabs(zx*zx - zy*zy);
            double zy2 = 2.0*zx*zy;
            zx = zx2 + x;
            zy = zy2 + y;
            if (magnitudeSqr(zx, zy) > 4.0) break;
            iter++;
        }
        return iter;
    }
    case 6: // Newton's method for f(z)=z^3-1
    {
        double zx = x, zy = y;
        const double EPS = 1e-12;
        for(int i = 0; i < maxIter; i++)
        {
            double rx = zx*zx - zy*zy;
            double ry = 2.0*zx*zy;
            double fx = rx*zx - ry*zy - 1.0;
            double fy = rx*zy + ry*zx;
            double dfx = 3.0*rx;
            double dfy = 3.0*ry;
            double denom = dfx*dfx + dfy*dfy + EPS;
            double nx = (fx*dfx + fy*dfy) / denom;
            double ny = (fy*dfx - fx*dfy) / denom;
            zx -= nx; zy -= ny;
            double mag = sqrt(magnitudeSqr(zx*zx - zy*zy, 2*zx*zy));
            if(fabs(mag - 1.0) < 1e-3)
                return i;
        }
        return maxIter - 1;
    }
    default: // Fallback: Mandelbrot
    {
        double zx = 0.0, zy = 0.0;
        int iter = 0;
        while(iter < maxIter)
        {
            double zx2 = zx*zx - zy*zy + x;
            double zy2 = 2.0*zx*zy + y;
            zx = zx2; zy = zy2;
            if (magnitudeSqr(zx, zy) > 4.0) break;
            iter++;
        }
        return iter;
    }
    }
}

// -----------------------------------------------------------------------------
// CUDA: Fractal Kernel (now accepts maxIter parameter)
// -----------------------------------------------------------------------------
__global__
void fractalKernel(uchar4* outColor, int width, int height,
                   double centerX, double centerY, double scale,
                   int fractalID, double juliaCx, double juliaCy, int maxIter)
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

    double u = (double)px / (width - 1);
    double v = (double)py / (height - 1);

    double x = x0 + (x1 - x0)*u;
    double y = y0 + (y1 - y0)*v;

    int iter = fractalCompute(x, y, fractalID, juliaCx, juliaCy, maxIter);
    outColor[py*width + px] = iterationToColor(iter, maxIter);
}

// -----------------------------------------------------------------------------
// Utility: Map fractalID to a name for display
// -----------------------------------------------------------------------------
const char* getFractalName(int id) {
    switch(id) {
        case 1: return "Mandelbrot";
        case 2: return "Julia";
        case 3: return "Burning Ship";
        case 4: return "Tricorn";
        case 5: return "Celtic";
        case 6: return "Newton";
        default: return "Unknown";
    }
}

// -----------------------------------------------------------------------------
// OpenGL: Shader Utility Function
// -----------------------------------------------------------------------------
static GLuint createShaderProg(const char* vs, const char* fs)
{
    auto compile = [&](GLenum type, const char* src) {
        GLuint s = glCreateShader(type);
        glShaderSource(s, 1, &src, nullptr);
        glCompileShader(s);
        GLint success = 0;
        glGetShaderiv(s, GL_COMPILE_STATUS, &success);
        if(!success){
            char log[512];
            glGetShaderInfoLog(s, 512, nullptr, log);
            std::cerr << "Shader error:\n" << log << "\n";
        }
        return s;
    };
    GLuint vsO = compile(GL_VERTEX_SHADER, vs);
    GLuint fsO = compile(GL_FRAGMENT_SHADER, fs);
    GLuint p = glCreateProgram();
    glAttachShader(p, vsO);
    glAttachShader(p, fsO);
    glLinkProgram(p);
    glDeleteShader(vsO);
    glDeleteShader(fsO);
    return p;
}

// -----------------------------------------------------------------------------
// GLFW Callbacks
// -----------------------------------------------------------------------------
static void keyCB(GLFWwindow* win, int key, int scancode, int action, int mods)
{
    if(action == GLFW_PRESS || action == GLFW_REPEAT)
    {
        // Switch fractal type with number keys
        if(key >= GLFW_KEY_1 && key <= GLFW_KEY_9)
        {
            g_fractalID = key - GLFW_KEY_0;
            std::cout << "Switched to fractal ID = " << g_fractalID << "\n";
        }
        // Full reset (R key)
        if(key == GLFW_KEY_R)
        {
            g_centerX = -0.5;
            g_centerY = 0.0;
            g_scale   = 3.0;
        }
        // Arrow keys: pan the view (shift center)
        if(key == GLFW_KEY_LEFT)
            g_centerX -= 0.05 * g_scale;
        if(key == GLFW_KEY_RIGHT)
            g_centerX += 0.05 * g_scale;
        if(key == GLFW_KEY_UP)
            g_centerY += 0.05 * (g_scale / ((double)g_width / g_height));
        if(key == GLFW_KEY_DOWN)
            g_centerY -= 0.05 * (g_scale / ((double)g_width / g_height));
        // Adjust Julia parameters (only if fractalID==2)
        if(g_fractalID == 2)
        {
            if(key == GLFW_KEY_LEFT_BRACKET)
            {
                if(mods & GLFW_MOD_SHIFT)
                    g_juliaCy -= 0.01;
                else
                    g_juliaCx -= 0.01;
            }
            if(key == GLFW_KEY_RIGHT_BRACKET)
            {
                if(mods & GLFW_MOD_SHIFT)
                    g_juliaCy += 0.01;
                else
                    g_juliaCx += 0.01;
            }
        }
        // Adjust maximum iterations (+ / - keys)
        if(key == GLFW_KEY_KP_ADD || key == GLFW_KEY_EQUAL)
        {
            g_maxIter += 10;
            std::cout << "Max Iterations: " << g_maxIter << "\n";
        }
        if(key == GLFW_KEY_KP_SUBTRACT || key == GLFW_KEY_MINUS)
        {
            if(g_maxIter > 10)
            {
                g_maxIter -= 10;
                std::cout << "Max Iterations: " << g_maxIter << "\n";
            }
        }
        if(key == GLFW_KEY_ESCAPE)
            glfwSetWindowShouldClose(win, true);
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
    if(button == GLFW_MOUSE_BUTTON_RIGHT)
    {
        g_rmb = (action == GLFW_PRESS);
        glfwGetCursorPos(w, &g_lastX, &g_lastY);
    }
}

static void cursorPosCB(GLFWwindow* w, double x, double y)
{
    if(g_rmb)
    {
        double dx = x - g_lastX;
        double dy = y - g_lastY;
        g_lastX = x;
        g_lastY = y;
        double aspect = (double)g_width / g_height;
        double halfW  = g_scale * 0.5;
        double halfH  = (g_scale / aspect) * 0.5;
        g_centerX -= (dx / g_width) * (2.0 * halfW);
        g_centerY += (dy / g_height) * (2.0 * halfH);
    }
}

static void scrollCB(GLFWwindow* w, double xoff, double yoff)
{
    double factor = pow(0.9, yoff);
    g_scale *= factor;
    if(g_scale < 1e-15) g_scale = 1e-15;
    if(g_scale > 1e6)   g_scale = 1e6;
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main()
{
    // Initialize GLFW
    if(!glfwInit()){
        std::cerr << "GLFW init failed\n";
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(g_width, g_height, "Multi-Fractal (CUDA+OpenGL)", nullptr, nullptr);
    if(!window){
        std::cerr << "Window creation failed\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebufferCB);
    glfwSetMouseButtonCallback(window, mouseButtonCB);
    glfwSetCursorPosCallback(window, cursorPosCB);
    glfwSetScrollCallback(window, scrollCB);
    glfwSetKeyCallback(window, keyCB);

    glewExperimental = GL_TRUE;
    if(glewInit() != GLEW_OK){
        std::cerr << "GLEW init failed\n";
        return -1;
    }

    // Create Pixel Buffer Object and register with CUDA
    glGenBuffers(1, &g_pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, g_pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, g_width * g_height * 4, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    cudaGraphicsGLRegisterBuffer(&g_pboCuda, g_pbo, cudaGraphicsMapFlagsWriteDiscard);

    // Create texture to display the PBO
    glGenTextures(1, &g_tex);
    glBindTexture(GL_TEXTURE_2D, g_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Fullscreen quad (vertex positions)
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

    // Create main shader program for fractal
    g_prog = createShaderProg(s_vsSource, s_fsSource);

    // --- Setup Text Rendering ---
    GLuint textProgram = createShaderProg(text_vsSource, text_fsSource);
    GLuint textVAO, textVBO;
    glGenVertexArrays(1, &textVAO);
    glGenBuffers(1, &textVBO);
    glBindVertexArray(textVAO);
    glBindBuffer(GL_ARRAY_BUFFER, textVBO);
    // Allocate buffer for up to 10,000 floats (should be plenty for our text)
    glBufferData(GL_ARRAY_BUFFER, 10000 * sizeof(float), NULL, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    // Each vertex from stb_easy_font consists of 4 floats; we use only the first 2 (x,y)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, (void*)0);
    glBindVertexArray(0);

    // Main loop
    while(!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        // Handle window resize for PBO reallocation
        static int oldW = g_width, oldH = g_height;
        if(oldW != g_width || oldH != g_height)
        {
            oldW = g_width; oldH = g_height;
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, g_pbo);
            glBufferData(GL_PIXEL_UNPACK_BUFFER, g_width * g_height * 4, nullptr, GL_DYNAMIC_DRAW);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        }

        // --- Launch CUDA Kernel ---
        cudaGraphicsMapResources(1, &g_pboCuda, 0);
        uchar4* dPtr = nullptr;
        size_t size = 0;
        cudaGraphicsResourceGetMappedPointer((void**)&dPtr, &size, g_pboCuda);

        dim3 block(16, 16);
        dim3 grid((g_width + block.x - 1) / block.x, (g_height + block.y - 1) / block.y);
        fractalKernel<<<grid, block>>>(dPtr, g_width, g_height,
                                       g_centerX, g_centerY, g_scale,
                                       g_fractalID, g_juliaCx, g_juliaCy, g_maxIter);
        cudaDeviceSynchronize();
        cudaGraphicsUnmapResources(1, &g_pboCuda, 0);

        // --- Update Texture from PBO ---
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, g_pbo);
        glBindTexture(GL_TEXTURE_2D, g_tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, g_width, g_height, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // --- Render Fractal Quad ---
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(g_prog);
        glBindVertexArray(g_vao);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glBindVertexArray(0);

        // --- Render Text Overlay ---
        {
            // Setup orthographic projection for text (origin at top-left)
            glm::mat4 proj = glm::ortho(0.0f, (float)g_width, (float)g_height, 0.0f, -1.0f, 1.0f);
            glUseProgram(textProgram);
            GLuint projLoc = glGetUniformLocation(textProgram, "projection");
            glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(proj));
            GLuint colorLoc = glGetUniformLocation(textProgram, "textColor");
            glUniform3f(colorLoc, 1.0f, 1.0f, 1.0f);

            // Get current mouse position and convert to fractal coordinates
            double mouseX, mouseY;
            glfwGetCursorPos(window, &mouseX, &mouseY);
            double aspect = (double)g_width / g_height;
            double halfW = g_scale * 0.5;
            double halfH = (g_scale / aspect) * 0.5;
            double x0 = g_centerX - halfW;
            double y0 = g_centerY - halfH;
            double x1 = g_centerX + halfW;
            double y1 = g_centerY + halfH;
            double u = mouseX / (g_width - 1);
            double v = mouseY / (g_height - 1);
            double mX = x0 + (x1 - x0) * u;
            double mY = y0 + (y1 - y0) * v;

            // Build info string
            char info[256];
            sprintf(info, "Fractal: %s\nZoom: %.5f\nMax Iter: %d\nMouse: (%.5f, %.5f)",
                    getFractalName(g_fractalID), g_scale, g_maxIter, mX, mY);

            // Generate vertex data using stb_easy_font
            char buffer[99999];
            int num_quads = stb_easy_font_print(10, 10, info, NULL, buffer, sizeof(buffer));

            // Convert the quads to triangles (since GL_QUADS is not available in core)
            int numQuads = num_quads; // each quad has 4 vertices
            std::vector<float> vertices;
            vertices.resize(numQuads * 6 * 4); // 6 vertices per quad, 4 floats per vertex
            float* quadVertices = (float*)buffer;
            for (int q = 0; q < numQuads; q++) {
                int base = q * 4;
                float v0[4] = { quadVertices[(base + 0) * 4 + 0], quadVertices[(base + 0) * 4 + 1],
                                 quadVertices[(base + 0) * 4 + 2], quadVertices[(base + 0) * 4 + 3] };
                float v1[4] = { quadVertices[(base + 1) * 4 + 0], quadVertices[(base + 1) * 4 + 1],
                                 quadVertices[(base + 1) * 4 + 2], quadVertices[(base + 1) * 4 + 3] };
                float v2[4] = { quadVertices[(base + 2) * 4 + 0], quadVertices[(base + 2) * 4 + 1],
                                 quadVertices[(base + 2) * 4 + 2], quadVertices[(base + 2) * 4 + 3] };
                float v3[4] = { quadVertices[(base + 3) * 4 + 0], quadVertices[(base + 3) * 4 + 1],
                                 quadVertices[(base + 3) * 4 + 2], quadVertices[(base + 3) * 4 + 3] };
                int offset = q * 6 * 4;
                // First triangle: v0, v1, v2
                memcpy(&vertices[offset + 0], v0, 4*sizeof(float));
                memcpy(&vertices[offset + 4], v1, 4*sizeof(float));
                memcpy(&vertices[offset + 8], v2, 4*sizeof(float));
                // Second triangle: v2, v3, v0
                memcpy(&vertices[offset + 12], v2, 4*sizeof(float));
                memcpy(&vertices[offset + 16], v3, 4*sizeof(float));
                memcpy(&vertices[offset + 20], v0, 4*sizeof(float));
            }
            glBindBuffer(GL_ARRAY_BUFFER, textVBO);
            glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.size() * sizeof(float), vertices.data());
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            glBindVertexArray(textVAO);
            glDrawArrays(GL_TRIANGLES, 0, numQuads * 6);
            glBindVertexArray(0);
        }

        glfwSwapBuffers(window);
    }

    // Cleanup
    cudaGraphicsUnregisterResource(g_pboCuda);
    glDeleteBuffers(1, &g_pbo);
    glDeleteTextures(1, &g_tex);
    glDeleteProgram(g_prog);
    glDeleteVertexArrays(1, &g_vao);

    glDeleteBuffers(1, &textVBO);
    glDeleteVertexArrays(1, &textVAO);
    glDeleteProgram(textProgram);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
