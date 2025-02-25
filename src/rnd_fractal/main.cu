#include <iostream>
#include <cmath>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <curand_kernel.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// -----------------------------------------------------------------------------
// Global camera + config
// -----------------------------------------------------------------------------
static int   g_width  = 800;
static int   g_height = 600;

static float g_centerX = 0.f, g_centerY = 0.f;
static float g_scale   = 1.f;
static bool  g_lmb=false, g_rmb=false;
static double g_lastX=0., g_lastY=0.;

// We'll keep the same # of points visible
static const int g_numPoints = 100000;

// We'll store the Sierpinski transforms
// T0(x,y) = (0.5x,       0.5y      )
// T1(x,y) = (0.5(x+1),   0.5y      )
// T2(x,y) = (0.5x,       0.5(y+1)  )

// -----------------------------------------------------------------------------
// Simple pass-through vertex + fragment shader
// We'll color points by angle+radius in the vertex shader
// -----------------------------------------------------------------------------
static const char* vsSource = R"(
#version 330 core
layout(location = 0) in vec2 inPos;

uniform mat4 u_mvp;

out vec4 vColor;

vec3 hsv2rgb(float h, float s, float v)
{
    h = fract(h);
    float r, g, b;
    float hd = h*6.0;
    int   i = int(hd);
    float f = hd - float(i);
    float p = v*(1.0 - s);
    float q = v*(1.0 - f*s);
    float t = v*(1.0 - (1.0 - f)*s);
    if(i==0){ r=v; g=t; b=p; }
    else if(i==1){ r=q; g=v; b=p; }
    else if(i==2){ r=p; g=v; b=t; }
    else if(i==3){ r=p; g=q; b=v; }
    else if(i==4){ r=t; g=p; b=v; }
    else         { r=v; g=p; b=q; }
    return vec3(r,g,b);
}

void main()
{
    gl_PointSize = 2.0;
    gl_Position = u_mvp * vec4(inPos, 0.0, 1.0);

    // color by polar coords
    float x = inPos.x;
    float y = inPos.y;
    float r = sqrt(x*x + y*y);
    float theta = atan(y, x); // [-pi, pi]
    float hue = (theta + 3.1415926535) / (2.0*3.1415926535);
    float saturation = 1.0;
    float brightness = 1.0 - 1.0/(1.0 + r*0.5);

    vColor = vec4(hsv2rgb(hue, saturation, brightness), 1.0);
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
static GLuint compileShader(GLenum type, const char* src)
{
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
}

static GLuint createShaderProgram(const char* vs, const char* fs)
{
    GLuint v = compileShader(GL_VERTEX_SHADER, vs);
    GLuint f = compileShader(GL_FRAGMENT_SHADER, fs);

    GLuint prog = glCreateProgram();
    glAttachShader(prog, v);
    glAttachShader(prog, f);
    glLinkProgram(prog);

    GLint success;
    glGetProgramiv(prog, GL_LINK_STATUS, &success);
    if(!success){
        char log[512];
        glGetProgramInfoLog(prog, 512, nullptr, log);
        std::cerr << "Program link error:\n" << log << std::endl;
    }
    glDeleteShader(v);
    glDeleteShader(f);
    return prog;
}

// -----------------------------------------------------------------------------
// CURAND setup
// -----------------------------------------------------------------------------
__global__ void setupCurandStates(curandState* states, unsigned int seed)
{
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

// -----------------------------------------------------------------------------
// "Big Generator" kernel to fill up to N points that lie in the user’s bounding box
//
// Strategy:
//  - We have a giant array of threads that each do some # of tries (each try does
//    ~some chaos-game iterations).
//  - If the final point is inside the bounding box, we do an atomicAdd on a global
//    counter to get an index in outPoints. If that index < N, we store the point.
//  - If we fill outPoints[] entirely, further successes are ignored.
// -----------------------------------------------------------------------------
//
// bounding box is given by [boxMinX..boxMaxX], [boxMinY..boxMaxY]
//
// One detail: the fractal might be small, so only a fraction of tries end up in the
// bounding box. If the user zooms in a lot, it may take many tries to fill up
// the N points.  We do a "maxTriesPerThread" to avoid infinite loops.  If we
// can't fill in one pass, we’ll do repeated calls from the host until we fill
// or exceed a certain pass count.
//
// Each thread does repeated "chaos game" steps to find a final point. If it’s in
// box => store it. Then it can keep trying until it either hits max tries or
// the array is filled.
//
__global__ void fillFractalInBox(
    curandState* states,
    float2* outPoints,
    int     maxPoints,           // number of points we want total
    float   boxMinX, float boxMaxX,
    float   boxMinY, float boxMaxY,
    int     maxTriesPerThread,
    int*    d_counter            // global atomic counter
)
{
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    curandState rng = states[idx];

    for(int attempt=0; attempt<maxTriesPerThread; attempt++)
    {
        // 1) pick random start (some region that definitely covers fractal,
        //    or just [-1,1]^2 as in the normal chaos game)
        float x = curand_uniform(&rng)*2.f - 1.f;
        float y = curand_uniform(&rng)*2.f - 1.f;

        // 2) apply e.g. 50 chaos steps for Sierpinski
        for(int i=0; i<500; i++){
            float r = curand_uniform(&rng);
            if(r < 1.f/3.f){
                // T0
                x = 0.5f*x;
                y = 0.5f*y;
            }
            else if(r < 2.f/3.f){
                // T1
                x = 0.5f*(x+1.f);
                y = 0.5f*y;
            }
            else {
                // T2
                x = 0.5f*x;
                y = 0.5f*(y+1.f);
            }
        }

        // 3) if final (x,y) in bounding box => attempt to store
        if(x >= boxMinX && x <= boxMaxX &&
           y >= boxMinY && y <= boxMaxY )
        {
            // get next insertion index
            int insertIdx = atomicAdd(d_counter, 1);
            if(insertIdx < maxPoints) {
                // store point
                outPoints[insertIdx] = make_float2(x,y);
            }
            else {
                // array is already full => no need to keep going
                break;
            }
        }
    }

    // store back the updated rng
    states[idx] = rng;
}

// -----------------------------------------------------------------------------
// GPU/GL resources
// -----------------------------------------------------------------------------
static GLuint               g_vbo        = 0;
static cudaGraphicsResource* g_cudaVbo   = nullptr;
static curandState*         g_states     = nullptr;

// We'll keep a small device integer that tracks how many points have been filled
static int* g_dCounter = nullptr;

// -----------------------------------------------------------------------------
// Functions to do the fill on camera changes
// -----------------------------------------------------------------------------

// CPU function that does repeated calls to `fillFractalInBox` until we fill
// up outPoints or we exceed some pass limit.
void fillFractalPointsGPU(float boxMinX, float boxMaxX,
                          float boxMinY, float boxMaxY)
{
    // 1) Reset the global counter
    int zero = 0;
    cudaMemcpy(g_dCounter, &zero, sizeof(int), cudaMemcpyHostToDevice);

    // 2) Map the VBO
    cudaGraphicsMapResources(1, &g_cudaVbo, 0);
    float2* dPtr = nullptr;
    size_t   dSize=0;
    cudaGraphicsResourceGetMappedPointer((void**)&dPtr, &dSize, g_cudaVbo);

    // We'll do multiple passes if needed
    int threads = 256;
    int blocks  = (g_numPoints + threads-1)/threads;
    int passLimit = 20;        // up to 20 passes
    int maxTriesPerThread = 50; // each thread does 50 tries per pass

    for(int pass=0; pass<passLimit; pass++)
    {
        fillFractalInBox<<<blocks, threads>>>(
            g_states,
            dPtr,
            g_numPoints,
            boxMinX, boxMaxX,
            boxMinY, boxMaxY,
            maxTriesPerThread,
            g_dCounter
        );
        cudaDeviceSynchronize();

        // Check how many points so far
        int countSoFar=0;
        cudaMemcpy(&countSoFar, g_dCounter, sizeof(int), cudaMemcpyDeviceToHost);
        if(countSoFar >= g_numPoints) {
            // We have filled the buffer
            break;
        }
    }

    cudaGraphicsUnmapResources(1, &g_cudaVbo, 0);
}

// -----------------------------------------------------------------------------
// GLFW callbacks
// -----------------------------------------------------------------------------
void framebufferCB(GLFWwindow* w, int width, int height)
{
    g_width  = width;
    g_height = height;
    glViewport(0,0,width,height);
}

void mouseButtonCB(GLFWwindow* w, int button, int action, int mods)
{
    if(button==GLFW_MOUSE_BUTTON_LEFT){
        g_lmb = (action==GLFW_PRESS);
        glfwGetCursorPos(w, &g_lastX, &g_lastY);
    }
    else if(button==GLFW_MOUSE_BUTTON_RIGHT){
        g_rmb = (action==GLFW_PRESS);
        glfwGetCursorPos(w, &g_lastX, &g_lastY);
    }
}

void cursorPosCB(GLFWwindow* w, double x, double y)
{
    double dx = x - g_lastX;
    double dy = y - g_lastY;
    g_lastX = x;
    g_lastY = y;

    // Right drag => pan
    if(g_rmb) {
        float factor = 1.f/g_scale*0.002f;
        g_centerX -= (float)dx*factor*( (float)g_width/(float)g_height );
        g_centerY += (float)dy*factor;
    }
}

void scrollCB(GLFWwindow* w, double xoff, double yoff)
{
    float zoomFactor = powf(1.1f, (float)yoff);
    g_scale *= zoomFactor;
    if(g_scale<1e-6f) g_scale=1e-6f;
    if(g_scale>1e6f)  g_scale=1e6f;
}

// -----------------------------------------------------------------------------
// We'll recalculate the fractal each time the user "settles" or
// for demonstration, each frame if the camera changed. Real apps might
// do something more sophisticated to avoid a heavy re-gen on every small movement.
//
// For simplicity, let's do it whenever "the user changes camera" each frame.
// -----------------------------------------------------------------------------
int main()
{
    // 1) Init GLFW
    if(!glfwInit()){
        std::cerr << "GLFW init failed\n";
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(g_width, g_height, "Fractal Zoom - Resample", nullptr, nullptr);
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

    // 2) GLEW
    glewExperimental = true;
    if(glewInit()!=GLEW_OK){
        std::cerr<<"GLEW init failed\n";
        return -1;
    }

    // 3) Create VBO
    glGenBuffers(1, &g_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
    glBufferData(GL_ARRAY_BUFFER, g_numPoints*sizeof(float2), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER,0);

    cudaGraphicsGLRegisterBuffer(&g_cudaVbo, g_vbo, cudaGraphicsMapFlagsWriteDiscard);

    // 4) Curand states
    cudaMalloc(&g_states, g_numPoints*sizeof(curandState));
    int threads=256;
    int blocks=(g_numPoints+threads-1)/threads;
    setupCurandStates<<<blocks,threads>>>(g_states, 1234u);
    cudaDeviceSynchronize();

    // 5) Atomic counter
    cudaMalloc(&g_dCounter, sizeof(int));

    // 6) VAO
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float2), (void*)0);
    glBindVertexArray(0);

    // 7) Shader
    GLuint prog = createShaderProgram(vsSource, fsSource);
    glUseProgram(prog);
    GLint mvpLoc = glGetUniformLocation(prog, "u_mvp");

    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_DEPTH_TEST);

    // do an initial fill (for initial camera)
    {
        // Let's figure out the bounding box for the camera
        // The user sees an orthographic range:
        // x in [centerX - aspect*(1/scale), centerX + aspect*(1/scale)]
        // y in [centerY - (1/scale), centerY + (1/scale)]
        float aspect = (float)g_width/(float)g_height;
        float halfH = 1.f/g_scale;
        float halfW = aspect*halfH;

        float boxMinX = g_centerX - halfW;
        float boxMaxX = g_centerX + halfW;
        float boxMinY = g_centerY - halfH;
        float boxMaxY = g_centerY + halfH;

        fillFractalPointsGPU(boxMinX, boxMaxX, boxMinY, boxMaxY);
    }

    // We'll track old camera to detect changes
    float old_centerX=g_centerX, old_centerY=g_centerY, old_scale=g_scale;

    // Main loop
    while(!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        // If camera changed significantly, re-fill
        if( (fabs(old_centerX - g_centerX) > 1e-7f) ||
            (fabs(old_centerY - g_centerY) > 1e-7f) ||
            (fabs(old_scale   - g_scale)   > 1e-7f) )
        {
            old_centerX=g_centerX;
            old_centerY=g_centerY;
            old_scale  =g_scale;

            float aspect = (float)g_width/(float)g_height;
            float halfH = 1.f/g_scale;
            float halfW = aspect*halfH;

            float boxMinX = g_centerX - halfW;
            float boxMaxX = g_centerX + halfW;
            float boxMinY = g_centerY - halfH;
            float boxMaxY = g_centerY + halfH;

            fillFractalPointsGPU(boxMinX, boxMaxX, boxMinY, boxMaxY);
        }

        // Clear
        glClearColor(0,0,0,1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Build orthographic MVP
        float aspect = (float)g_width/(float)g_height;
        glm::mat4 proj = glm::ortho(-aspect, aspect, -1.f, 1.f, -1.f, 1.f);
        proj = glm::scale(proj, glm::vec3(1.f/g_scale, 1.f/g_scale, 1.f));
        proj = glm::translate(proj, glm::vec3(-g_centerX, -g_centerY, 0.f));

        glUseProgram(prog);
        glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, glm::value_ptr(proj));

        // Draw
        glBindVertexArray(vao);
        glDrawArrays(GL_POINTS, 0, g_numPoints);
        glBindVertexArray(0);

        glfwSwapBuffers(window);
    }

    // Cleanup
    cudaGraphicsUnregisterResource(g_cudaVbo);
    cudaFree(g_states);
    cudaFree(g_dCounter);

    glDeleteBuffers(1, &g_vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(prog);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
