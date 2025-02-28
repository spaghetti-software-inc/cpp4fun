/******************************************************************************
 *  DoubleSlit_ResetOnPanOrZoom.cu
 *
 *  Demonstrates a double-slit simulation with:
 *    - Time accumulation in a float texture (to gradually build up an image),
 *    - Dynamic Imax computation each frame (to normalize probabilities),
 *    - Resets accumulation when user pans (mouse drag/arrow keys) or zooms.
 *
 *  CONTROLS:
 *    - Mouse drag or Arrow keys: Pan (resets accumulation)
 *    - Mouse wheel: Zoom in/out (resets accumulation)
 *    - I/K: Increase/Decrease intensity boost
 *    - C: Clear accumulation manually
 *    - ESC: Exit
 ******************************************************************************/

// Standard C++ includes for console I/O and math
#include <iostream>     // For std::cerr, std::cout
#include <cmath>        // For sinf, cosf, fabsf, etc.
#include <cstdio>       // For sprintf, printf

// Includes for OpenGL functionality
#include <GL/glew.h>    // GLEW library helps manage modern OpenGL extensions
#include <GLFW/glfw3.h> // GLFW for window/context/input

// CUDA runtime + CURAND for random numbers
#include <cuda_runtime.h>       
#include <curand_kernel.h>      
#include <cuda_gl_interop.h>    // For interop between CUDA and OpenGL

// Text rendering library (header-only, used in-line)
#define STB_EASY_FONT_IMPLEMENTATION
#include "stb_easy_font.h"

// ----------------------------------------------------------
// Simulation / Physical Parameters
// ----------------------------------------------------------

// Default wavelength in meters (0.5e-6 = 500 nm, visible light)
static const float DEFAULT_LAMBDA    = 0.5e-6f;

// Distance between the two slits in meters (1 mm)
static const float DEFAULT_SLIT_DIST = 1.0e-3f;

// The width of each slit in meters (0.2 mm)
static const float DEFAULT_SLIT_WIDTH= 0.2e-3f;

// Distance from the slits to the screen in meters (1 m)
static const float DEFAULT_SCREEN_Z  = 1.0f;

// Half of the total range in x-direction for sampling/visualization. Here ±2 cm
static const float XRANGE           = 0.02f;

// Number of photons to generate and render each frame
static const size_t NUM_POINTS = 100000;

// Typical CUDA block size
static const int    BLOCK_SIZE = 256;

// ----------------------------------------------------------
// Globals for camera/pan/zoom, intensity, window size
// ----------------------------------------------------------

// Variables to track how the user has panned and zoomed in/out
float panX = 0.0f;  // Horizontal shift of the view
float panY = 0.0f;  // Vertical shift of the view
float zoom = 1.0f;  // Zoom factor (1.0 = default, larger = zoomed in)
float intensityBoost = 1.0f; // A user-controlled multiplier for brightness

// Initial dimensions of the window
int windowWidth  = 800;
int windowHeight = 600;

// Mouse tracking variables
bool   mouseDragging = false; // True when left mouse is down
double lastMouseX    = 0.0;   // Last recorded mouse X
double lastMouseY    = 0.0;   // Last recorded mouse Y

// Track last pan & zoom to detect changes (so we can reset accumulation if changed)
float lastPanX = 0.0f;
float lastPanY = 0.0f;
float lastZoom = 1.0f;

// Slit parameters (could be changed dynamically, but here they’re fixed for demo)
float LAMBDA    = DEFAULT_LAMBDA; 
float SLIT_DIST = DEFAULT_SLIT_DIST;
float SLIT_WIDTH= DEFAULT_SLIT_WIDTH;
float SCREEN_Z  = DEFAULT_SCREEN_Z;

// ----------------------------------------------------------
// Device-side double-slit intensity
// ----------------------------------------------------------
/*
  THE NEXT TWO __device__ FUNCTIONS RUN ON THE GPU.

  "sinc^2" (sinc2f) models single-slit diffraction. 
  "doubleSlitIntensity" combines single-slit diffraction with the cos^2 factor
  that arises from the two-slit interference pattern.
*/

__device__ __inline__
float sinc2f(float x)
{
    // If x is extremely small, use limiting value 1.0 
    // to avoid numerical instability (sin(x)/x near zero).
    if (fabsf(x) < 1.0e-7f) return 1.0f;

    float val = sinf(x)/x;
    return val * val; // (sin x / x)^2
}

__device__ __inline__
float doubleSlitIntensity(float x, float wavelength, float d, float a, float z)
{
    // alpha: phase difference from slit separation
    float alpha = M_PI * d * x / (wavelength * z);

    // beta: phase factor from slit width
    float beta  = M_PI * a * x / (wavelength * z);

    // cos(alpha)^2 => two-slit interference
    // sinc2f(beta) => single-slit diffraction envelope
    return cosf(alpha)*cosf(alpha) * sinc2f(beta);
}

// ----------------------------------------------------------
// CPU version for scanning to find max intensity
// ----------------------------------------------------------
/*
  On the CPU side, we do a brute-force sample of the intensity across [-XRANGE, +XRANGE]
  to find the maximum intensity (Imax). This is used for normalization in
  rejection sampling on the GPU.
*/

float doubleSlitIntensityCPU(float x, float wavelength, float d, float a, float z)
{
    // Duplicate of the formula above, but in host code (non-inlined version).
    float alpha = (float)M_PI * d * x / (wavelength * z);
    float beta  = (float)M_PI * a * x / (wavelength * z);

    float c   = cosf(alpha);
    float val = c * c;

    // Single-slit envelope via sinc^2:
    float denom = (fabsf(beta) < 1e-7f) ? 1.0f : beta;
    float s     = sinf(denom)/denom;
    val *= (s*s);

    return val;
}

float computeMaxIntensity(int sampleCount)
{
    /*
      sampleCount determines how many points we test in [-XRANGE, XRANGE].
      We find the peak intensity for the double-slit pattern. 
      This is returned as "maxI" and used on the GPU for acceptance tests.
    */
    float maxI = 0.0f;

    for (int i = 0; i < sampleCount; ++i) {
        // fraction from 0 to 1
        float frac = (float)i / (float)(sampleCount - 1);

        // map that fraction to an x in [-XRANGE, XRANGE]
        float x    = -XRANGE + 2.0f * XRANGE * frac;

        // compute intensity at x
        float I    = doubleSlitIntensityCPU(x, LAMBDA, SLIT_DIST, SLIT_WIDTH, SCREEN_Z);

        // track max
        if (I > maxI) maxI = I;
    }
    return maxI;
}

// ----------------------------------------------------------
// CURAND setup + Photon generation
// ----------------------------------------------------------
/*
  We use CURAND states for pseudo-random number generation on the GPU. 
  Then, we do a "rejection sampling" approach in generateDoubleSlitPhotons().
*/

// KERNEL TO SET UP CURAND STATES (one state per thread)
__global__
void setupCurandStates(curandState *states, unsigned long long seed, int n)
{
    // Compute global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // If within bounds, init the RNG state with the given seed
    if (idx < n)
        curand_init(seed, idx, 0, &states[idx]);
}

__global__
void generateDoubleSlitPhotons(
    float2*      pos,           // OUTPUT array of 2D positions for photons
    curandState* states,        // Input array of RNG states
    int          n,             // number of photons to generate
    float        wavelength,    
    float        slitDistance,
    float        slitWidth,
    float        screenZ,
    float        xRange,
    float        Imax           // maximum intensity for acceptance test
)
{
    // This kernel generates 'n' photons. Each thread tries random x in [-xRange, xRange],
    // computing intensity. If a random test < intensity/Imax, we accept that x.

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return; // out of range => do nothing

    // Copy our thread's RNG state to local
    curandState localState = states[idx];

    float x, y, I;
    bool accepted = false;

    // We'll do up to 1000 attempts to find an x that passes acceptance
    for (int attempts = 0; attempts < 1000 && !accepted; attempts++) {
        // pick a random x in the range
        x = -xRange + 2.0f*xRange*curand_uniform(&localState);

        // compute the double-slit intensity at that x
        I = doubleSlitIntensity(x, wavelength, slitDistance, slitWidth, screenZ);

        // pick a uniform random test [0, 1]
        float testVal = curand_uniform(&localState);

        // if testVal < (I / Imax), we accept
        if (testVal < (I / Imax)) {
            accepted = true;
        }
    }

    // We'll assign a small random y offset so all photons aren't exactly on a line
    y = -0.01f + 0.02f * curand_uniform(&localState);

    // Write final (x,y) back to global memory
    pos[idx] = make_float2(x, y);

    // Store updated RNG state
    states[idx] = localState;
}

// ----------------------------------------------------------
// GLSL Shaders
// ----------------------------------------------------------
/*
  We have two sets of GLSL shaders:

  1) A point-shader (vertexShaderSource + fragmentShaderSource) to color each
     photon according to the double-slit formula (visual only).
  2) A quad-shader (quadVertexShaderSource + quadFragmentShaderSource) to draw
     a fullscreen textured quad that displays our accumulation texture.
*/

// Vertex shader for points. Simple pass-through
const char* vertexShaderSource = R"(
#version 120
attribute vec2 vertexPosition;  // Incoming 2D position for each "photon" vertex
varying vec2 fragPos;           // We pass this to the fragment shader

void main() {
    // Pass the XY position to the fragment shader
    fragPos = vertexPosition;

    // Standard transform (uses fixed-function ModelView/Projection in old style)
    gl_Position = gl_ModelViewProjectionMatrix * vec4(vertexPosition, 0.0, 1.0);
}
)";

// Fragment shader for points. Computes color based on double-slit intensity & wavelength
const char* fragmentShaderSource = R"(
#version 120
varying vec2 fragPos;           // XY from the vertex shader

uniform float wavelength;       // The light wavelength (500 nm, etc.)
uniform float slitDistance;     // Distance between slits
uniform float slitWidth;        // Width of each slit
uniform float screenZ;          // Distance from slit to screen
uniform float Imax;             // Normalizing maximum intensity
uniform float intensityBoost;   // User-controlled brightness multiplier

// Inline function to do sinc^2
float sinc2(float x) {
    if (abs(x) < 1e-7) return 1.0;
    float s = sin(x)/x;
    return s*s;
}

// Evaluate double-slit intensity
float doubleSlitIntensity(float x) {
    // alpha -> from slitDistance
    float alpha = 3.14159265359 * slitDistance * x / (wavelength * screenZ);

    // beta -> from slitWidth
    float beta  = 3.14159265359 * slitWidth    * x / (wavelength * screenZ);

    return cos(alpha)*cos(alpha) * sinc2(beta);
}

// Approximate mapping from wavelength (in m) to a visible RGB color
vec3 wavelengthToRGB(float lambda) {
    float nm = lambda * 1e9; // convert to nm
    float R, G, B;

    // Basic piecewise approximation of color
    if(nm >= 380.0 && nm < 440.0) {
        R = -(nm - 440.0) / (440.0 - 380.0);
        G = 0.0;
        B = 1.0;
    } else if(nm >= 440.0 && nm < 490.0) {
        R = 0.0;
        G = (nm - 440.0) / (490.0 - 440.0);
        B = 1.0;
    } else if(nm >= 490.0 && nm < 510.0) {
        R = 0.0;
        G = 1.0;
        B = -(nm - 510.0)/(510.0 - 490.0);
    } else if(nm >= 510.0 && nm < 580.0) {
        R = (nm - 510.0)/(580.0 - 510.0);
        G = 1.0;
        B = 0.0;
    } else if(nm >= 580.0 && nm < 645.0) {
        R = 1.0;
        G = -(nm - 645.0)/(645.0 - 580.0);
        B = 0.0;
    } else if(nm >= 645.0 && nm <= 780.0) {
        R = 1.0;
        G = 0.0;
        B = 0.0;
    } else {
        R = 0.0; 
        G = 0.0; 
        B = 0.0;
    }

    // Factor to dim the color at edges of visible range
    float factor;
    if(nm >= 380.0 && nm < 420.0)
        factor = 0.3 + 0.7*(nm - 380.0)/(420.0 - 380.0);
    else if(nm >= 420.0 && nm <= 700.0)
        factor = 1.0;
    else if(nm > 700.0 && nm <= 780.0)
        factor = 0.3 + 0.7*(780.0 - nm)/(780.0 - 700.0);
    else
        factor = 0.0;

    return vec3(R, G, B)*factor;
}

void main() {
    // Local intensity from double-slit at our frag.x
    float localI = doubleSlitIntensity(fragPos.x) / Imax;

    // Scale by user intensityBoost, clamp to [0, 1]
    float val = clamp(localI * intensityBoost, 0.0, 1.0);

    // Convert the wavelength to an approximate RGB color
    vec3 baseColor = wavelengthToRGB(wavelength);

    // Output final color
    gl_FragColor = vec4(baseColor * val, 1.0);
}
)";

// Fullscreen quad vertex shader
const char* quadVertexShaderSource = R"(
#version 120
attribute vec2 pos;  // positions in [-1, +1]
varying vec2 uv;     // pass to fragment as texture coordinates

void main() {
    // map from [-1,1] to [0,1] for UV
    uv = (pos * 0.5) + 0.5;

    // Standard clip space position
    gl_Position = vec4(pos, 0.0, 1.0);
}
)";

// Fullscreen quad fragment shader
const char* quadFragmentShaderSource = R"(
#version 120
uniform sampler2D accumTex;  // we will bind our accumulation texture here
varying vec2 uv;             // texture coords from the vertex shader

void main(){
    // sample from accumTex at uv
    vec4 color = texture2D(accumTex, uv);
    gl_FragColor = color;
}
)";

// ----------------------------------------------------------
// Shader helpers
// ----------------------------------------------------------
/*
  These helper functions compile, link, and create a GLSL program object 
  from the given vertex/fragment shader source strings.
*/

// Compiles a single GLSL shader (vertex or fragment)
GLuint compileShader(GLenum type, const char* source)
{
    // Create the shader object
    GLuint shader = glCreateShader(type);

    // Set the source code
    glShaderSource(shader, 1, &source, nullptr);

    // Compile
    glCompileShader(shader);

    // Check for compile errors
    GLint compiled;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
        // If error, get length, then read the info log
        GLint len;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
        char* log = new char[len];
        glGetShaderInfoLog(shader, len, &len, log);
        std::cerr << "Shader compile error:\n" << log << std::endl;
        delete[] log;
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

// Links a GLSL program from a vertex shader and fragment shader source
GLuint createShaderProgram(const char* vsSource, const char* fsSource, const char* attrib0Name)
{
    // First compile each shader
    GLuint vs = compileShader(GL_VERTEX_SHADER, vsSource);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsSource);
    if (!vs || !fs) return 0;

    // Create the program
    GLuint program = glCreateProgram();

    // Attach the two compiled shaders
    glAttachShader(program, vs);
    glAttachShader(program, fs);

    // Bind attribute location (so we can specify which index = vertex position)
    glBindAttribLocation(program, 0, attrib0Name);

    // Link the program
    glLinkProgram(program);

    // Check for linking errors
    GLint linked;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (!linked) {
        GLint len;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &len);
        char* log = new char[len];
        glGetProgramInfoLog(program, len, &len, log);
        std::cerr << "Shader link error:\n" << log << std::endl;
        delete[] log;
        glDeleteProgram(program);
        return 0;
    }

    // Once linked, we can delete the individual shaders
    glDeleteShader(vs);
    glDeleteShader(fs);

    return program;
}

// ----------------------------------------------------------
// GLFW callbacks
// ----------------------------------------------------------
/*
  Input callbacks for:
  - mouse wheel => zoom
  - window resize => update viewport
  - mouse press => begin dragging
  - mouse move => if dragging, update pan
*/

// Zoom in/out on mouse wheel
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    float zoomFactor = 1.1f;
    if (yoffset > 0) zoom *= zoomFactor;
    else if (yoffset < 0) zoom /= zoomFactor;
}

// Called when window is resized
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    windowWidth = width;
    windowHeight = height;
    glViewport(0, 0, width, height);
}

// Called when mouse button is pressed or released
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            mouseDragging = true;
            glfwGetCursorPos(window, &lastMouseX, &lastMouseY);
        }
        else if (action == GLFW_RELEASE) {
            mouseDragging = false;
        }
    }
}

// Called when the mouse moves
void cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (mouseDragging) {
        // compute how far the mouse moved
        double dx = xpos - lastMouseX;
        double dy = ypos - lastMouseY;

        // store new last positions
        lastMouseX = xpos;
        lastMouseY = ypos;

        // Convert mouse movement in screen space to "world" space 
        // based on the current zoom
        float worldWidth  = (XRANGE*1.1f*2.0f)/zoom;
        float worldHeight = (0.05f*2.0f)/zoom;

        // Adjust panX, panY accordingly (note dy is reversed in many coordinate systems)
        panX -= dx*(worldWidth / windowWidth);
        panY += dy*(worldHeight / windowHeight);
    }
}

// ----------------------------------------------------------
// Text Overlay
// ----------------------------------------------------------
/*
  We use the stb_easy_font library to draw simple debug text in the top-left corner.
*/

void renderTextOverlay()
{
    // Set up 2D orthographic projection for the text
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, windowWidth, windowHeight, 0, -1, 1);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    // Prepare the info string
    char info[512];
    sprintf(info,
            "Controls:\n"
            " Pan: Arrow keys / Mouse drag => resets accumulation\n"
            " Zoom: Mouse wheel => resets accumulation\n"
            " I/K = intensity boost +/-\n"
            " C = clear accumulation\n"
            " ESC = Quit\n"
            "Params:\n"
            " Zoom: %.2f  IntBoost: %.2f\n"
            " lambda=%.3g m, dist=%.3g m, width=%.3g m, screenZ=%.3g m\n"
            " panX=%.4f, panY=%.4f",
            zoom, intensityBoost, LAMBDA, SLIT_DIST, SLIT_WIDTH, SCREEN_Z, panX, panY);

    // stb_easy_font prints text to a list of quads
    char buffer[99999];
    int num_quads = stb_easy_font_print(10, 10, info, NULL, buffer, sizeof(buffer));

    // Render the quads
    glColor3f(1,1,1);                  // white text
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(2, GL_FLOAT, 16, buffer);
    glDrawArrays(GL_QUADS, 0, num_quads*4);
    glDisableClientState(GL_VERTEX_ARRAY);

    // Restore matrices
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

// ----------------------------------------------------------
// Main
// ----------------------------------------------------------
int main()
{
    // Initialize GLFW library
    if (!glfwInit()) {
        std::cerr << "Failed to init GLFW\n";
        return -1;
    }

    // Create the GLFW window
    GLFWwindow* window = glfwCreateWindow(800, 600, "Double-Slit (Reset on Pan/Zoom)", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create window\n";
        glfwTerminate();
        return -1;
    }
    // Make this window the current OpenGL context
    glfwMakeContextCurrent(window);

    // Set various callbacks for input
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);

    // Initialize GLEW (for modern OpenGL function loading)
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to init GLEW\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    // Query the actual framebuffer size
    glfwGetFramebufferSize(window, &windowWidth, &windowHeight);
    glViewport(0, 0, windowWidth, windowHeight);

    // --------------------------------------------------
    // Create VBO + register with CUDA
    // --------------------------------------------------
    // We'll store the photon positions in an OpenGL VBO (vertex buffer).
    // Then, we'll map it in CUDA to write into it directly from the GPU kernel.

    GLuint vbo;
    glGenBuffers(1, &vbo);                 // create 1 buffer handle
    glBindBuffer(GL_ARRAY_BUFFER, vbo);    // bind it
    size_t bufferSize = NUM_POINTS*2*sizeof(float); 
    // allocate memory on the GPU side (NULL data for now)
    glBufferData(GL_ARRAY_BUFFER, bufferSize, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Register VBO with CUDA so we can map it
    cudaGraphicsResource* cudaVboResource;
    cudaGraphicsGLRegisterBuffer(&cudaVboResource, vbo, cudaGraphicsMapFlagsWriteDiscard);

    // Allocate device memory for CURAND states (one per photon)
    curandState* d_rngStates = nullptr;
    cudaMalloc((void**)&d_rngStates, NUM_POINTS*sizeof(curandState));

    // Compute grid dimension for CUDA kernel launch
    int grid = (NUM_POINTS + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Initialize RNG states (seed=1234)
    setupCurandStates<<<grid, BLOCK_SIZE>>>(d_rngStates, 1234ULL, NUM_POINTS);
    cudaDeviceSynchronize();

    // --------------------------------------------------
    // Create Shaders
    // --------------------------------------------------
    // We compile/link the two sets of shaders: pointShader and quadShader.

    GLuint pointShaderProgram = createShaderProgram(vertexShaderSource, fragmentShaderSource, "vertexPosition");
    GLuint quadShaderProgram  = createShaderProgram(quadVertexShaderSource, quadFragmentShaderSource, "pos");
    if (!pointShaderProgram || !quadShaderProgram) {
        std::cerr << "Shader program creation failed\n";
        return -1;
    }

    // --------------------------------------------------
    // Create accumulation FBO/Texture
    // --------------------------------------------------
    // We want to accumulate many frames of photons over time. We'll create an 
    // offscreen framebuffer (FBO) with a floating-point texture so we can 
    // keep adding intensities.

    GLuint accumFBO;
    glGenFramebuffers(1, &accumFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, accumFBO);

    GLuint accumTex;
    glGenTextures(1, &accumTex);
    glBindTexture(GL_TEXTURE_2D, accumTex);

    // allocate a float RGBA texture the size of our window
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, windowWidth, windowHeight, 0,
                 GL_RGBA, GL_FLOAT, nullptr);

    // nearest neighbor filtering
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // attach the texture to the FBO
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, accumTex, 0);

    // check completeness
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cerr << "accumFBO incomplete!\n";

    // Clear the accumulation texture to black
    glClearColor(0,0,0,0);
    glClear(GL_COLOR_BUFFER_BIT);

    // unbind the FBO
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // --------------------------------------------------
    // Fullscreen quad to display accumTex
    // --------------------------------------------------
    // We'll draw a simple quad from [-1,1]^2 to fill the screen. We'll sample accumTex.

    GLuint quadVAO, quadVBO;
    glGenVertexArrays(1, &quadVAO);
    glBindVertexArray(quadVAO);

    static const GLfloat fsQuadVerts[] = {
         -1.f, -1.f,
          1.f, -1.f,
         -1.f,  1.f,
          1.f,  1.f
    };

    glGenBuffers(1, &quadVBO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(fsQuadVerts), fsQuadVerts, GL_STATIC_DRAW);

    // We say that attribute 0 is the 2D position
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);

    // unbind
    glBindVertexArray(0);

    // Initialize our "last known" camera values
    lastPanX = panX;
    lastPanY = panY;
    lastZoom = zoom;

    // We'll store the maximum intensity in a float
    float Imax = 1.0f;

    // MAIN LOOP
    while (!glfwWindowShouldClose(window))
    {
        // Process events (keyboard, mouse, etc.)
        glfwPollEvents();

        // ARROW KEYS => Pan
        float panSpeed = 0.0005f / zoom; // scale with zoom
        if (glfwGetKey(window, GLFW_KEY_LEFT)  == GLFW_PRESS) panX -= panSpeed;
        if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) panX += panSpeed;
        if (glfwGetKey(window, GLFW_KEY_UP)    == GLFW_PRESS) panY += panSpeed;
        if (glfwGetKey(window, GLFW_KEY_DOWN)  == GLFW_PRESS) panY -= panSpeed;

        // I/K => adjust intensityBoost
        if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS)
            intensityBoost += 0.01f;
        if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS)
            intensityBoost = (intensityBoost > 0.02f) ? (intensityBoost - 0.01f) : 0.01f;

        // C => clear accumulation
        if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS) {
            glBindFramebuffer(GL_FRAMEBUFFER, accumFBO);
            glClear(GL_COLOR_BUFFER_BIT);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
        }

        // Check if pan changed => reset accumulation
        float panDeltaX = fabsf(panX - lastPanX);
        float panDeltaY = fabsf(panY - lastPanY);
        if (panDeltaX > 1e-7f || panDeltaY > 1e-7f) {
            // Clear accum
            glBindFramebuffer(GL_FRAMEBUFFER, accumFBO);
            glClear(GL_COLOR_BUFFER_BIT);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);

            // Update trackers
            lastPanX = panX;
            lastPanY = panY;
        }

        // Check if zoom changed => reset accumulation
        if (fabsf(zoom - lastZoom) > 1e-7f) {
            glBindFramebuffer(GL_FRAMEBUFFER, accumFBO);
            glClear(GL_COLOR_BUFFER_BIT);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);

            lastZoom = zoom;
        }

        // 1) Compute dynamic Imax each frame 
        Imax = computeMaxIntensity(2000);

        // 2) Generate new photons via CUDA
        // Map the VBO so we can write into it
        cudaGraphicsMapResources(1, &cudaVboResource, 0);
        void* dPtr = nullptr;
        size_t dSize = 0;
        cudaGraphicsResourceGetMappedPointer(&dPtr, &dSize, cudaVboResource);

        // Launch kernel to fill 'dPtr' with photon positions
        generateDoubleSlitPhotons<<<grid, BLOCK_SIZE>>>(
            (float2*)dPtr, d_rngStates, NUM_POINTS,
            LAMBDA, SLIT_DIST, SLIT_WIDTH, SCREEN_Z, XRANGE,
            Imax
        );
        cudaDeviceSynchronize();

        // Unmap so OpenGL can use the VBO
        cudaGraphicsUnmapResources(1, &cudaVboResource, 0);

        // PASS 1: Render these photons to accumFBO
        glBindFramebuffer(GL_FRAMEBUFFER, accumFBO);
        glViewport(0, 0, windowWidth, windowHeight);

        // Enable additive blending (GL_ONE, GL_ONE => sum of colors)
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE);

        // Set up an orthographic projection based on current pan/zoom
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();

        float left   = panX - (XRANGE*1.1f)/zoom;
        float right  = panX + (XRANGE*1.1f)/zoom;
        float bottom = panY - (0.05f)/zoom;
        float top    = panY + (0.05f)/zoom;
        glOrtho(left, right, bottom, top, -1.0, 1.0);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        // Use the point shader program (we use the intensity formula for coloring)
        glUseProgram(pointShaderProgram);

        // Pass in uniform parameters
        glUniform1f(glGetUniformLocation(pointShaderProgram, "wavelength"),    LAMBDA);
        glUniform1f(glGetUniformLocation(pointShaderProgram, "slitDistance"), SLIT_DIST);
        glUniform1f(glGetUniformLocation(pointShaderProgram, "slitWidth"),    SLIT_WIDTH);
        glUniform1f(glGetUniformLocation(pointShaderProgram, "screenZ"),      SCREEN_Z);
        glUniform1f(glGetUniformLocation(pointShaderProgram, "Imax"),         Imax);
        glUniform1f(glGetUniformLocation(pointShaderProgram, "intensityBoost"), intensityBoost);

        // Bind the photon position VBO
        glBindBuffer(GL_ARRAY_BUFFER, vbo);

        // Enable vertex attrib array for position (location=0)
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

        // Draw the points
        glDrawArrays(GL_POINTS, 0, NUM_POINTS);

        // Cleanup
        glDisableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glUseProgram(0);

        // Disable blending
        glDisable(GL_BLEND);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        // PASS 2: draw accumTex => screen
        glViewport(0, 0, windowWidth, windowHeight);
        glClear(GL_COLOR_BUFFER_BIT);

        // Use the quad shader
        glUseProgram(quadShaderProgram);

        // Bind the accumulation texture to texture unit 0
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, accumTex);

        // Set the sampler uniform to 0
        GLint loc = glGetUniformLocation(quadShaderProgram, "accumTex");
        glUniform1i(loc, 0);

        // Bind the VAO with our fullscreen quad
        glBindVertexArray(quadVAO);

        // Draw the quad as a TRIANGLE_STRIP of 4 verts
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        // Cleanup
        glBindVertexArray(0);
        glUseProgram(0);

        // Overlay text
        renderTextOverlay();

        // Swap the front/back buffers
        glfwSwapBuffers(window);

        // ESC => exit
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, 1);
    }

    // Cleanup resources before exit

    // Free CUDA memory
    cudaFree(d_rngStates);

    // Unregister and delete the VBO
    cudaGraphicsUnregisterResource(cudaVboResource);
    glDeleteBuffers(1, &vbo);

    // Delete shaders
    glDeleteProgram(pointShaderProgram);
    glDeleteProgram(quadShaderProgram);

    // Delete the accumulation texture and FBO
    glDeleteTextures(1, &accumTex);
    glDeleteFramebuffers(1, &accumFBO);

    // Delete the quad VAO and VBO
    glDeleteVertexArrays(1, &quadVAO);
    glDeleteBuffers(1, &quadVBO);

    // Destroy the window and terminate GLFW
    glfwDestroyWindow(window);
    glfwTerminate();

    // Done!
    return 0;
}
