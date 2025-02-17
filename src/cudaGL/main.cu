#include <helper_gl.h>
#if defined(__APPLE__) || defined(__MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif
#if defined(WIN32)
#include <GL/wglew.h>
#endif


// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h


// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check

#include <helper_image.h>


#include <iostream>
#include <memory>
#include <vector>

#include "GLSLProgram.h"

class Renderer {
  private:
      const unsigned int _mesh_width    = 256;
      const unsigned int _mesh_height   = 256;
  
      // vbo variables
      GLuint _vbo;
      struct cudaGraphicsResource* _cuda_vbo_resource = nullptr;
      void* _d_vbo_buffer = nullptr;


      GLuint _floor_tex = 0;
      std::unique_ptr<GLSLProgram> _floor_prog = nullptr;
  
      GLuint createTexture(GLenum target, GLint internalformat, GLenum format, int w, int h, void *data);
      GLuint loadTexture(const std::string& filename);
      
  
  public:
      Renderer();
      ~Renderer();
  
      void cleanup();
  
      void render();

      
      

  };
  


Renderer::Renderer() {
  checkCudaErrors(cudaMalloc((void **)&_d_vbo_buffer, _mesh_width*_mesh_height*4*sizeof(float)));

  assert(&_vbo);

  // create buffer object
  glGenBuffers(1, &_vbo);
  glBindBuffer(GL_ARRAY_BUFFER, _vbo);

  // initialize buffer object
  unsigned int size = _mesh_width * _mesh_height * 4 * sizeof(float);
  glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // register this buffer object with CUDA
  checkCudaErrors(cudaGraphicsGLRegisterBuffer(&_cuda_vbo_resource, _vbo, cudaGraphicsMapFlagsWriteDiscard));

  // SDK_CHECK_ERROR_GL();    

  // load textures
  _floor_tex = loadTexture("data/floortile.ppm");
}

Renderer::~Renderer() {
}

void Renderer::render() {
  glClearColor(0.529f, 0.808f, 0.922f, 1.0f); // Light sky blue
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void Renderer::cleanup() {
  // unregister this buffer object with CUDA
  checkCudaErrors(cudaGraphicsUnregisterResource(_cuda_vbo_resource));

  glBindBuffer(1, _vbo);
  glDeleteBuffers(1, &_vbo);
  _vbo = 0;

  glDeleteTextures(1, &_floor_tex);
  _floor_tex = 0;
}

GLuint Renderer::createTexture(GLenum target, GLint internalformat, GLenum format, 
                     int w, int h, void *data) {
  GLuint tex;
  glGenTextures(1, &tex);
  glBindTexture(target, tex);
  glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(target, GL_GENERATE_MIPMAP_SGIS, GL_TRUE);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexImage2D(target, 0, internalformat, w, h, 0, format, GL_UNSIGNED_BYTE,
  data);
  return tex;
}

GLuint Renderer::loadTexture(const std::string& filename) {
  unsigned char *data = 0;
  unsigned int width, height;
  sdkLoadPPM4ub(filename.c_str(), &data, &width, &height);

  if (!data) {
    printf("Error opening file '%s'\n", filename.c_str());
    return 0;
  }

  printf("Loaded '%s', %d x %d pixels\n", filename.c_str(), width, height);

  return createTexture(GL_TEXTURE_2D, GL_RGBA8, GL_RGBA, width, height, data);
}

std::unique_ptr<Renderer> renderer = nullptr;

void cleanup() {
  renderer->cleanup();
}



// main rendering loop
void display() {
    renderer->render();
    glutSwapBuffers();
 
    // glClearColor(0.529f, 0.808f, 0.922f, 1.0f); // Light sky blue
    // glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // glutSwapBuffers();
    // glutReportErrors();
  }

// GLUT callback functions
void reshape(int w, int h) {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (float)w / (float)h, 0.01, 100.0);
  
    glMatrixMode(GL_MODELVIEW);
    glViewport(0, 0, w, h);
}  
  

// initialize OpenGL
void initGL(int *argc, char **argv) {
    int winWidth = 1024;
    int winHeight = 768;

    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(winWidth, winHeight);
    glutCreateWindow("CUDA Smoke Show");
  
    if (!isGLVersionSupported(2, 0)) {
      fprintf(stderr,
              "The following required OpenGL extensions "
              "missing:\n\tGL_VERSION_2_0\n\tGL_VERSION_1_5\n");
      exit(EXIT_SUCCESS);
    }
  
    if (!areGLExtensionsSupported("GL_ARB_multitexture "
                                  "GL_ARB_vertex_buffer_object "
                                  "GL_EXT_geometry_shader4")) {
      fprintf(stderr,
              "The following required OpenGL extensions "
              "missing:\n\tGL_ARB_multitexture\n\tGL_ARB_vertex_buffer_"
              "object\n\tGL_EXT_geometry_shader4.\n");
      exit(EXIT_SUCCESS);
    }
  
  #if defined(WIN32)
  
    if (wglewIsSupported("WGL_EXT_swap_control")) {
      // disable vertical sync
      wglSwapIntervalEXT(0);
    }
  
  #endif

    glEnable(GL_DEPTH_TEST);
  
    std::cout << "" << std::endl;
    std::cout << "" << "OpenGL Vendor: " << glGetString(GL_VENDOR) << std::endl;
    std::cout << "" << "OpenGL Renderer: " << glGetString(GL_RENDERER) << std::endl;
    std::cout << "" << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "" << "OpenGL Shading Language Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
    std::cout << "" << std::endl;

    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                    GL_LINEAR_MIPMAP_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 16.0f);
  

    glutReportErrors();
  }

int main(int argc, char **argv) {
#if defined(__linux__)
    setenv("DISPLAY", ":0", 0);
#endif
    

    // 1st initialize OpenGL context, so we can properly set the GL for CUDA.
    // This is needed to achieve optimal performance with OpenGL/CUDA interop.
    initGL(&argc, argv);
  

    
    // findCudaDevice(argc, (const char **)argv);
  
    //   // This is the normal code path for SmokeParticles
    //   initParticles(numParticles, true, true);
    //   initParams();
    //   initMenus();
  
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    //   glutMouseFunc(mouse);
    //   glutMotionFunc(motion);
    //   glutKeyboardFunc(key);
    //   glutKeyboardUpFunc(keyUp);
    //   glutSpecialFunc(special);
    //   glutIdleFunc(idle);
#if defined (__APPLE__) || defined(MACOSX)
        atexit(cleanup);
#else
        glutCloseFunc(cleanup);
#endif
  
    renderer = std::make_unique<Renderer>();
    glutMainLoop();
    return 0;
}