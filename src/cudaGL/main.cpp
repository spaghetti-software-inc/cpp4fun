
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

#include "Renderer.h"



#include <iostream>
#include <memory>
#include <vector>


#include "Renderer.h"


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
}

Renderer::~Renderer() {
    // unregister this buffer object with CUDA
    checkCudaErrors(cudaGraphicsUnregisterResource(_cuda_vbo_resource));

    glBindBuffer(1, _vbo);
    glDeleteBuffers(1, &_vbo);

    _vbo = 0;
}

void Renderer::render() {
  glClearColor(0.529f, 0.808f, 0.922f, 1.0f); // Light sky blue
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}




std::unique_ptr<Renderer> renderer = nullptr;



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
  
    renderer = std::make_unique<Renderer>();
    glutMainLoop();
    return 0;
}