# Fractal Demo User Manual

## Overview

The Fractal Demo is an interactive visualization tool that renders various fractals in real time using CUDA and OpenGL. The demo supports multiple fractal types, including the Mandelbrot set, Julia sets, Burning Ship, Tricorn, Celtic, and Newton fractals. It features real-time user controls for zooming, panning, and adjusting fractal parameters, with an on-screen text overlay displaying the current state.

## Requirements

- **Hardware:** CUDA-capable GPU.
- **Software:**  
  - OpenGL 3.3 (or later)  
  - CUDA Toolkit  
  - Libraries: GLFW, GLEW, GLM, stb_easy_font (for text overlay)
- **Compiler:** C++ compiler supporting C++11 (or later)

## Installation & Compilation

1. **Download the Source Code:**  
   Clone or download the repository containing the demo.

2. **Dependencies:**  
   Ensure that GLFW, GLEW, GLM, and CUDA are installed on your system. Place `stb_easy_font.h` in your include path.

3. **Compile the Demo:**  
   For example, using `g++`:
   ```bash
   g++ fractal_demo.cpp -o fractal_demo -lGL -lglfw -lGLEW -lcuda -lcudart
   ```
   Adjust the compiler flags and paths according to your system configuration.

## Running the Demo

- Launch the executable:
  ```bash
  ./fractal_demo
  ```
- An 800x600 window will open displaying the currently selected fractal.

## User Controls

### Keyboard Shortcuts

- **Switch Fractal Type:**  
  Press number keys `1` through `9`:
  - **1:** Mandelbrot
  - **2:** Julia
  - **3:** Burning Ship
  - **4:** Tricorn
  - **5:** Celtic
  - **6:** Newton

- **Reset View:**  
  Press `R` to reset the camera to its default settings (center at `-0.5, 0.0` and scale `3.0`).

- **Pan the View:**  
  Use the **Arrow Keys** to shift the view:
  - `Left/Right`: Pan horizontally.
  - `Up/Down`: Pan vertically.

- **Adjust Julia Parameters (only for Julia fractal):**  
  - Press `[` or `]` to tweak the Julia set’s **Cx** (real part).  
  - Hold `SHIFT` and press `[` or `]` to adjust the Julia set’s **Cy** (imaginary part).

- **Adjust Maximum Iteration Count:**  
  - Press `+` (or `=`/`KP_ADD`) to increase the max iterations.  
  - Press `-` (or `KP_SUBTRACT`) to decrease the max iterations.

- **Exit Application:**  
  Press `ESC` to close the demo.

### Mouse Controls

- **Right Mouse Button Drag:**  
  Click and hold the right mouse button, then drag to pan the fractal view.

- **Scroll Wheel:**  
  Scroll to zoom in or out of the fractal.

## On-Screen Information

A text overlay is rendered in the top-left corner of the window displaying:
- **Fractal:** The name of the current fractal.
- **Zoom:** The current scale (zoom level) of the view.
- **Max Iter:** The current maximum iteration count used for rendering.
- **Mouse Coordinates:** The coordinates in the complex plane corresponding to the current mouse position.

## Troubleshooting

- **CUDA or OpenGL Errors:**  
  Ensure that your GPU drivers and CUDA toolkit are up-to-date and that your system meets the minimum hardware requirements.

- **Compilation Issues:**  
  Verify that all required libraries are installed and properly linked. Check that `stb_easy_font.h` is accessible in your include path.

## License & Credits

This demo utilizes:
- [CUDA](https://developer.nvidia.com/cuda-zone)
- [OpenGL](https://www.opengl.org/)
- [GLFW](https://www.glfw.org/)
- [GLEW](http://glew.sourceforge.net/)
- [GLM](https://github.com/g-truc/glm)
- [stb_easy_font](https://github.com/nothings/stb)

Please refer to each library's individual license for more details.


