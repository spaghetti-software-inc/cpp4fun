// main.cpp
#include <iostream>
#include <stdexcept>

#include <GLFW/glfw3.h>
// If you are using GLAD, include glad before glfw3:
// #include <glad/glad.h>

// #include "imgui.h"
// #include "imgui_impl_glfw.h"
// #include "imgui_impl_opengl3.h"

// Forward declarations
static void glfw_error_callback(int error, const char* description)
{
    std::cerr << "GLFW Error " << error << ": " << description << std::endl;
}

int main()
{
    // 1. Set up error callback (optional but useful)
    glfwSetErrorCallback(glfw_error_callback);

    // 2. Initialize GLFW
    if (!glfwInit())
    {
        throw std::runtime_error("Failed to initialize GLFW");
    }

    // 3. Configure GLFW for OpenGL 3.3+ (Core Profile)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    // If you want a more modern setup, you can do:
    // glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    // glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    // 4. Create a windowed mode window and its OpenGL context
    GLFWwindow* window = glfwCreateWindow(1280, 720, "Dear ImGui + GLFW Demo", nullptr, nullptr);
    if (!window)
    {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }
    glfwMakeContextCurrent(window);

    // 5. Load OpenGL functions using GLAD (if using glad)
    //    If you’re using another loader like GLEW, do that init here.
    /*
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        glfwTerminate();
        throw std::runtime_error("Failed to initialize GLAD");
    }
    */

    // Optionally enable vsync
    glfwSwapInterval(1);

    // // 6. Initialize Dear ImGui
    // IMGUI_CHECKVERSION();
    // ImGui::CreateContext();
    // // Optionally configure Dear ImGui style
    // ImGui::StyleColorsDark();

    // // 7. Initialize Dear ImGui GLFW & OpenGL backends
    // //    For OpenGL 3.x you can use the string "#version 130" or higher as the GLSL version.
    // ImGui_ImplGlfw_InitForOpenGL(window, true);
    // ImGui_ImplOpenGL3_Init("#version 130");

    // // 8. Main loop
    // while (!glfwWindowShouldClose(window))
    // {
    //     // Poll and handle events
    //     glfwPollEvents();

    //     // Start the Dear ImGui frame
    //     ImGui_ImplOpenGL3_NewFrame();
    //     ImGui_ImplGlfw_NewFrame();
    //     ImGui::NewFrame();

    //     // Show the Dear ImGui Demo window (optional)
    //     ImGui::ShowDemoWindow();

    //     // Rendering
    //     ImGui::Render();
    //     // Clear the background
    //     glViewport(0, 0, 1280, 720);
    //     glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
    //     glClear(GL_COLOR_BUFFER_BIT);

    //     // Render Dear ImGui's draw data
    //     ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    //     // Swap front and back buffers
    //     glfwSwapBuffers(window);
    // }

    // // 9. Cleanup
    // ImGui_ImplOpenGL3_Shutdown();
    // ImGui_ImplGlfw_Shutdown();
    // ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
