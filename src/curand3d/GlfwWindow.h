#pragma once

#include <GLFW/glfw3.h>
#include <stdexcept>
#include <functional>

class GlfwWindow {
public:
    GlfwWindow(int width, int height, const char* title) {
        if (!glfwInit()) {
            throw std::runtime_error("Failed to initialize GLFW");
        }

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        window_ = glfwCreateWindow(width, height, title, nullptr, nullptr);
        if (!window_) {
            glfwTerminate();
            throw std::runtime_error("Failed to create window");
        }
        glfwMakeContextCurrent(window_);

        // Set user pointer to this, for static callbacks
        glfwSetWindowUserPointer(window_, this);
    }

    ~GlfwWindow() {
        if (window_) {
            glfwDestroyWindow(window_);
        }
        glfwTerminate();
    }

    GlfwWindow(const GlfwWindow&) = delete;
    GlfwWindow& operator=(const GlfwWindow&) = delete;
    GlfwWindow(GlfwWindow&&) = delete;
    GlfwWindow& operator=(GlfwWindow&&) = delete;

    GLFWwindow* get() const { return window_; }

    bool shouldClose() const { return glfwWindowShouldClose(window_); }
    void pollEvents() const  { glfwPollEvents(); }
    void swapBuffers() const { glfwSwapBuffers(window_); }

private:
    GLFWwindow* window_ = nullptr;
};
