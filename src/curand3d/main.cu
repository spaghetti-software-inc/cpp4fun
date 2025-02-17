#include <iostream>
#include <vector>

// GLEW
#include <GL/glew.h>

// GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// Local headers
#include "GlfwWindow.h"
#include "Shader.h"
#include "Sphere.h"

// Vertex + fragment shader
static const char* vsSource = R"(
#version 330 core
layout (location = 0) in vec4 inPosRad; // (x, y, z, radius)

uniform mat4 u_mvp;

out vec4 vColor; // pass color to fragment shader

void main()
{
    gl_PointSize = 5.0;
    gl_Position  = u_mvp * vec4(inPosRad.xyz, 1.0);

    float r = inPosRad.w;
    float t = clamp(r / 2.0, 0.0, 1.0);
    vColor = vec4(1.0 - t, 0.0, t, 1.0);
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

int main()
{
    try {
        GlfwWindow window(800, 600, "Multiple Random Spheres");
        // You must call glewInit() *after* creating an OpenGL context
        if (glewInit() != GLEW_OK) {
            throw std::runtime_error("Failed to initialize GLEW");
        }

        // Create the shader
        Shader shader(vsSource, fsSource);

        // Create multiple spheres with different random seeds
        std::vector<Sphere> spheres;
        spheres.reserve(5);
        for (int i = 0; i < 5; ++i) {
            unsigned long long seed = 12345ULL + i * 1000ULL;
            spheres.emplace_back(50'000, seed);
        }

        // Basic camera parameters
        float yaw   = 0.f;
        float pitch = 0.f;
        float dist  = 4.f;

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_PROGRAM_POINT_SIZE);

        while (!window.shouldClose()) {
            window.pollEvents();

            // Clear
            glClearColor(0.1f, 0.15f, 0.2f, 1.f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            // Set up your MVP
            int width, height;
            glfwGetFramebufferSize(window.get(), &width, &height);
            float aspect = float(width) / float(height);

            glm::mat4 proj  = glm::perspective(glm::radians(45.f), aspect, 0.1f, 100.f);
            glm::mat4 view  = glm::translate(glm::mat4(1.f), glm::vec3(0, 0, -dist));
            view            = glm::rotate(view, glm::radians(pitch), glm::vec3(1, 0, 0));
            view            = glm::rotate(view, glm::radians(yaw),   glm::vec3(0, 1, 0));
            glm::mat4 model = glm::mat4(1.f); // or each sphere might have its own transform

            glm::mat4 mvp = proj * view * model;

            // Use the shader and set uniform
            shader.use();
            GLint mvpLoc = glGetUniformLocation(shader.id(), "u_mvp");
            glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, glm::value_ptr(mvp));

            // Draw each sphere
            for (auto& sphere : spheres) {
                sphere.draw();
            }

            window.swapBuffers();
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
