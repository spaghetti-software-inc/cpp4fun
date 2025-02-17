#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class Camera {
public:
    Camera() = default;
    ~Camera() = default;

    // Basic get/set
    float yaw()   const { return yaw_; }
    float pitch() const { return pitch_; }
    float dist()  const { return dist_; }

    // Access a view matrix
    glm::mat4 getViewMatrix() const {
        // translate back by dist, then rotate by pitch around X, yaw around Y
        glm::mat4 view = glm::translate(glm::mat4(1.f), glm::vec3(0.f, 0.f, -dist_));
        view           = glm::rotate(view, glm::radians(pitch_), glm::vec3(1, 0, 0));
        view           = glm::rotate(view, glm::radians(yaw_),   glm::vec3(0, 1, 0));
        return view;
    }

    // Called when mouse button is pressed/released
    void onMouseButton(int button, int action, double x, double y) {
        if(button == 0) { // Left mouse
            leftButtonDown_ = (action == 1); // GLFW_PRESS=1, GLFW_RELEASE=0
            lastX_ = x;
            lastY_ = y;
        } else if(button == 1) { // Right mouse
            rightButtonDown_ = (action == 1);
            lastX_ = x;
            lastY_ = y;
        }
    }

    // Called when mouse moves
    void onCursorPos(double x, double y) {
        float dx = float(x - lastX_);
        float dy = float(y - lastY_);
        lastX_ = x;
        lastY_ = y;

        // If left button is down => rotate
        if(leftButtonDown_) {
            yaw_   += dx * 0.3f;
            pitch_ += dy * 0.3f;
        }
        // If right button is down => dolly
        if(rightButtonDown_) {
            dist_ += dy * 0.01f;
            if(dist_ < 0.1f) dist_ = 0.1f;
        }
    }

    // Called when mouse wheel scrolls
    void onScroll(double xoff, double yoff) {
        dist_ -= float(yoff)*0.2f;
        if(dist_ < 0.1f) dist_ = 0.1f;
    }

private:
    float yaw_   = 0.f;
    float pitch_ = 0.f;
    float dist_  = 4.f;

    bool   leftButtonDown_  = false;
    bool   rightButtonDown_ = false;
    double lastX_           = 0.0;
    double lastY_           = 0.0;
};
