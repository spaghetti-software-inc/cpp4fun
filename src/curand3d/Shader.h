#pragma once

#include <GL/glew.h>
#include <stdexcept>
#include <string>
#include <iostream>

class Shader {
public:
    Shader(const char* vertexSrc, const char* fragmentSrc) {
        program_ = createShaderProgram(vertexSrc, fragmentSrc);
        if (!program_) {
            throw std::runtime_error("Failed to create shader program");
        }
    }

    ~Shader() {
        if (program_) {
            glDeleteProgram(program_);
        }
    }

    Shader(const Shader&) = delete;
    Shader& operator=(const Shader&) = delete;

    GLuint id() const { return program_; }

    void use() const {
        glUseProgram(program_);
    }

private:
    GLuint program_ = 0;

    static GLuint compileShader(GLenum type, const char* src) {
        GLuint s = glCreateShader(type);
        glShaderSource(s, 1, &src, nullptr);
        glCompileShader(s);
        GLint success;
        glGetShaderiv(s, GL_COMPILE_STATUS, &success);
        if (!success) {
            char log[512];
            glGetShaderInfoLog(s, 512, nullptr, log);
            std::cerr << "Shader compile error:\n" << log << std::endl;
            glDeleteShader(s);
            return 0;
        }
        return s;
    }

    static GLuint createShaderProgram(const char* vsSrc, const char* fsSrc) {
        GLuint vs = compileShader(GL_VERTEX_SHADER, vsSrc);
        if(!vs) return 0;
        GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsSrc);
        if(!fs) {
            glDeleteShader(vs);
            return 0;
        }

        GLuint prog = glCreateProgram();
        glAttachShader(prog, vs);
        glAttachShader(prog, fs);
        glLinkProgram(prog);

        glDeleteShader(vs);
        glDeleteShader(fs);

        GLint success;
        glGetProgramiv(prog, GL_LINK_STATUS, &success);
        if(!success) {
            char log[512];
            glGetProgramInfoLog(prog, 512, nullptr, log);
            std::cerr << "Program link error:\n" << log << std::endl;
            glDeleteProgram(prog);
            return 0;
        }
        return prog;
    }
};
