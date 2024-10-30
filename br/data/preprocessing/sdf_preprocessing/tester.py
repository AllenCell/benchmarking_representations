# import pygame, os
# from pygame.locals import *
# from OpenGL.GL import *

# # Initialize Pygame and OpenGL
# pygame.init()
# width, height = 800, 600
# pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
# print(pygame.display)
# # Main loop
# while True:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             pygame.quit()
#             exit()
    
#     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
#     pygame.display.flip()

# import sys, pygame, os
# # os.environ["PYOPENGL_PLATFORM"] = "egl"
# pygame.init()

# size = width, height = 320, 240
# speed = [2, 2]
# black = 0, 0, 0

# screen = pygame.display.set_mode(size)

# ball = pygame.image.load("intro_ball.gif")
# ballrect = ball.get_rect()

# while True:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT: sys.exit()

#     ballrect = ballrect.move(speed)
#     if ballrect.left < 0 or ballrect.right > width:
#         speed[0] = -speed[0]
#     if ballrect.top < 0 or ballrect.bottom > height:
#         speed[1] = -speed[1]

#     screen.fill(black)
#     screen.blit(ball, ballrect)
#     pygame.display.flip()


# import sys
# import pygame
# from OpenGL import GL
# from OpenGL import EGL

# # Initialize Pygame
# pygame.init()

# # Setup EGL
# display = EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY)
# EGL.eglInitialize(display, None, None)

# # Choose EGL configuration
# config_attribs = [EGL.EGL_SURFACE_TYPE, EGL.EGL_WINDOW_BIT,
#                   EGL.EGL_RENDERABLE_TYPE, EGL.EGL_OPENGL_ES2_BIT,
#                   EGL.EGL_NONE]
# config = EGL.EGLConfig()
# num_configs = EGL.EGLint()
# EGL.eglChooseConfig(display, config_attribs, config, 1, num_configs)

# # Create an EGL context
# context_attribs = [EGL.EGL_CONTEXT_CLIENT_VERSION, 2, EGL.EGL_NONE]
# context = EGL.eglCreateContext(display, config, EGL.EGL_NO_CONTEXT, context_attribs)

# # Create an EGL surface
# pygame.display.set_mode((320, 240), pygame.OPENGL | pygame.DOUBLEBUF)
# surface = EGL.eglCreateWindowSurface(display, config, pygame.display.get_wm_info()["window"], None)

# # Make the context current
# EGL.eglMakeCurrent(display, surface, surface, context)

# # Your main draw loop
# while True:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             EGL.eglTerminate(display)
#             sys.exit()

#     # Drawing code goes here (like clearing the screen or rendering shapes)
#     GL.glClear(GL.GL_COLOR_BUFFER_BIT)
    
#     # Swap buffers
#     EGL.eglSwapBuffers(display, surface)

#     pygame.time.wait(10)

# # Terminate EGL
# EGL.eglTerminate(display)

import pyglet
from pyglet import gl
import numpy as np
from ctypes import c_void_p
pyglet.options["headless"] = True

# Set up a window
window = pyglet.window.Window(width=800, height=600, caption='OpenGL Triangle Example')

# Vertex Shader source
vertex_shader_source = """
#version 330 core
layout(location = 0) in vec2 position;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

# Fragment Shader source
fragment_shader_source = """
#version 330 core
out vec4 FragColor;
void main() {
    FragColor = vec4(1.0, 0.5, 0.2, 1.0); // Set color to orange
}
"""

# Compile a shader
def compile_shader(source, shader_type):
    shader = gl.glCreateShader(shader_type)
    gl.glShaderSource(shader, source)
    gl.glCompileShader(shader)

    # Check for compile errors
    success = gl.GLint()
    gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS, success)
    if not success:
        info_log = gl.glGetShaderInfoLog(shader)
        raise Exception(f"Shader compilation failed: {info_log.decode()}")
    return shader

# Set up shaders
vertex_shader = compile_shader(vertex_shader_source, gl.GL_VERTEX_SHADER)
fragment_shader = compile_shader(fragment_shader_source, gl.GL_FRAGMENT_SHADER)

shader_program = gl.glCreateProgram()
gl.glAttachShader(shader_program, vertex_shader)
gl.glAttachShader(shader_program, fragment_shader)
gl.glLinkProgram(shader_program)

# Vertex data for a triangle
vertices = np.array([
    -0.5, -0.5,  # Bottom-left
     0.5, -0.5,  # Bottom-right
     0.0,  0.5,  # Top
], dtype='f')

# Create Vertex Buffer Object (VBO)
vbo = gl.GLuint()
gl.glGenBuffers(1, vbo)
gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices.ctypes.data, gl.GL_STATIC_DRAW)

# Set up vertex attribute pointers
gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, False, 2 * vertices.itemsize, c_void_p(0))
gl.glEnableVertexAttribArray(0)

@window.event
def on_draw():
    # Clear the color buffer
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)

    # Use the shader program
    gl.glUseProgram(shader_program)

    # Draw the triangle
    gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)

@window.event
def on_key_press(symbol, modifiers):
    # Exit when the ESC key is pressed
    if symbol == pyglet.window.key.ESCAPE:
        pyglet.app.exit()

# Run the application
pyglet.app.run()