"""Real caller-owned GL framebuffer -> native Nodus tensor readback proof.

Run with NODUS_TENSOR_CORE_DLL pointing at the built nodus_tensor_core.dll.
This exercises transfer only; the protected production GL child has no draw API.
"""
import ctypes
import os
from pathlib import Path
import sys

import pygame
from OpenGL.GL import (GL_COLOR_BUFFER_BIT, GL_PACK_ALIGNMENT, GL_PACK_ROW_LENGTH,
                       GL_PACK_SKIP_PIXELS, GL_PACK_SKIP_ROWS, GL_RGBA,
                       GL_SCISSOR_TEST, GL_UNSIGNED_BYTE, glClear,
                       glClearColor, glDisable, glEnable, glGetIntegerv,
                       glPixelStorei, glReadPixels, glScissor)


class Context(ctypes.Structure):
    _fields_ = [("context_id", ctypes.c_size_t),
                ("get_proc", ctypes.c_void_p),
                ("current_context", ctypes.c_void_p),
                ("user", ctypes.c_void_p)]


def main():
    dll_path = Path(os.environ["NODUS_TENSOR_CORE_DLL"]).resolve()
    os.add_dll_directory(str(dll_path.parent))
    core = ctypes.CDLL(str(dll_path))
    gl_dll = ctypes.WinDLL("opengl32.dll")
    gl_dll.wglGetCurrentContext.restype = ctypes.c_void_p
    gl_dll.wglGetProcAddress.argtypes = [ctypes.c_char_p]
    gl_dll.wglGetProcAddress.restype = ctypes.c_void_p
    get_proc_type = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p)
    current_type = ctypes.CFUNCTYPE(ctypes.c_size_t, ctypes.c_void_p)

    @get_proc_type
    def get_proc(name, _):
        value = gl_dll.wglGetProcAddress(name)
        if value in (None, 0, 1, 2, 3, ctypes.c_void_p(-1).value):
            return ctypes.cast(getattr(gl_dll, name.decode()), ctypes.c_void_p).value
        return value

    @current_type
    def current(_):
        return gl_dll.wglGetCurrentContext()

    core.gp_mem_backend_register_gl_context.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
    core.gp_mem_backend_register_gl_context.restype = ctypes.c_int
    core.gp_mem_backend_unregister_gl_context.argtypes = [ctypes.c_size_t]
    core.gp_gl_frame_read_rgba8.argtypes = [ctypes.POINTER(Context), ctypes.c_int32,
        ctypes.c_int32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64), ctypes.c_char_p, ctypes.c_size_t]
    core.gp_gl_frame_read_rgba8.restype = ctypes.c_int
    core.nodus_tensor_read.argtypes = [ctypes.c_uint64, ctypes.c_uint64,
                                       ctypes.c_void_p, ctypes.c_uint64]
    core.nodus_tensor_read.restype = ctypes.c_int64
    core.nodus_tensor_destroy.argtypes = [ctypes.c_uint64]

    pygame.display.init()
    pygame.display.set_mode((64, 64), pygame.OPENGL | pygame.DOUBLEBUF | pygame.HIDDEN)
    identity = gl_dll.wglGetCurrentContext()
    assert identity and core.gp_mem_backend_register_gl_context(identity, identity)
    context = Context(identity, ctypes.cast(get_proc, ctypes.c_void_p).value,
                      ctypes.cast(current, ctypes.c_void_p).value, None)
    old_row_length = glGetIntegerv(GL_PACK_ROW_LENGTH)
    old_skip_pixels = glGetIntegerv(GL_PACK_SKIP_PIXELS)
    old_skip_rows = glGetIntegerv(GL_PACK_SKIP_ROWS)
    try:
        glClearColor(0.1, 0.2, 0.3, 1)
        glClear(GL_COLOR_BUFFER_BIT)
        glEnable(GL_SCISSOR_TEST)
        glScissor(0, 0, 32, 64)
        glClearColor(0.8, 0.1, 0.2, 1)
        glClear(GL_COLOR_BUFFER_BIT)
        glDisable(GL_SCISSOR_TEST)
        old_alignment = glGetIntegerv(GL_PACK_ALIGNMENT)
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        expected = bytes(glReadPixels(0, 0, 64, 64, GL_RGBA, GL_UNSIGNED_BYTE))
        glPixelStorei(GL_PACK_ALIGNMENT, old_alignment)
        glPixelStorei(GL_PACK_ROW_LENGTH, 80)
        glPixelStorei(GL_PACK_SKIP_PIXELS, 2)
        glPixelStorei(GL_PACK_SKIP_ROWS, 3)
        handle, error = ctypes.c_uint64(), ctypes.create_string_buffer(512)
        assert core.gp_gl_frame_read_rgba8(ctypes.byref(context), 0, 0, 64, 64,
                len(expected), ctypes.byref(handle), error, len(error)), error.value
        try:
            actual = ctypes.create_string_buffer(len(expected))
            assert core.nodus_tensor_read(handle.value, 0, actual, len(expected)) == len(expected)
            assert actual.raw == expected
            assert glGetIntegerv(GL_PACK_ALIGNMENT) == old_alignment
            assert glGetIntegerv(GL_PACK_ROW_LENGTH) == 80
            assert glGetIntegerv(GL_PACK_SKIP_PIXELS) == 2
            assert glGetIntegerv(GL_PACK_SKIP_ROWS) == 3
            assert len(set(expected[i:i+4] for i in range(0, len(expected), 4))) == 2
        finally:
            core.nodus_tensor_destroy(handle.value)
        denied = ctypes.c_uint64(999)
        assert not core.gp_gl_frame_read_rgba8(ctypes.byref(context), 0, 0, 64, 64,
                len(expected)-1, ctypes.byref(denied), error, len(error))
        assert denied.value == 0

        spectral_root = os.environ.get("SPECTRAL_ANALYZER_ROOT")
        if spectral_root:
            import numpy as np
            from PIL import Image
            sys.path.insert(0, spectral_root)
            from base_gl_renderer import BaseGLRenderer
            from material_db import MaterialDatabase
            from ordinary_gl_mesh_viewer import _upload_mesh
            from OpenGL.GL import glViewport

            database = MaterialDatabase()
            material = database.register("frame_transfer_proof", {
                "albedo_rgb": [0.18, 0.56, 0.92], "roughness": 0.32,
                "metallic": 0.08, "ior": 1.48, "opacity": 1.0,
                "emission_rgb": [0.035, 0.10, 0.18]})
            renderer = BaseGLRenderer(database, auto_drain=False)
            renderer.init_gl()
            renderer.set_point_lights(np.array([[0.0, 0.0, 2.0]], np.float32),
                                      np.array([[1.0, 0.92, 0.82]], np.float32),
                                      np.array([8.0], np.float32))
            quad = np.array([
                [-.75, -.75, 0, 0, 0, 1, 0, 0],
                [ .75, -.75, 0, 0, 0, 1, 1, 0],
                [ .75,  .75, 0, 0, 0, 1, 1, 1],
                [-.75, -.75, 0, 0, 0, 1, 0, 0],
                [ .75,  .75, 0, 0, 0, 1, 1, 1],
                [-.75,  .75, 0, 0, 0, 1, 0, 1]], np.float32)
            vao, *_ = _upload_mesh(quad, material)
            glViewport(0, 0, 64, 64)
            glClearColor(.012, .018, .03, 1)
            glClear(GL_COLOR_BUFFER_BIT)
            matrix = np.eye(4, dtype=np.float32)
            renderer.draw_mesh(vao, len(quad), matrix, matrix, enable_blend=False)
            phong_handle = ctypes.c_uint64()
            assert core.gp_gl_frame_read_rgba8(ctypes.byref(context), 0, 0, 64, 64,
                    64*64*4, ctypes.byref(phong_handle), error, len(error)), error.value
            try:
                phong = ctypes.create_string_buffer(64*64*4)
                assert core.nodus_tensor_read(phong_handle.value, 0, phong, len(phong)) == len(phong)
                glPixelStorei(GL_PACK_ROW_LENGTH, 0)
                glPixelStorei(GL_PACK_SKIP_PIXELS, 0)
                glPixelStorei(GL_PACK_SKIP_ROWS, 0)
                assert phong.raw == bytes(glReadPixels(0, 0, 64, 64, GL_RGBA, GL_UNSIGNED_BYTE))
                colors = set(phong.raw[i:i+4] for i in range(0, len(phong), 4))
                assert len(colors) > 2, "original Phong draw produced no lit surface"
                if output := os.environ.get("NODUS_PHONG_FRAME_PNG"):
                    Image.frombytes("RGBA", (64, 64), phong.raw).transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(output)
                if output := os.environ.get("NODUS_PHONG_FRAME_RAW"):
                    Path(output).write_bytes(phong.raw)
            finally:
                core.nodus_tensor_destroy(phong_handle.value)
    finally:
        glPixelStorei(GL_PACK_ROW_LENGTH, old_row_length)
        glPixelStorei(GL_PACK_SKIP_PIXELS, old_skip_pixels)
        glPixelStorei(GL_PACK_SKIP_ROWS, old_skip_rows)
        core.gp_mem_backend_unregister_gl_context(identity)
        pygame.display.quit()


if __name__ == "__main__":
    main()
