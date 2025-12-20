from __future__ import annotations

"""
Canvas demo that now ships a tiny HTTP server.

The server exposes the canvas state as a rasterized, compressed PNG and accepts
mouse events from a simple HTML/JS client. This doubles as a quick-and-dirty API
for the canvas system when you do not have tighter integration points yet.

Run the server:
    python demo_canvas_tables.py --mode server --port 8000

Keep the legacy pygame demo:
    python demo_canvas_tables.py --mode pygame

The server is intentionally minimal: it serves one HTML page, a `/frame` PNG
endpoint, and an `/event` endpoint that forwards mouse events into the canvas.
"""

import argparse
import ctypes
import http.server
import json
import os
import pathlib
import platform
import socketserver
import struct
import sys
import threading
import time
import urllib.parse
import zlib
import subprocess
import gc
from dataclasses import dataclass
from typing import Optional, List

ROOT = pathlib.Path(__file__).resolve().parent

@dataclass
class CanvasModuleDesc(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int32),
        ("y", ctypes.c_int32),
        ("w", ctypes.c_int32),
        ("h", ctypes.c_int32),
        ("left_contacts", ctypes.c_int32),
        ("right_contacts", ctypes.c_int32),
        ("label", ctypes.c_char * 64),
    ]

@dataclass
class CanvasEdgeDesc(ctypes.Structure):
    _fields_ = [
        ("a_module", ctypes.c_int32),
        ("a_contact_idx", ctypes.c_int32),
        ("b_module", ctypes.c_int32),
        ("b_contact_idx", ctypes.c_int32),
    ]


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)


def encode_rgba_to_png(width: int, height: int, rgba: bytes) -> bytes:
    """Encode raw RGBA bytes into a minimal PNG (no external deps)."""
    raw = bytearray()
    stride = width * 4
    for y in range(height):
        raw.append(0)  # filter type 0
        start = y * stride
        raw.extend(rgba[start:start + stride])

    compressed = zlib.compress(bytes(raw), level=6)
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)

    png = bytearray()
    png.extend(b"\x89PNG\r\n\x1a\n")
    png.extend(_png_chunk(b"IHDR", header))
    png.extend(_png_chunk(b"IDAT", compressed))
    png.extend(_png_chunk(b"IEND", b""))
    return bytes(png)


def find_canvas_library() -> pathlib.Path:
    """Locate the native canvas_tables library across common build layouts."""
    env_override = os.getenv("CANVAS_TABLES_LIB") or os.getenv("C_MENU_LIB")
    if env_override:
        path = pathlib.Path(env_override)
        if path.exists():
            return path

    search_roots = [
        ROOT / "build" / "Release",
        ROOT / "build" / "Debug",
        ROOT / "build",
        ROOT,
    ]
    system = platform.system().lower()
    names: List[str]
    if system == "windows":
        names = ["canvas_tables.dll", "c_menu.dll"]
    elif system == "darwin":
        names = ["libcanvas_tables.dylib", "canvas_tables.dylib", "libc_menu.dylib", "c_menu.dylib"]
    else:
        names = ["libcanvas_tables.so", "canvas_tables.so", "libc_menu.so", "c_menu.so"]

    for root in search_roots:
        for name in names:
            candidate = root / name
            if candidate.exists():
                return candidate

    raise OSError(
        "canvas_tables library not found. Build it (cmake --build build) or set CANVAS_TABLES_LIB."
    )


def load_canvas_lib():
    lib_path = find_canvas_library()
    lib = ctypes.CDLL(str(lib_path))
    # prototypes
    lib.gp_canvas_create.restype = ctypes.c_void_p
    lib.gp_canvas_create.argtypes = (ctypes.c_int, ctypes.c_int)
    lib.gp_canvas_destroy.restype = None
    lib.gp_canvas_destroy.argtypes = (ctypes.c_void_p,)
    lib.gp_canvas_add_module.restype = ctypes.c_int
    lib.gp_canvas_add_module.argtypes = (ctypes.c_void_p, ctypes.POINTER(CanvasModuleDesc))
    lib.gp_canvas_move_module.restype = ctypes.c_int
    lib.gp_canvas_move_module.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int)
    lib.gp_canvas_add_edge.restype = ctypes.c_int
    lib.gp_canvas_add_edge.argtypes = (ctypes.c_void_p, ctypes.POINTER(CanvasEdgeDesc))
    lib.gp_canvas_set_cable_style.restype = ctypes.c_int
    lib.gp_canvas_set_cable_style.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int)
    lib.gp_canvas_set_edge_hues.restype = ctypes.c_int
    lib.gp_canvas_set_edge_hues.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_float)
    lib.gp_canvas_raster_rgba.restype = ctypes.c_int
    lib.gp_canvas_raster_rgba.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int)
    lib.gp_canvas_on_click.restype = ctypes.c_int
    lib.gp_canvas_on_click.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int)
    # new mouse/drag and table helpers
    # prefer the real on_mouse_down symbol if exported; otherwise fall back to on_click
    try:
        lib.gp_canvas_on_mouse_down.restype = ctypes.c_int
        lib.gp_canvas_on_mouse_down.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int)
    except AttributeError:
        # fallback: alias to gp_canvas_on_click so older builds still work
        lib.gp_canvas_on_mouse_down = lib.gp_canvas_on_click
        lib.gp_canvas_on_mouse_down.restype = ctypes.c_int
        lib.gp_canvas_on_mouse_down.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int)
    lib.gp_canvas_on_mouse_move.restype = ctypes.c_int
    lib.gp_canvas_on_mouse_move.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int)
    lib.gp_canvas_on_mouse_up.restype = ctypes.c_int
    lib.gp_canvas_on_mouse_up.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int)
    lib.gp_canvas_set_offset.restype = ctypes.c_int
    lib.gp_canvas_set_offset.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int)
    lib.gp_canvas_get_offset.restype = ctypes.c_int
    lib.gp_canvas_get_offset.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
    lib.gp_canvas_get_scroll_flags.restype = ctypes.c_int
    lib.gp_canvas_get_scroll_flags.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
    lib.gp_canvas_step.restype = ctypes.c_int
    lib.gp_canvas_step.argtypes = (ctypes.c_void_p, ctypes.c_float)
    # keyboard forwarding for interactive demos
    try:
        lib.gp_canvas_on_key.restype = ctypes.c_int
        lib.gp_canvas_on_key.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)
    except AttributeError:
        # older builds may not export on_key
        pass
    lib.gp_canvas_attach_table.restype = ctypes.c_int
    lib.gp_canvas_attach_table.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int)
    lib.gp_canvas_create_table.restype = ctypes.c_int
    lib.gp_canvas_create_table.argtypes = (ctypes.c_void_p, ctypes.c_int)
    lib.gp_canvas_destroy_table.restype = ctypes.c_int
    lib.gp_canvas_destroy_table.argtypes = (ctypes.c_void_p, ctypes.c_int)
    lib.gp_canvas_set_container_table.restype = ctypes.c_int
    lib.gp_canvas_set_container_table.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int)
    lib.gp_canvas_get_container_table.restype = ctypes.c_void_p
    lib.gp_canvas_get_container_table.argtypes = (ctypes.c_void_p,)
    return lib


def make_module(x, y, w, h, left, right, label: str) -> CanvasModuleDesc:
    m = CanvasModuleDesc()
    m.x = int(x)
    m.y = int(y)
    m.w = int(w)
    m.h = int(h)
    m.left_contacts = int(left)
    m.right_contacts = int(right)
    b = label.encode("utf-8")[:63]
    padded = b + b"\0" * (64 - len(b))
    m.label = padded
    return m


class CanvasBackend:
    """Owns the native canvas context and exposes raster/event helpers."""

    def __init__(self, width: int, height: int, step_dt: float = 1.0 / 60.0):
        self.width = width
        self.height = height
        self.step_dt = step_dt
        self.lib = load_canvas_lib()
        self.ctx = self.lib.gp_canvas_create(width, height)
        if not self.ctx:
            raise RuntimeError("gp_canvas_create failed")

        self.buf_len = width * height * 4
        self.buf = (ctypes.c_ubyte * self.buf_len)()
        self.lock = threading.Lock()
        self._running = threading.Event()
        self._running.set()

        self.lib.gp_canvas_set_cable_style(self.ctx, 5, 2)
        hues = (ctypes.c_float * 3)(0.0, 0.33, 0.66)
        self.lib.gp_canvas_set_edge_hues(self.ctx, hues, 3, ctypes.c_float(0.9))
        # Kick the raster once so internal state (rope sim) is ready.
        self.lib.gp_canvas_raster_rgba(self.ctx, self.buf, self.buf_len)

        self._step_thread = threading.Thread(target=self._step_loop, daemon=True)
        self._step_thread.start()

    def _step_loop(self) -> None:
        while self._running.is_set():
            with self.lock:
                self.lib.gp_canvas_step(self.ctx, ctypes.c_float(self.step_dt))
            time.sleep(self.step_dt)

    def shutdown(self) -> None:
        self._running.clear()
        self._step_thread.join(timeout=1.0)
        with self.lock:
            self.lib.gp_canvas_destroy(self.ctx)

    def handle_mouse(self, kind: str, x: int, y: int) -> bool:
        with self.lock:
            if kind == "down":
                return bool(self.lib.gp_canvas_on_mouse_down(self.ctx, x, y))
            if kind == "up":
                return bool(self.lib.gp_canvas_on_mouse_up(self.ctx, x, y))
            if kind == "move":
                return bool(self.lib.gp_canvas_on_mouse_move(self.ctx, x, y))
            if kind == "click":
                return bool(self.lib.gp_canvas_on_click(self.ctx, x, y))
        return False

    def raster_png(self) -> bytes:
        with self.lock:
            ok = self.lib.gp_canvas_raster_rgba(self.ctx, self.buf, self.buf_len)
            if not ok:
                raise RuntimeError("gp_canvas_raster_rgba failed")
            rgba = bytes(bytearray(self.buf))
        return encode_rgba_to_png(self.width, self.height, rgba)


def _free_cdll(cdll: ctypes.CDLL) -> None:
    """Attempt to free a ctypes CDLL handle so the DLL file can be overwritten.

    Caller must ensure no native threads or objects from the library are live.
    """
    handle = getattr(cdll, "_handle", None)
    if not handle:
        return
    try:
        if sys.platform.startswith("win"):
            ctypes.windll.kernel32.FreeLibrary(ctypes.c_void_p(handle))
        else:
            # POSIX: call dlclose from the C library
            libc = ctypes.CDLL(None)
            try:
                libc.dlclose(ctypes.c_void_p(handle))
            except AttributeError:
                # Some platforms expose dlclose in libdl
                libdl = ctypes.CDLL("libdl.so")
                libdl.dlclose(ctypes.c_void_p(handle))
    except Exception:
        pass
    # drop references and run finalizers
    try:
        del cdll
    except Exception:
        pass
    gc.collect()
    time.sleep(0.05)


def _build_project() -> None:
    # Configure and build into `build` (mirrors simple CMake usage)
    subprocess.check_call(["cmake", "-B", "build", "-S", "."])
    subprocess.check_call(["cmake", "--build", "build", "--config", "Release"])


def _server_reload_backend(server: "CanvasHTTPServer") -> None:
    """Rebuild native library and swap a fresh CanvasBackend into the running server.

    This function assumes the caller holds `server.backend_lock` to prevent races.
    """
    old = server.backend
    width = old.width
    height = old.height
    step_dt = old.step_dt

    # stop and destroy native state
    old.shutdown()

    # try to release the loaded library so it can be overwritten
    try:
        _free_cdll(old.lib)
    except Exception:
        pass

    # build native project (may raise)
    _build_project()

    # replace backend with a newly-loaded one
    new = CanvasBackend(width, height, step_dt)
    server.backend = new
    server.rendered_html = HTML_TEMPLATE.format(w=width, h=height)


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Canvas tables server</title>
  <style>
    body {{ background: #0c0c10; color: #ddd; font-family: sans-serif; }}
    #frame-wrap {{ position: relative; width: {w}px; height: {h}px; }}
    #canvas-img {{ width: {w}px; height: {h}px; display: block; }}
    #hit-layer {{
      position: absolute;
      left: 0;
      top: 0;
      width: {w}px;
      height: {h}px;
      cursor: crosshair;
    }}
    #info {{ margin-top: 8px; font-size: 14px; color: #aaa; }}
  </style>
</head>
<body>
  <h3>Canvas tables API shim</h3>
  <div id="frame-wrap">
    <img id="canvas-img" src="/frame" alt="canvas frame">
    <div id="hit-layer"></div>
  </div>
  <div id="info">Dragging and clicking sends events to the backend. Frames auto-refresh.</div>
    <div style="margin-top:8px">
          <button id="reload-btn">Reload native library</button>
          <button id="unload-btn" style="margin-left:8px">Unload native library</button>
          <button id="load-btn" style="margin-left:8px">Load native library</button>
          <span id="reload-status" style="margin-left:8px;color:#aaa;font-size:13px"></span>
    </div>
  <script>
    const hit = document.getElementById('hit-layer');
    const img = document.getElementById('canvas-img');
    let mouseDown = false;

    async function sendEvent(kind, ev) {{
      const rect = hit.getBoundingClientRect();
      const x = Math.round(ev.clientX - rect.left);
      const y = Math.round(ev.clientY - rect.top);
      await fetch('/event', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ type: kind, x, y }})
      }});
    }}

    hit.addEventListener('mousedown', ev => {{ mouseDown = true; sendEvent('down', ev); }});
    hit.addEventListener('mouseup', ev => {{ mouseDown = false; sendEvent('up', ev); }});
    hit.addEventListener('mouseleave', ev => {{ if (mouseDown) sendEvent('up', ev); mouseDown = false; }});
    hit.addEventListener('mousemove', ev => {{ if (mouseDown) sendEvent('move', ev); }});
    // Avoid sending a separate 'click' event in addition to 'down'/'up'.
    // Some canvas backends treat 'click' and 'up' as distinct actions and
    // sending both can toggle a control twice (appear to revert immediately).
    // If a single logical click is preferred, rely on the 'down'+'up' sequence
    // or replace this with a single 'click' handler and remove the down/up
    // handlers instead.

    async function refreshFrame() {{
      const resp = await fetch('/frame');
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      img.src = url;
      img.onload = () => URL.revokeObjectURL(url);
    }}

    setInterval(refreshFrame, 150);

        document.getElementById('reload-btn').addEventListener('click', async () => {{
            const status = document.getElementById('reload-status');
            status.textContent = 'reloading...';
            try {{
                const r = await fetch('/reload', {{ method: 'POST' }});
                if (r.ok) {{
                    status.textContent = 'reloaded';
                    setTimeout(() => status.textContent = '', 3000);
                }} else {{
                    const t = await r.text();
                    status.textContent = 'error: ' + t;
                }}
            }} catch (e) {{
                status.textContent = 'error';
            }}
            }});

            document.getElementById('unload-btn').addEventListener('click', async () => {{
                const status = document.getElementById('reload-status');
                status.textContent = 'unloading...';
                try {{
                    const r = await fetch('/unload', {{ method: 'POST' }});
                    if (r.ok) {{
                        status.textContent = 'unloaded';
                        setTimeout(() => status.textContent = '', 3000);
                    }} else {{
                        const t = await r.text();
                        status.textContent = 'error: ' + t;
                    }}
                }} catch (e) {{
                    status.textContent = 'error';
                }}
            }});

            document.getElementById('load-btn').addEventListener('click', async () => {{
                const status = document.getElementById('reload-status');
                status.textContent = 'loading...';
                try {{
                    const r = await fetch('/load', {{ method: 'POST' }});
                    if (r.ok) {{
                        status.textContent = 'loaded';
                        setTimeout(() => status.textContent = '', 3000);
                    }} else {{
                        const t = await r.text();
                        status.textContent = 'error: ' + t;
                    }}
                }} catch (e) {{
                    status.textContent = 'error';
                }}
            }});
  </script>
</body>
</html>
"""


class CanvasRequestHandler(http.server.BaseHTTPRequestHandler):
    server_version = "CanvasTablesServer/0.1"

    def _send_response(self, code: int, body: bytes, content_type: str = "text/plain") -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            with self.server.backend_lock:
                html = self.server.rendered_html  # type: ignore[attr-defined]
            self._send_response(200, html.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/frame":
            with self.server.backend_lock:
                if getattr(self.server, "backend", None) is None:
                    body = json.dumps({"error": "native library unloaded"}).encode("utf-8")
                    self._send_response(503, body, "application/json")
                    return
                try:
                    frame = self.server.backend.raster_png()  # type: ignore[attr-defined]
                except Exception as exc:  # pragma: no cover - informational
                    body = json.dumps({"error": str(exc)}).encode("utf-8")
                    self._send_response(500, body, "application/json")
                    return
            self._send_response(200, frame, "image/png")
            return
        if parsed.path == "/meta":
            with self.server.backend_lock:
                if getattr(self.server, "backend", None) is None:
                    body = json.dumps({"error": "native library unloaded"}).encode("utf-8")
                    self._send_response(503, body, "application/json")
                    return
                data = {
                    "width": self.server.backend.width,  # type: ignore[attr-defined]
                    "height": self.server.backend.height,  # type: ignore[attr-defined]
                    "step_dt": self.server.backend.step_dt,  # type: ignore[attr-defined]
                }
            self._send_response(200, json.dumps(data).encode("utf-8"), "application/json")
            return
        self.send_error(404, "Not Found")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/unload":
            try:
                with self.server.backend_lock:
                    if getattr(self.server, "backend", None) is None:
                        body = json.dumps({"error": "already unloaded"}).encode("utf-8")
                        self._send_response(400, body, "application/json")
                        return
                    old = self.server.backend
                    # remember size for reload
                    try:
                        self.server.last_backend_size = (old.width, old.height, old.step_dt)  # type: ignore[attr-defined]
                    except Exception:
                        self.server.last_backend_size = None
                    old.shutdown()
                    try:
                        _free_cdll(old.lib)
                    except Exception:
                        pass
                    self.server.backend = None
                    # keep page dimensions stable
                    w, h = (self.server.last_backend_size[0], self.server.last_backend_size[1]) if self.server.last_backend_size else (1000, 360)
                    self.server.rendered_html = HTML_TEMPLATE.format(w=w, h=h)
            except Exception as exc:
                body = json.dumps({"error": str(exc)}).encode("utf-8")
                self._send_response(500, body, "application/json")
                return
            body = json.dumps({"unloaded": True}).encode("utf-8")
            self._send_response(200, body, "application/json")
            return

        if parsed.path == "/load":
            try:
                with self.server.backend_lock:
                    if getattr(self.server, "backend", None) is not None:
                        body = json.dumps({"error": "already loaded"}).encode("utf-8")
                        self._send_response(400, body, "application/json")
                        return
                    if getattr(self.server, "last_backend_size", None):
                        w, h, step_dt = self.server.last_backend_size
                    else:
                        w, h, step_dt = 1000, 360, 1.0 / 60.0
                    new = CanvasBackend(w, h, step_dt)
                    self.server.backend = new
                    self.server.rendered_html = HTML_TEMPLATE.format(w=w, h=h)
            except Exception as exc:
                body = json.dumps({"error": str(exc)}).encode("utf-8")
                self._send_response(500, body, "application/json")
                return
            body = json.dumps({"loaded": True}).encode("utf-8")
            self._send_response(200, body, "application/json")
            return

        if parsed.path == "/reload":
            try:
                with self.server.backend_lock:
                    _server_reload_backend(self.server)
            except Exception as exc:
                body = json.dumps({"error": str(exc)}).encode("utf-8")
                self._send_response(500, body, "application/json")
                return
            body = json.dumps({"reloaded": True}).encode("utf-8")
            self._send_response(200, body, "application/json")
            return

        if parsed.path != "/event":
            self.send_error(404, "Not Found")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error(400, "Invalid content length")
            return
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw or "{}")
            kind = str(payload.get("type", ""))
            x = int(payload.get("x", -1))
            y = int(payload.get("y", -1))
        except Exception:
            self.send_error(400, "Malformed JSON")
            return

        handled = False
        with self.server.backend_lock:
            if getattr(self.server, "backend", None) is None:
                body = json.dumps({"error": "native library unloaded"}).encode("utf-8")
                self._send_response(503, body, "application/json")
                return
            if 0 <= x < self.server.backend.width and 0 <= y < self.server.backend.height:  # type: ignore[attr-defined]
                handled = self.server.backend.handle_mouse(kind, x, y)  # type: ignore[attr-defined]
        body = json.dumps({"handled": handled}).encode("utf-8")
        self._send_response(200, body, "application/json")

    def log_message(self, fmt: str, *args) -> None:  # noqa: D401
        """Reduce noise in stdout by suppressing default logging."""
        return


class CanvasHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True

    def __init__(self, server_address, RequestHandlerClass, backend: CanvasBackend):
        super().__init__(server_address, RequestHandlerClass)
        self.backend = backend
        self.backend_lock = threading.Lock()
        self.rendered_html = HTML_TEMPLATE.format(w=backend.width, h=backend.height)
        # remember last known backend size so unload/load can preserve dimensions
        try:
            self.last_backend_size = (backend.width, backend.height, backend.step_dt)
        except Exception:
            self.last_backend_size = None


def run_server(host: str, port: int, width: int, height: int, step_dt: float) -> int:
    backend = CanvasBackend(width, height, step_dt=step_dt)
    server = CanvasHTTPServer((host, port), CanvasRequestHandler, backend)
    print(f"Serving canvas frames on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        server.server_close()
        backend.shutdown()
    return 0


def run_pygame_demo(width: int, height: int) -> int:
    # Import pygame lazily so the server mode does not require SDL.
    import pygame

    pygame.init()
    table_w = width
    table_h = 0
    canvas_h = height
    info_h = 28
    screen = pygame.display.set_mode((table_w, table_h + canvas_h + info_h))
    pygame.display.set_caption("Canvas + Tables demo")

    try:
        lib = load_canvas_lib()
    except Exception as e:
        print("Failed to load canvas API:", e)
        return 1

    cctx = lib.gp_canvas_create(table_w, canvas_h)
    if not cctx:
        print("gp_canvas_create failed")
        return 1

    lib.gp_canvas_set_cable_style(cctx, 5, 2)
    hues = (ctypes.c_float * 3)(0.0, 0.33, 0.66)
    lib.gp_canvas_set_edge_hues(cctx, hues, 3, ctypes.c_float(0.9))

    buf_len = table_w * canvas_h * 4
    buf = (ctypes.c_ubyte * buf_len)()
    lib.gp_canvas_raster_rgba(cctx, buf, buf_len)

    clock = pygame.time.Clock()
    running = True
    font = pygame.font.SysFont(None, 18)

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                else:
                    sc = getattr(ev, "scancode", 0)
                    mods = pygame.key.get_mods()
                    try:
                        lib.gp_canvas_on_key(cctx, int(ev.key), int(sc), 1, int(mods))
                    except Exception:
                        # ignore if symbol not present or call fails
                        pass
            elif ev.type == pygame.KEYUP:
                sc = getattr(ev, "scancode", 0)
                mods = pygame.key.get_mods()
                try:
                    lib.gp_canvas_on_key(cctx, int(ev.key), int(sc), 0, int(mods))
                except Exception:
                    pass
            elif ev.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION):
                mx, my = ev.pos
                if my >= table_h and my < table_h + canvas_h:
                    cx = int(mx)
                    cy = int(my - table_h)
                    if ev.type == pygame.MOUSEBUTTONDOWN:
                        lib.gp_canvas_on_mouse_down(cctx, cx, cy)
                    elif ev.type == pygame.MOUSEBUTTONUP:
                        lib.gp_canvas_on_mouse_up(cctx, cx, cy)
                    elif ev.type == pygame.MOUSEMOTION:
                        lib.gp_canvas_on_mouse_move(cctx, cx, cy)

        surf_table = None
        if table_h <= 0:
            table_h = 1
        if table_h > 0:
            surf_table = pygame.Surface((table_w, table_h), pygame.SRCALPHA)
            surf_table.fill((20, 20, 24))

        lib.gp_canvas_step(cctx, ctypes.c_float(1.0 / 60.0))
        ok = lib.gp_canvas_raster_rgba(cctx, buf, buf_len)
        if ok:
            pybuf = bytes(bytearray(buf))
            surf_canvas = pygame.image.frombuffer(pybuf, (table_w, canvas_h), "RGBA").convert_alpha()
        else:
            surf_canvas = None

        screen.fill((12, 12, 16))
        screen.blit(surf_table, (0, 0))
        if surf_canvas:
            screen.blit(surf_canvas, (0, table_h))

        info_rect = pygame.Rect(0, table_h + canvas_h, table_w, info_h)
        pygame.draw.rect(screen, (30, 30, 36), info_rect)
        txt = font.render("Canvas demo: click/drag to spawn and wire tables", True, (220, 220, 220))
        screen.blit(txt, (8, table_h + canvas_h + 4))

        pygame.display.flip()
        clock.tick(60)

    lib.gp_canvas_destroy(cctx)
    pygame.quit()
    return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canvas tables demo / server")
    parser.add_argument("--mode", choices=["server", "pygame"], default="server", help="Run HTTP server or local pygame demo")
    parser.add_argument("--host", default="0.0.0.0", help="Host for HTTP server")
    parser.add_argument("--port", type=int, default=8000, help="Port for HTTP server")
    parser.add_argument("--width", type=int, default=1000, help="Canvas width in pixels")
    parser.add_argument("--height", type=int, default=360, help="Canvas height in pixels")
    parser.add_argument("--step-dt", type=float, default=1.0 / 60.0, help="Simulation step size in seconds")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.mode == "server":
        return run_server(args.host, args.port, args.width, args.height, args.step_dt)
    return run_pygame_demo(args.width, args.height)


if __name__ == "__main__":
    sys.exit(main())
