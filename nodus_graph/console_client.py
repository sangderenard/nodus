"""Stdlib client for the canvas host read-only console (nodus-console/1 + JSON verbs)."""

from __future__ import annotations

import json
import socket
from typing import Optional


class ConsoleError(RuntimeError):
    """The server refused, closed, or answered something outside the protocol."""


class ConsoleClient:
    def __init__(self, host: str, port: int, token: str, *, timeout: float = 10.0) -> None:
        self.host, self.port, self.token, self.timeout = host, int(port), token, timeout
        self.canvas_name: Optional[str] = None
        self._sock: Optional[socket.socket] = None
        self._reader = None

    # ---- connection -------------------------------------------------------
    def __enter__(self) -> "ConsoleClient":
        self.hello()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def close(self) -> None:
        for item in (self._reader, self._sock):
            try:
                if item is not None:
                    item.close()
            except OSError:
                pass
        self._reader = self._sock = None

    def _line(self) -> str:
        try:
            raw = self._reader.readline(65536)
        except (OSError, socket.timeout) as exc:
            raise ConsoleError(f"console read failed: {exc}") from exc
        if not raw:
            raise ConsoleError("console closed the connection")
        text = raw.rstrip(b"\r\n").decode("utf-8", "replace")
        if text.startswith("REFUSED"):
            raise ConsoleError(f"console refused: {text[8:] or 'no reason'}")
        return text

    def _exact(self, count: int) -> bytes:
        try:
            data = self._reader.read(count)
        except (OSError, socket.timeout) as exc:
            raise ConsoleError(f"console read failed: {exc}") from exc
        if data is None or len(data) != count:
            raise ConsoleError("console closed mid-payload")
        return data

    def _send(self, line: str) -> None:
        if self._sock is None:
            raise ConsoleError("not connected; call hello() first")
        try:
            self._sock.sendall(line.encode("utf-8") + b"\n")
        except OSError as exc:
            raise ConsoleError(f"console write failed: {exc}") from exc

    # ---- verbs ------------------------------------------------------------
    def hello(self) -> str:
        """Connect and authenticate; returns the canvas name."""
        self.close()
        try:
            self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        except OSError as exc:
            raise ConsoleError(f"cannot connect to console: {exc}") from exc
        self._sock.settimeout(self.timeout)
        self._reader = self._sock.makefile("rb")
        self._send(f"HELLO nodus-console/1 {self.token}")
        reply = self._line().split(" ")
        if len(reply) < 4 or reply[:2] != ["OK", "console"]:
            raise ConsoleError(f"unexpected hello reply: {' '.join(reply)!r}")
        self.canvas_name = " ".join(reply[2:-1])
        return self.canvas_name

    def snapshot(self) -> dict:
        """{"revision", "canvas": bytes, "state": dict}"""
        self._send("SNAPSHOT")
        head = self._line().split(" ")
        if len(head) != 3 or head[0] != "CANVAS":
            raise ConsoleError(f"unexpected snapshot header: {' '.join(head)!r}")
        data = self._exact(int(head[2]))
        self._exact(1)  # newline after the canvas bytes
        state = self._line()
        if not state.startswith("STATE "):
            raise ConsoleError(f"unexpected snapshot trailer: {state[:40]!r}")
        return {"revision": int(head[1]), "canvas": data, "state": json.loads(state[6:])}

    def _json(self, verb: str) -> dict:
        self._send(verb)
        head = self._line().split(" ")
        if len(head) != 3 or head[0] != "JSON" or head[1] != verb:
            raise ConsoleError(f"unexpected {verb} header: {' '.join(head)!r}")
        return json.loads(self._exact(int(head[2])).decode("utf-8"))

    def digest(self) -> dict:
        return self._json("DIGEST")

    def graph(self) -> dict:
        return self._json("GRAPH")

    def tools(self) -> dict:
        return self._json("TOOLS")

    def activity(self) -> dict:
        return self._json("ACTIVITY")
