import asyncio
import http.server
import json
import threading
from pathlib import Path

import websockets

_STATIC_DIR = Path(__file__).parent / "static"


class _HTMLHandler(http.server.BaseHTTPRequestHandler):
    ws_port: int = 8081

    def do_GET(self) -> None:
        html = (_STATIC_DIR / "index.html").read_text(encoding="utf-8")
        html = html.replace("__WS_PORT__", str(self.ws_port))
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        pass


class WebServer:
    def __init__(self, http_port: int, ws_port: int) -> None:
        self.http_port = http_port
        self.ws_port = ws_port
        self._clients: set = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self) -> None:
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._run_ws, daemon=True).start()

        _HTMLHandler.ws_port = self.ws_port
        httpd = http.server.HTTPServer(("0.0.0.0", self.http_port), _HTMLHandler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()

        print(f"🌐 字幕ページ: http://localhost:{self.http_port}", flush=True)

    # --- WebSocket ---

    def _run_ws(self) -> None:
        asyncio.set_event_loop(self._loop)
        assert self._loop is not None
        self._loop.run_until_complete(self._serve_ws())

    async def _serve_ws(self) -> None:
        async with websockets.serve(self._ws_handler, "0.0.0.0", self.ws_port):
            await asyncio.get_running_loop().create_future()  # run forever

    async def _ws_handler(self, ws: object) -> None:
        self._clients.add(ws)
        try:
            await ws.wait_closed()  # type: ignore[union-attr]
        finally:
            self._clients.discard(ws)

    # --- ブロードキャスト (スレッドセーフ) ---

    def broadcast(self, text: str) -> None:
        if self._loop is None or not self._clients:
            return
        payload = json.dumps({"text": text})
        asyncio.run_coroutine_threadsafe(self._broadcast(payload), self._loop)

    async def _broadcast(self, payload: str) -> None:
        if self._clients:
            await asyncio.gather(
                *(ws.send(payload) for ws in set(self._clients)),  # type: ignore[union-attr]
                return_exceptions=True,
            )
