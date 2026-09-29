"""Small synchronous HTTP service; inference performance is intentionally unoptimized."""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch

from .data import DataError, strict_loads
from .token_safety import TokenSafetyError


def make_server(predictor, host="127.0.0.1", port=8000):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, value):
            body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/v1/models":
                self.respond(
                    200,
                    {"models": [predictor.metadata]},
                )
            else:
                self.respond(404, {"error": {"type": "not_found", "message": "Unknown route"}})

        def do_POST(self):
            if self.path != "/v1/systemone":
                self.respond(404, {"error": {"type": "not_found", "message": "Unknown route"}})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 16 * 1024 * 1024:
                    raise DataError("Content-Length must be between 1 byte and 16 MiB")
                request = strict_loads(self.rfile.read(size).decode("utf-8"))
                if not isinstance(request, dict):
                    raise DataError("Request must be a JSON object")
                result = predictor.predict(request)
            except (ValueError, TokenSafetyError, UnicodeDecodeError) as exc:
                if "exceeds token budget" in str(exc):
                    # A capacity limit, not a malformed request; clients look for these words.
                    message = f"Request has too many tokens: {exc}"
                    self.respond(422, {"error": {"type": "capacity", "message": message}})
                else:
                    self.respond(400, {"error": {"type": "invalid_request", "message": str(exc)}})
                return
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                message = "Request has too many tokens for the memory of this device; input was not truncated"
                self.respond(422, {"error": {"type": "capacity", "message": message}})
                return
            except Exception:
                import traceback

                traceback.print_exc()
                self.respond(
                    500, {"error": {"type": "inference_error", "message": "Model execution failed"}}
                )
                return
            self.respond(200, result)

    return HTTPServer((host, port), Handler)
