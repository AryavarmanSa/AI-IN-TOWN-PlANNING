"""WSGI entrypoint for deploying CivicLens on Vercel."""

import mimetypes
from email.message import Message
from http import HTTPStatus
from io import BytesIO

from api import ASSETS_DIR, RequestHandler


class _WSGIRequestHandler(RequestHandler):
    def __init__(self, environ):
        self.command = environ.get("REQUEST_METHOD", "GET").upper()
        self.path = environ.get("PATH_INFO", "/")
        query = environ.get("QUERY_STRING", "")
        if query:
            self.path += "?" + query
        self.headers = Message()
        for key, value in environ.items():
            if key.startswith("HTTP_"):
                name = key[5:].replace("_", "-").title()
                self.headers[name] = value
        for key in ("CONTENT_TYPE", "CONTENT_LENGTH"):
            if environ.get(key):
                self.headers[key.replace("_", "-").title()] = environ[key]
        content_length = int(environ.get("CONTENT_LENGTH") or 0)
        self.rfile = BytesIO(environ["wsgi.input"].read(content_length))
        self.wfile = BytesIO()
        self.response_status = 200
        self.response_headers = []
        self.client_address = (environ.get("REMOTE_ADDR", ""), 0)

    def send_response(self, code, message=None):
        self.response_status = code

    def send_header(self, keyword, value):
        self.response_headers.append((keyword, value))

    def end_headers(self):
        pass

    def log_message(self, format_string, *args):
        pass


def application(environ, start_response):
    path = environ.get("PATH_INFO", "/")
    method = environ.get("REQUEST_METHOD", "GET").upper()

    if method in {"GET", "HEAD"} and path.startswith("/assets/"):
        asset_root = ASSETS_DIR.resolve()
        asset_path = (asset_root / path.removeprefix("/assets/")).resolve()
        if asset_path.is_relative_to(asset_root) and asset_path.is_file():
            body = asset_path.read_bytes()
            content_type = mimetypes.guess_type(asset_path.name)[0] or "application/octet-stream"
            status = HTTPStatus.OK
            headers = [("Content-Type", content_type), ("Content-Length", str(len(body)))]
            start_response(f"{status.value} {status.phrase}", headers)
            return [b"" if method == "HEAD" else body]

    request = _WSGIRequestHandler(environ)
    dispatch = {
        "GET": request.do_GET,
        "HEAD": request.do_GET,
        "POST": request.do_POST,
        "OPTIONS": request.do_OPTIONS,
    }.get(method)
    if dispatch is None:
        request._send_json({"error": "Method not allowed"}, 405)
    else:
        dispatch()

    try:
        phrase = HTTPStatus(request.response_status).phrase
    except ValueError:
        phrase = "Unknown"
    start_response(
        f"{request.response_status} {phrase}", request.response_headers
    )
    body = request.wfile.getvalue()
    return [b"" if method == "HEAD" else body]