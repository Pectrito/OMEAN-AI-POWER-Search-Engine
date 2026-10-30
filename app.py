from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from search_engine import LocalSearchEngine, crawl_web


class SearchServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], data_dir: Path) -> None:
        super().__init__(address, SearchHandler)
        self.engine = LocalSearchEngine(data_dir)
        self.engine_lock = threading.Lock()
        self.data_dir = data_dir


class SearchHandler(BaseHTTPRequestHandler):
    server: SearchServer

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_file(Path(__file__).parent / "static" / "index.html", "text/html; charset=utf-8")
        elif parsed.path == "/app.js":
            self._send_file(Path(__file__).parent / "static" / "app.js", "text/javascript; charset=utf-8")
        elif parsed.path == "/styles.css":
            self._send_file(Path(__file__).parent / "static" / "styles.css", "text/css; charset=utf-8")
        elif parsed.path == "/api/search":
            query = parse_qs(parsed.query).get("q", [""])[0].strip()
            with self.server.engine_lock:
                results = self.server.engine.search(query)
            self._send_json({"query": query, "results": [result.__dict__ for result in results]})
        elif parsed.path == "/api/health":
            metadata = self.server.engine.model_metadata
            self._send_json(
                {
                    "status": "ok",
                    "neural_model": {
                        "available": self.server.engine._neural_model is not None,
                        "trained": bool(metadata),
                        "parameters": metadata.get("parameter_count", 0),
                        "steps": metadata.get("training_steps", 0),
                        "error": self.server.engine._neural_model_error,
                    },
                    "indexed_documents": self.server.engine.document_count(),
                }
            )
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        if route == "/api/feedback":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8192:
                    raise ValueError("Request body must be between 1 and 8192 bytes.")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("Request body must be a JSON object.")
                query = payload.get("query")
                source_id = payload.get("source_id")
                relevant = payload.get("relevant")
                if not isinstance(query, str) or not isinstance(source_id, str):
                    raise ValueError("query and source_id must be strings.")
                if not isinstance(relevant, bool):
                    raise ValueError("relevant must be true or false.")
                with self.server.engine_lock:
                    status = self.server.engine.rate_result(
                        query,
                        source_id,
                        relevant,
                    )
                self._send_json(status)
            except (ValueError, json.JSONDecodeError) as error:
                self._send_json({"error": str(error)}, status=400)
            return
        if route != "/api/reindex":
            self.send_error(404)
            return
        with self.server.engine_lock:
            result = self.server.engine.index_directory()
            response = dict(result)
            response["indexed_documents"] = self.server.engine.document_count()
        self._send_json(response)

    def _send_json(self, payload: object, status: int = 200) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _send_file(self, path: Path, content_type: str) -> None:
        try:
            content = path.read_bytes()
        except OSError:
            self.send_error(500, "Could not read the web interface.")
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args: object) -> None:
        print(f"[search] {self.address_string()} - {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local document search and summarization prototype.")
    parser.add_argument("--data", type=Path, default=Path("data"), help="Folder containing documents (default: ./data).")
    parser.add_argument("--host", default="127.0.0.1", help="Interface to bind (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port (default: 8000).")
    parser.add_argument("--crawl", metavar="URL", help="Crawl a small same-origin set of public pages, then exit.")
    parser.add_argument("--max-pages", type=int, default=10, help="Maximum crawl pages (1-50; default: 10).")
    parser.add_argument("--index-only", action="store_true", help="Index local files and exit.")
    args = parser.parse_args()

    if args.crawl:
        result = crawl_web(args.crawl, args.data, args.max_pages)
        engine = LocalSearchEngine(args.data)
        try:
            output: dict[str, object] = dict(result)
            output["index"] = engine.index_directory()
        finally:
            engine.close()
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    if args.index_only:
        engine = LocalSearchEngine(args.data)
        try:
            print(json.dumps(engine.index_directory(), ensure_ascii=False, indent=2))
        finally:
            engine.close()
        return

    server = SearchServer((args.host, args.port), args.data)
    try:
        result = server.engine.index_directory()
        print(f"Indexed {result['indexed']} documents from {server.data_dir}")
        print(f"Search UI: http://{args.host}:{args.port}")
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping search server.")
    finally:
        server.engine.close()
        server.server_close()


if __name__ == "__main__":
    main()
