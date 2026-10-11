# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Local HTTP benchmark service with bounded, killable, per-step tool children."""

import argparse
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .core import score_trace


class BenchmarkHTTPServer(ThreadingHTTPServer):
    # A 16-worker client submits bursts of task/score requests. The stdlib
    # default backlog of five is smaller than one such burst.
    request_queue_size = 128


class Benchmark:
    def __init__(self, root, workers=16):
        self.tasks = {}
        for split in ("train", "test"):
            with open(Path(root) / f"{split}.jsonl") as handle:
                for line in handle:
                    row = json.loads(line)
                    self.tasks[row["task_id"]] = row
        self.limit = threading.BoundedSemaphore(workers)

    def call(self, request):
        operation = request["operation"]
        if operation == "health":
            return {"tasks": len(self.tasks), "protocol": "ftrl-v1"}
        task = self.tasks[request["task_id"]]
        if operation == "task":
            return {k: task[k] for k in ("task_id", "messages", "tools", "data_source")}
        if operation == "score":
            return score_trace(task, request["events"])
        if operation != "execute":
            raise ValueError("Unknown operation")
        calls = request["calls"]
        if not calls:
            return []
        with self.limit:
            try:
                child = subprocess.run([sys.executable, str(Path(__file__).with_name("execute.py"))], input=json.dumps({"codes": task["codes"], "calls": calls}), capture_output=True, text=True, timeout=12)
            except subprocess.TimeoutExpired:
                raise RuntimeError("Tool step exceeded hard timeout; child killed; no retry") from None
            if child.returncode:
                raise RuntimeError(f"Tool child exited {child.returncode}: {child.stderr[-1000:]}")
            return json.loads(child.stdout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--port", type=int, default=28980)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    benchmark = Benchmark(args.root, args.workers)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = {}
            try:
                size = int(self.headers["Content-Length"])
                if size > 8 * 1024**2:
                    raise ValueError("Request too large")
                request = json.loads(self.rfile.read(size))
                result = benchmark.call(request)
                payload, status = {"result": result}, 200
            except Exception as exc:
                payload, status = {"error": str(exc)}, 500
                print(json.dumps({"event": "request_error", "request": request, "error": str(exc)}, ensure_ascii=False), flush=True)
            blob = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)

        def log_message(self, *args):
            pass

    server = BenchmarkHTTPServer(("127.0.0.1", args.port), Handler)
    print(json.dumps({"port": args.port, "tasks": len(benchmark.tasks), "request_queue_size": server.request_queue_size}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
