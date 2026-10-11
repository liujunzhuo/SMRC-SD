# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

"""Local CPU WebShop RPC server. The training process talks to it over HTTP."""

import argparse
import json
import os
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from agent_system.environments.env_package.webshop.local_backend import LocalWebshopWorker


class WebshopServer(HTTPServer):
    request_queue_size = 512

    def __init__(self, address):
        super().__init__(address, Handler)
        self.sessions = {}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, status, output):
        payload = json.dumps({"output": output}, default=lambda value: value.item()).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/health":
            self.reply(200, {"pid": os.getpid(), "sessions": len(self.server.sessions)})
        else:
            self.reply(404, "Unknown endpoint")

    def do_POST(self):
        try:
            size = int(self.headers.get("Content-Length", 0))
            if not 0 < size <= 8 * 1024 * 1024:
                raise ValueError("Invalid request length")
            payload = json.loads(self.rfile.read(size))
            if self.path == "/create":
                kwargs = dict(payload["env_kwargs"])
                data_dir = os.environ.get("WEBSHOP_DATA_DIR")
                if data_dir:
                    small = kwargs.get("num_products") == 1000
                    suffix = "_1000" if small else ""
                    data = Path(data_dir)
                    kwargs["file_path"] = str(data / f"items_shuffle{suffix}.json")
                    kwargs["attr_path"] = str(data / f"items_ins_v2{suffix}.json")
                worker = LocalWebshopWorker(payload["seed"], kwargs)
                session = uuid.uuid4().hex
                self.server.sessions[session] = worker
                result = session
            elif self.path == "/rpc":
                method = payload["method"]
                if method not in {"reset", "step", "get_goals", "get_available_actions", "render", "close"}:
                    raise ValueError("Unknown method")
                worker = self.server.sessions[payload["session"]]
                result = getattr(worker, method)(*payload.get("args", []))
                if method == "close":
                    del self.server.sessions[payload["session"]]
            else:
                self.reply(404, "Unknown endpoint")
                return
            self.reply(200, result)
        except Exception as exc:
            traceback.print_exc()
            self.reply(500, {"error": type(exc).__name__, "detail": str(exc)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    with WebshopServer(("127.0.0.1", args.port)) as server:
        print(f"WebShop CPU server ready on {args.port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
