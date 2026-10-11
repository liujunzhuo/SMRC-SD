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

"""Standard-library transport for isolated CPU WebShop services."""

import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


def configured_service_urls():
    path = os.environ.get("WEBSHOP_SERVICE_FILE")
    if not path:
        return []
    urls = json.loads(Path(path).read_text())["urls"]
    if not urls:
        raise ValueError(f"No WebShop services in {path}")
    return urls


class RemoteWebshopWorker:
    def __init__(self, seed, env_kwargs, service_url, timeout=180):
        self.url = service_url.rstrip("/")
        self.timeout = timeout
        self.opener = build_opener(ProxyHandler({}))
        self.closed = False
        self.session = self._post("/create", {"seed": seed, "env_kwargs": env_kwargs})

    def _post(self, route, payload):
        request = Request(self.url + route, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                return json.load(response)["output"]
        except HTTPError as exc:
            raise RuntimeError(f"WebShop {route}: HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc
        except (URLError, OSError) as exc:
            # A timed-out step may have executed already; never replay it blindly.
            raise RuntimeError(f"WebShop {route} at {self.url}: {exc}") from exc

    def _call(self, method, *args):
        if self.closed:
            raise RuntimeError("WebShop session is closed")
        return self._post("/rpc", {"session": self.session, "method": method, "args": args})

    def reset(self, idx, kwargs=None):
        return self._call("reset", idx, kwargs)

    def step(self, action):
        return self._call("step", action)

    def get_goals(self):
        return self._call("get_goals")

    def get_available_actions(self):
        return self._call("get_available_actions")

    def render(self, mode):
        return self._call("render", mode)

    def close(self):
        if not self.closed:
            self._call("close")
            self.closed = True
