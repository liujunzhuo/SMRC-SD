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

"""Bounded retries for read-only requests; tool execution is never retried."""

import errno
import http.client
import json
import logging
import time
import urllib.error
import urllib.request
from urllib.request import ProxyHandler, build_opener

logger = logging.getLogger(__name__)
READ_ONLY_OPERATIONS = frozenset({"health", "task", "score"})
# The tool service is local. An empty proxy map keeps HTTP_PROXY from intercepting it.
_LOCAL_OPENER = build_opener(ProxyHandler({}))


def open_request(request, timeout):
    return _LOCAL_OPENER.open(request, timeout=timeout)


def transient_transport_error(exc):
    reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    return isinstance(reason, (TimeoutError, ConnectionError, http.client.IncompleteRead)) or isinstance(reason, OSError) and reason.errno in {errno.ECONNRESET, errno.ECONNREFUSED, errno.ECONNABORTED, errno.EPIPE, errno.ETIMEDOUT}


class Client:
    def __init__(self, url="http://127.0.0.1:28980"):
        self.url = url

    def request(self, operation, **kwargs):
        body = json.dumps({"operation": operation, **kwargs}, ensure_ascii=False).encode()
        request = urllib.request.Request(self.url, body, {"Content-Type": "application/json"})
        attempts = 3 if operation in READ_ONLY_OPERATIONS else 1
        for attempt in range(1, attempts + 1):
            try:
                with open_request(request, timeout=120) as response:
                    payload = json.load(response)
                return payload["result"]
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")
                raise RuntimeError(f"FTRL {operation} task={kwargs.get('task_id')} HTTP {exc.code}: {detail}") from None
            except (OSError, http.client.HTTPException) as exc:
                if not transient_transport_error(exc) or attempt == attempts:
                    raise RuntimeError(f"FTRL {operation} task={kwargs.get('task_id')} transport failed after {attempt}/{attempts} attempts: {exc}") from exc
                logger.warning("FTRL read-only transport retry operation=%s task=%s attempt=%s/%s error=%r", operation, kwargs.get("task_id"), attempt, attempts, exc)
                time.sleep(0.25 * attempt)
