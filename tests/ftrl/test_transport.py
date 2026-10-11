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

"""Transport failures must not change benchmark semantics or replay tools."""

import http.client
import io
import json
import socket
import struct
import threading
import urllib.error

import pytest

from agent_system.environments.env_package.ftrl.client import Client


@pytest.mark.parametrize("operation", ["score", "task", "health"])
@pytest.mark.parametrize("error", [ConnectionResetError(104, "reset"), urllib.error.URLError(TimeoutError("timeout")), http.client.IncompleteRead(b"partial"), http.client.RemoteDisconnected("closed")])
def test_read_only_transport_retries_same_body(monkeypatch, operation, error):
    bodies = []
    sleeps = []

    def open_request(request, timeout):
        bodies.append(request.data)
        if len(bodies) < 3:
            raise error
        return io.BytesIO(b'{"result":{"solve_f1":0.5}}')

    monkeypatch.setattr("agent_system.environments.env_package.ftrl.client.open_request", open_request)
    monkeypatch.setattr("agent_system.environments.env_package.ftrl.client.time.sleep", sleeps.append)
    assert Client().request(operation, task_id="train:x", events=[]) == {"solve_f1": 0.5}
    assert len(bodies) == 3 and len(set(bodies)) == 1
    assert sleeps == [0.25, 0.5]


@pytest.mark.parametrize("operation,attempts", [("score", 3), ("execute", 1), ("unknown", 1)])
def test_exhausted_transport_fails_with_identity(monkeypatch, operation, attempts):
    calls = []

    def fail(request, timeout):
        calls.append(request.data)
        raise ConnectionResetError(104, "reset")

    monkeypatch.setattr("agent_system.environments.env_package.ftrl.client.open_request", fail)
    monkeypatch.setattr("agent_system.environments.env_package.ftrl.client.time.sleep", lambda _: None)
    with pytest.raises(RuntimeError, match=f"{operation} task=train:x transport failed after {attempts}/{attempts}"):
        Client().request(operation, task_id="train:x")
    assert len(calls) == attempts


@pytest.mark.parametrize("error", [urllib.error.HTTPError("http://local", 500, "benchmark error", {}, io.BytesIO(b"bad task")), ValueError("bad json")])
def test_application_errors_are_not_retried(monkeypatch, error):
    calls = []

    def fail(request, timeout):
        calls.append(request.data)
        raise error

    monkeypatch.setattr("agent_system.environments.env_package.ftrl.client.open_request", fail)
    with pytest.raises((RuntimeError, ValueError)):
        Client().request("score", task_id="train:x")
    assert len(calls) == 1


def test_real_tcp_reset_recovers_score_without_reexecuting_tools():
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    port = listener.getsockname()[1]
    received = []
    errors = []

    def serve():
        try:
            listener.settimeout(5)
            for attempt in range(2):
                connection, _ = listener.accept()
                with connection:
                    connection.settimeout(5)
                    data = b""
                    while b"\r\n\r\n" not in data:
                        data += connection.recv(4096)
                    headers, body = data.split(b"\r\n\r\n", 1)
                    size = int(next(line.split(b":", 1)[1] for line in headers.split(b"\r\n") if line.lower().startswith(b"content-length:")))
                    while len(body) < size:
                        body += connection.recv(4096)
                    received.append(json.loads(body))
                    if attempt == 0:
                        connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                    else:
                        payload = b'{"result":{"solve_f1":0.5}}'
                        connection.sendall(b"HTTP/1.0 200 OK\r\nContent-Length: " + str(len(payload)).encode() + b"\r\n\r\n" + payload)
        except Exception as exc:
            errors.append(exc)
        finally:
            listener.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    assert Client(f"http://127.0.0.1:{port}").request("score", task_id="train:x", events=[]) == {"solve_f1": 0.5}
    thread.join(timeout=6)
    assert not thread.is_alive() and not errors
    assert len(received) == 2 and received[0] == received[1]
    assert all(x["operation"] == "score" for x in received)
