import uuid

import pytest
import zmq

from erlab.interactive.imagetool.manager._server import _recv_multipart, _send_multipart


@pytest.fixture
def zmq_pair():
    ctx = zmq.Context.instance()
    endpoint = f"inproc://multipart-{uuid.uuid4()}"
    sender = ctx.socket(zmq.PAIR)
    receiver = ctx.socket(zmq.PAIR)
    sender.bind(endpoint)
    receiver.connect(endpoint)
    yield sender, receiver
    sender.close(0)
    receiver.close(0)


class _FakeSocket:
    def __init__(self, frames):
        self._frames = frames

    def recv_multipart(self, copy=False, **kwargs):
        return self._frames


def test_send_recv_roundtrip_small(zmq_pair):
    sender, receiver = zmq_pair
    payload = {"packet_type": "command", "command": "ping"}

    _send_multipart(sender, payload)
    result = _recv_multipart(receiver)

    assert result["packet_type"] == "command"
    assert result["command"] == "ping"
    assert result["pickler_kind"] == "pickle"


def test_chunked_message_flag_and_reassembly(zmq_pair):
    sender, receiver = zmq_pair
    large_bytes = b"x" * 1024
    payload = {"packet_type": "add", "data": large_bytes}

    _send_multipart(sender, payload, max_frame_size=128)
    frames = receiver.recv_multipart(copy=False)

    assert frames[1].bytes == b"chunked-v1"

    result = _recv_multipart(_FakeSocket(frames))
    assert result["data"] == large_bytes
    assert result["packet_type"] == "add"
    assert result["pickler_kind"] == "pickle"
