"""Append-only buffers."""


def push(buffer=None, item="x"):
    if buffer is None:
        buffer = []
    buffer.append(item)
    return buffer
