"""Append-only buffers."""


def push(buffer=[], item="x"):
    buffer.append(item)
    return buffer
