"""Failure taxonomy for the client."""


class RecoverableError(Exception):
    """Worth another attempt."""


class BadRequest(Exception):
    """Never retry."""
