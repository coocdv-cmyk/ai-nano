"""Error types used by the retry helper."""


class TransientError(Exception):
    """A failure that is worth retrying."""


class FatalError(Exception):
    """A failure that must never be retried."""
