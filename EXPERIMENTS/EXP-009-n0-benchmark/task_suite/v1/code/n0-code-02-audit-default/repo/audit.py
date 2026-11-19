"""Event log helpers."""


def add_event(event, events=[]):
    events.append(event)
    return events


class EventLog:
    def __init__(self):
        self._events = []

    def add(self, event):
        return add_event(event, self._events)

    def snapshot(self):
        return list(self._events)
