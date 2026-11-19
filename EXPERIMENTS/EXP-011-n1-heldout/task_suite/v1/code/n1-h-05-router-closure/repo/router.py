"""Build route callables."""


def make_routes(table):
    routes = {}
    for path, handler_id in table.items():
        routes[path] = lambda: handler_id
    return routes
