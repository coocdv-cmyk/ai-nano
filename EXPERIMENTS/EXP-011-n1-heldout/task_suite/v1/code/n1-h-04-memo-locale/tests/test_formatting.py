import formatting


def test_template_is_part_of_the_cache_key():
    assert formatting.render("Ada", "Hello {name}") == "Hello Ada"
    assert formatting.render("Ada", "Bye {name}") == "Bye Ada"
    assert formatting.render("Ada", "Hi {name}") == "Hi Ada"


def test_repeated_call_is_stable():
    first = formatting.render("Lin", "Hey {name}")
    second = formatting.render("Lin", "Hey {name}")
    assert first == second == "Hey Lin"
