import report


def test_same_user_different_locales():
    assert report.build_report(7, "en") == "Hello, user 7"
    assert report.build_report(7, "ar") == "مرحبا, user 7"
    assert report.build_report(7, "fr") == "Bonjour, user 7"


def test_repeated_call_is_stable():
    first = report.build_report(9, "en")
    second = report.build_report(9, "en")
    assert first == second == "Hello, user 9"


def test_unknown_locale_falls_back_to_english():
    assert report.build_report(3, "de") == "Hello, user 3"
    assert report.build_report(7, "de") == "Hello, user 7"
