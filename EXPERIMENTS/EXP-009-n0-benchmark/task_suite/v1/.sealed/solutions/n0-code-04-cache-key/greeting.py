GREETINGS = {
    "en": "Hello",
    "ar": "مرحبا",
    "fr": "Bonjour",
}


def greeting(locale):
    return GREETINGS.get(locale, GREETINGS["en"])
