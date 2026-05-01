import util.tokenizer as tokenizer

def test_tokenize_de_complex_sentence_exact_tokens():
    tok = tokenizer.Tokenizer()
    text = (
        "Obwohl der Ingenieur — der gestern (um 23:45 Uhr) aus München "
        "zurückgekehrt ist — die Spezifikation bereits sorgfältig geprüft "
        "hatte, sagte er: \"Wir müssen Kapitel 3.1-3.3 noch einmal "
        "überarbeiten; sonst riskieren wir Fehler ... oder?\""
    )

    out = tok.tokenize_de(text)

    expected = [
        "Obwohl", "der", "Ingenieur", "—", "der", "gestern", "(", "um", "23:45",
        "Uhr", ")", "aus", "München", "zurückgekehrt", "ist", "—", "die",
        "Spezifikation", "bereits", "sorgfältig", "geprüft", "hatte", ",",
        "sagte", "er", ":", "\"", "Wir", "müssen", "Kapitel", "3.1", "-", "3.3",
        "noch", "einmal", "überarbeiten", ";", "sonst", "riskieren", "wir",
        "Fehler", "...", "oder", "?", "\""
    ]
    assert out == expected


def test_tokenize_en_complex_sentence_exact_tokens():
    tok = tokenizer.Tokenizer()
    text = (
        "Although the engineer—who'd returned from Munich yesterday (at 11:45 "
        "p.m.)—had already reviewed the 27-page specification, he said: "
        "\"We need to re-check sections 3.1-3.3; otherwise, we'll ship bugs "
        "... don't we?\""
    )

    out = tok.tokenize_en(text)

    expected = [
        "Although", "the", "engineer", "—", "who'd", "returned", "from",
        "Munich", "yesterday", "(", "at", "11:45", "p.m.)—had", "already",
        "reviewed", "the", "27", "-", "page", "specification", ",", "he",
        "said", ":", "\"", "We", "need", "to", "re", "-", "check", "sections",
        "3.1", "-", "3.3", ";", "otherwise", ",", "we", "'ll", "ship", "bugs",
        "...", "do", "n't", "we", "?", "\""
    ]
    assert out == expected
