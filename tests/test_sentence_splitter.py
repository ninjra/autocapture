from autocapture.answer.coverage import split_sentences


def test_sentence_splitter_respects_abbrev() -> None:
    text = "Dr. Smith went home. It was late."
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert sentences[0].startswith("Dr.")
    assert sentences[1].startswith("It was")


def test_sentence_splitter_handles_decimals() -> None:
    text = "Version 3.1 is out. Update soon."
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert sentences[0].endswith("out.")
    assert sentences[1].startswith("Update")
