from ai4gcnpy.utils import split_text_into_paragraphs, group_paragraphs_by_labels, header_regex_match


def test_extra_whitespace_between_paragraphs():
    raw = "  A  \n\n  \n\n  B  "
    expected = ["A", "B"]
    assert split_text_into_paragraphs(raw) == expected


def test_group_paragraphs_by_labels():
    paragraphs = ["Intro", "Methods", "Results"]
    tags = ["A", "B", "A"]
    expected = {
        "A": "Intro\n\nResults",
        "B": "Methods"
    }
    assert group_paragraphs_by_labels(paragraphs, tags) == expected


def test_header_regex_match():
    header = """
        TITLE:   GCN CIRCULAR
        NUMBER:  12345
        SUBJECT: GRB110915A  MITSuME Okayama J-band upper-limit
        DATE:    11/09/16 04:24:57 GMT
        FROM:    Daisuke Kuroda at OAO/NAOJ  <dikuroda@oao.nao.ac.jp>
    """

    result = header_regex_match(header)
    assert result == {
        "circularId": "12345",
        "subject": "GRB110915A  MITSuME Okayama J-band upper-limit",
        "createdOn": "11/09/16 04:24:57 GMT",
        "submitter": "Daisuke Kuroda at OAO/NAOJ",
        "email": "dikuroda@oao.nao.ac.jp"
    }