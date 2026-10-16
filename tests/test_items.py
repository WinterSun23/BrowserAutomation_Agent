import json

from sitegraph.browser.items import choose_item, pick_by_meaning, pick_by_text
from sitegraph.llm import FakeLLM, OfflineLLM

CATS = ["Travel", "Mystery", "Historical Fiction", "Sequential Art", "Classics", "Poetry"]


class StubEmbedder:
    """Maps known words to fixed unit vectors so 'meaning' is controllable."""
    DIM = 4
    TABLE = {"trips and holidays": [1, 0, 0, 0], "Travel": [0.98, 0.1, 0, 0], "Mystery": [0, 1, 0, 0],
             "Poetry": [0, 0, 1, 0], "Classics": [0, 0, 0, 1], "vague": [0.5, 0.5, 0.5, 0.5],
             "Historical Fiction": [0, 0, 0.1, 1], "Sequential Art": [0.1, 0, 0, 1]}

    def embed(self, texts):
        return [self.TABLE.get(t, [0.25, 0.25, 0.25, 0.25]) for t in texts]


def test_rule1_item_text_in_request_with_whole_words_and_longest_wins():
    assert pick_by_text(CATS, "open the travel category") == "Travel"
    assert pick_by_text(CATS + ["Fiction"], "show me historical fiction") == "Historical Fiction"   # longest
    assert pick_by_text(["Art"], "open the party page") is None                                      # not inside a word
    assert pick_by_text(["Go"], "go home") is None                                                    # too short


def test_rule2_meaning_needs_threshold_and_margin():
    emb = StubEmbedder()
    assert pick_by_meaning(CATS, "trips and holidays", emb, 0.6, 0.05) == "Travel"
    assert pick_by_meaning(CATS, "vague", emb, 0.6, 0.05) is None                                     # nothing clearly close


def test_rule3_llm_picks_index_but_not_when_offline_or_garbage():
    llm = FakeLLM(lambda s, u: json.dumps({"index": 4}))
    assert choose_item(CATS, "the one about old books", None, llm)[0] == "Classics"
    assert choose_item(CATS, "x", None, OfflineLLM()) == (None, "first")
    assert choose_item(CATS, "x", None, FakeLLM(lambda s, u: "no idea")) == (None, "first")
    assert choose_item(CATS, "x", None, FakeLLM(lambda s, u: json.dumps({"index": 99}))) == (None, "first")


def test_order_text_beats_meaning_beats_llm_and_first_is_the_fallback():
    emb = StubEmbedder()
    llm = FakeLLM(lambda s, u: json.dumps({"index": 1}))
    assert choose_item(CATS, "open the poetry section", emb, llm) == ("Poetry", "text")
    assert choose_item(CATS, "trips and holidays", emb, llm) == ("Travel", "meaning")
    assert choose_item(CATS, "vague", emb, llm) == ("Mystery", "llm")
    assert choose_item(CATS, "vague", emb, None) == (None, "first")
    assert choose_item(["Same", "Same"], "anything", emb, llm) == (None, "first")    # nothing to choose between
    assert choose_item(CATS, "   ", emb, llm) == (None, "first")
