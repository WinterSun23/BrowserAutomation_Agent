from sitegraph.embed import HashEmbedder, cosine


def test_hash_embedder_shape_and_similarity():
    e = HashEmbedder(384)
    a, b, c = e.embed(["open shopping cart", "open the cart", "weather in paris"])
    assert len(a) == 384
    assert abs(cosine(a, a) - 1.0) < 1e-6
    assert cosine(a, b) > cosine(a, c)


def test_empty_input():
    assert HashEmbedder().embed([]) == []
