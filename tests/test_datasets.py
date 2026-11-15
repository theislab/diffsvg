from diffsvg import datasets


def test_registry_pins_legnini23():
    """The example dataset is registered with a CDN key and a sha256, so no network is needed to check it."""
    entry = datasets._DATASETS["legnini23"]
    (file,) = entry.files
    assert file.resolve_url(datasets._BASE_URL) == "https://exampledata.scverse.org/diffsvg/legnini23.h5ad"
    assert len(file.sha256) == 64
    assert file.fallback_urls
