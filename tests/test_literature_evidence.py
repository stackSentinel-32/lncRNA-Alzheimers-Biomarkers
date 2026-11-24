import requests

from scripts.biology.literature_evidence import lookup, matches


def test_metadata_requires_both_identifiers_and_titles():
    p = dict(pmid="123",doi="10.1/A",title="RNA in cells.")
    c = dict(doi="10.1/a",title="<i>RNA</i> in cells")
    assert matches("123","10.1/a","RNA in cells",p,c)
    assert not matches("123","10.1/a","RNA in brain",p,c)
    assert not matches("999","10.1/a","RNA in cells",p,c)
    assert not matches("123","10.1/b","RNA in cells",p,c)
    assert not matches("123","10.1/a","RNA in cells",p,{})


def test_failed_lookup_cannot_verify(monkeypatch):
    def fail(*args, **kwargs):
        raise requests.Timeout()
    monkeypatch.setattr(requests,"get",fail)
    data, status = lookup("https://example.invalid")
    assert not data and "unverified" in status
    assert not matches("1","example","title",data,data)
