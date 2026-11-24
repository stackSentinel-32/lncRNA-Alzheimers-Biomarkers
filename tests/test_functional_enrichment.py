"""Unit tests for Stage 13 Functional Enrichment algorithms, background universe checks, and FDR correction."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

from scripts.biology.functional_enrichment import (
    bh_adjust,
    run_hypergeometric_enrichment,
    TABLES_DIR,
)


def test_bh_adjust_correctness_and_nan_handling():
    """Verify that bh_adjust matches statsmodels multipletests and properly handles NaNs."""
    p = np.array([0.001, 0.01, 0.04, 0.05, 0.50, np.nan, 0.90])
    q = bh_adjust(p)

    valid = np.isfinite(p)
    expected_q = multipletests(p[valid], method="fdr_bh")[1]

    np.testing.assert_allclose(q[valid], expected_q)
    assert np.isnan(q[~valid]).all()
    assert (q[valid] >= 0.0).all() and (q[valid] <= 1.0).all()


def test_hypergeometric_calculation_exactness():
    """Verify hypergeometric test against scipy hypergeom.sf."""
    # Five genes, two drawn, both in a two-gene term: 1 / C(5,2).
    fg = pd.DataFrame({"lnc_entrez":["10"]*3, "direction":["positive"]*3,
                       "coding_entrez":["01","2","2.0"]})
    terms = {"A":{"name":"hit","genes":{"1","2","6"}}, "B":{"name":"miss","genes":{"3","4"}}}
    all_tests, _, _ = run_hypergeometric_enrichment(fg, set("12345"), {}, {}, terms, "test",
                                                      min_term_size=1, min_foreground_size=1)
    assert all_tests.term_size.tolist() == [2,2]
    np.testing.assert_allclose(all_tests.p_value,[.1,1])
    np.testing.assert_allclose(all_tests.q_global,[.2,1])
    np.testing.assert_allclose(all_tests.fold_enrichment,[2.5,0])
    empty, sig, _ = run_hypergeometric_enrichment(fg, set("12345"), {}, {}, {}, "empty")
    assert empty.empty and sig.empty


def test_foreground_outside_background_raises_error():
    """Verify that any foreground containing genes outside the background universe raises ValueError."""
    background = {"10", "20", "30", "40", "50"}
    symbol_map = {g: f"SYM_{g}" for g in background}
    lnc_symbols = {"100": "LNC_TEST"}

    # Invalid foreground contains '999' not in background
    invalid_fg = pd.DataFrame([
        {"lnc_entrez": "100", "direction": "positive", "coding_entrez": "10"},
        {"lnc_entrez": "100", "direction": "positive", "coding_entrez": "999"},
    ])

    dummy_terms = {"TERM1": {"name": "Test Term", "genes": {"10", "20"}}}

    with pytest.raises(ValueError, match="outside measurable universe"):
        run_hypergeometric_enrichment(
            invalid_fg, background, symbol_map, lnc_symbols, dummy_terms, "TestDB", min_term_size=1, min_foreground_size=1
        )


def test_term_size_filtering():
    """Verify terms smaller than min_term_size or larger than max_term_size are excluded."""
    background = {str(i) for i in range(1, 101)}
    symbol_map = {g: f"S_{g}" for g in background}
    lnc_symbols = {"101": "LNC1"}

    fg = pd.DataFrame([{"lnc_entrez": "101", "direction": "positive", "coding_entrez": str(i)} for i in range(1, 15)])

    terms = {
        "TOO_SMALL": {"name": "Small", "genes": {"1", "2"}},                   # size 2 < 5
        "ELIGIBLE": {"name": "Eligible", "genes": {str(i) for i in range(1, 11)}}, # size 10 (5 <= 10 <= 20)
        "TOO_LARGE": {"name": "Large", "genes": {str(i) for i in range(1, 51)}},   # size 50 > 20
    }

    all_tests, _, summary = run_hypergeometric_enrichment(
        fg, background, symbol_map, lnc_symbols, terms, "TestDB", min_term_size=5, max_term_size=20, min_foreground_size=5
    )

    assert summary["eligible_terms"] == 1
    assert all_tests["term_id"].tolist() == ["ELIGIBLE"]




def test_annotation_identity_and_invalid_rows():
    from scripts.biology.functional_enrichment import entrez, load_snapshot
    for invalid in [None, "nan", 0, -1, 1.7]:
        with pytest.raises(ValueError):
            entrez(invalid)
    links = pd.DataFrame({"gene_id":["01","1","2",None,"x"],"pathway_id":["T1","T1","T2","T1","T1"]})
    names = pd.DataFrame({"pathway_id":["T1"],"pathway_name":["term"]})
    assert load_snapshot(links,names,r"T[0-9]") == {"T1":{"name":"term","genes":{"1"}}}
    with pytest.raises(ValueError,match="Conflicting"):
        load_snapshot(links,pd.concat([names,names.assign(pathway_name="other")]),r"T[0-9]")
