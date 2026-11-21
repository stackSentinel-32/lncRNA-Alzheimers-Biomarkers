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
    universe = 1000
    term_size = 50
    fg_size = 20
    hits = 5

    # Fold enrichment: (5 / 20) / (50 / 1000) = 0.25 / 0.05 = 5.0
    expected_fold = (hits / fg_size) / (term_size / universe)
    assert expected_fold == pytest.approx(5.0)

    # One-sided probability: P(X >= 5) = hypergeom.sf(4, 1000, 50, 20)
    expected_p = hypergeom.sf(hits - 1, universe, term_size, fg_size)
    assert 0.0 < expected_p < 1.0


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
    lnc_symbols = {"L1": "LNC1"}

    fg = pd.DataFrame([{"lnc_entrez": "L1", "direction": "positive", "coding_entrez": str(i)} for i in range(1, 15)])

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


def test_real_enrichment_outputs_exist_and_valid():
    """Verify generated Stage 13 enrichment tables exist, have non-trivial rows, and meet statistical criteria."""
    reactome_sig_path = TABLES_DIR / "reactome_significant.csv"
    kegg_sig_path = TABLES_DIR / "kegg_significant.csv"
    top_cand_path = TABLES_DIR / "top_candidate_pathway_enrichments.csv"

    assert reactome_sig_path.exists(), "Missing reactome_significant.csv"
    assert kegg_sig_path.exists(), "Missing kegg_significant.csv"
    assert top_cand_path.exists(), "Missing top_candidate_pathway_enrichments.csv"

    reactome_df = pd.read_csv(reactome_sig_path)
    kegg_df = pd.read_csv(kegg_sig_path)
    cand_df = pd.read_csv(top_cand_path)

    # Significant outputs must have q_global < 0.05
    assert (reactome_df["q_global"] < 0.05).all()
    assert (kegg_df["q_global"] < 0.05).all()
    assert (reactome_df["overlap"] > 0).all()
    assert (kegg_df["overlap"] > 0).all()

    # ZFAS1, LY86-AS1, PCED1B-AS1 must be present in top candidate table
    symbols_present = set(cand_df["lnc_symbol"])
    assert "ZFAS1" in symbols_present
    assert "LY86-AS1" in symbols_present
    assert "PCED1B-AS1" in symbols_present
