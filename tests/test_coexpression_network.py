"""Unit tests for Stage 13 NetworkX co-expression network construction, attributes, and topology metrics."""
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from scripts.biology.coexpression_network import (
    build_candidate_network,
    TABLES_DIR,
    FIG_DIR,
)


def test_build_candidate_network_synthetic():
    """Verify NetworkX graph construction, node typing, and edge attributes on synthetic inputs."""
    # Create synthetic edge summary
    synthetic_edges = pd.DataFrame([
        {
            "lnc_entrez": "1001",
            "lnc_symbol": "ZFAS1",
            "coding_entrez": "2001",
            "coding_symbol": "RPL34",
            "primary_r": 0.92,
            "sensitivity_r": 0.91,
            "primary_q": 1e-50,
            "direction": "positive",
        },
        {
            "lnc_entrez": "1001",
            "lnc_symbol": "ZFAS1",
            "coding_entrez": "2002",
            "coding_symbol": "RPL36A",
            "primary_r": 0.89,
            "sensitivity_r": 0.88,
            "primary_q": 1e-45,
            "direction": "positive",
        },
        {
            "lnc_entrez": "1002",
            "lnc_symbol": "LY86-AS1",
            "coding_entrez": "2003",
            "coding_symbol": "SYN2",
            "primary_r": 0.93,
            "sensitivity_r": 0.92,
            "primary_q": 1e-60,
            "direction": "positive",
        },
    ])

    synthetic_de = {
        "1001": {"symbol": "ZFAS1", "log2fc": -0.05, "fdr": 0.54, "de_status": "not_significant"},
        "1002": {"symbol": "LY86-AS1", "log2fc": -0.42, "fdr": 0.002, "de_status": "AD_down"},
        "2001": {"symbol": "RPL34", "log2fc": 0.12, "fdr": 0.04, "de_status": "AD_up"},
        "2002": {"symbol": "RPL36A", "log2fc": 0.08, "fdr": 0.15, "de_status": "not_significant"},
        "2003": {"symbol": "SYN2", "log2fc": -0.65, "fdr": 0.0001, "de_status": "AD_down"},
    }

    G, nodes_df, edges_df = build_candidate_network(
        synthetic_edges, synthetic_de, candidate_symbols=["ZFAS1", "LY86-AS1"], max_partners_per_lnc=5
    )

    # Check graph size
    assert G.number_of_nodes() == 5
    assert G.number_of_edges() == 3

    # Check node attributes
    assert G.nodes["ZFAS1"]["node_type"] == "lncRNA"
    assert G.nodes["RPL34"]["node_type"] == "protein_coding"
    assert G.nodes["ZFAS1"]["module"] == "Translation / Ribosome"
    assert G.nodes["LY86-AS1"]["module"] == "Synaptic Transmission"

    # Check edge attributes
    assert G["ZFAS1"]["RPL34"]["primary_r"] == pytest.approx(0.92)
    assert G["ZFAS1"]["RPL34"]["direction"] == "positive"
    assert G["ZFAS1"]["RPL34"]["weight"] == pytest.approx(0.92)

    # Check topological metrics in dataframe
    assert "degree" in nodes_df.columns
    assert "betweenness_centrality" in nodes_df.columns
    assert nodes_df.loc[nodes_df["symbol"] == "ZFAS1", "degree"].values[0] == 2
    assert nodes_df.loc[nodes_df["symbol"] == "LY86-AS1", "degree"].values[0] == 1


def test_real_network_files_and_figures_exist():
    """Verify that network nodes, edges, topology metrics, and figures were generated."""
    nodes_path = TABLES_DIR / "network_nodes.csv"
    edges_path = TABLES_DIR / "network_edges_biology.csv"
    metrics_path = TABLES_DIR / "network_topology_metrics.json"

    assert nodes_path.exists(), "Missing network_nodes.csv"
    assert edges_path.exists(), "Missing network_edges_biology.csv"
    assert metrics_path.exists(), "Missing network_topology_metrics.json"

    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)

    assert len(nodes_df) >= 30, "Too few nodes in network"
    assert len(edges_df) >= 20, "Too few edges in network"
    assert set(nodes_df["node_type"]) == {"lncRNA", "protein_coding"}

    # Check figures exist and are non-empty
    fig1 = FIG_DIR / "coexpression_network_modules.png"
    fig2 = FIG_DIR / "subgraph_top_candidates.png"
    fig3 = FIG_DIR / "pathway_enrichment_dotplot.png"

    assert fig1.exists() and fig1.stat().st_size > 50000, "Figure 1 missing or too small"
    assert fig2.exists() and fig2.stat().st_size > 50000, "Figure 2 missing or too small"
    assert fig3.exists() and fig3.stat().st_size > 50000, "Figure 3 missing or too small"
