"""Opt-in Stage 13 reconstruction from read-only GSE125583 association statistics."""
import json
import os
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from scipy.stats import false_discovery_control

from scripts.biology.coexpression_network import TABLES, topology

pytestmark = pytest.mark.skipif("STAGE13_VERIFY_ROOT" not in os.environ, reason="requires Stage 12 brain statistics")


def test_reconstruct_foregrounds_and_display():
    base = Path(os.environ["STAGE13_VERIFY_ROOT"]) / "outputs/tables/stage12"
    fg = pd.read_csv(base / "enrichment_foreground.csv.gz")
    edges = pd.read_csv(TABLES / "network_edges_biology.csv")
    candidates = set(edges.lnc_entrez)
    columns = ["lnc_entrez", "coding_entrez", "primary_r", "primary_q", "sensitivity_r", "sensitivity_q"]
    parts, display = [], []
    for batch in pq.ParquetFile(base/"all_pairs.parquet").iter_batches(batch_size=150000,columns=columns):
        frame = batch.to_pandas()
        stable = frame[(frame.primary_q < .05) & (frame.sensitivity_q < .05)
                       & (frame.primary_r * frame.sensitivity_r > 0)]
        strong = stable[(stable.primary_r.abs() >= .75) & (stable.sensitivity_r.abs() >= .75)]
        parts.append(strong.assign(direction=np.where(strong.primary_r>0,"positive","negative"))[fg.columns])
        candidate = stable[stable.lnc_entrez.isin(candidates)]
        display.append(candidate.assign(direction=np.where(candidate.primary_r>0,"positive","negative")))
    def ordered(frame, cols):
        return frame.sort_values(cols).reset_index(drop=True)
    pd.testing.assert_frame_equal(ordered(fg,list(fg)),ordered(pd.concat(parts),list(fg)))
    top = (pd.concat(display).sort_values("primary_r",key=abs,ascending=False,kind="stable")
           .groupby(["lnc_entrez","direction"]).head(3))
    keys = ["lnc_entrez","coding_entrez"]
    pd.testing.assert_frame_equal(ordered(top[columns],keys),ordered(edges[columns],keys),atol=0,rtol=1e-13)
    graph = nx.from_pandas_edgelist(edges,"lnc_entrez","coding_entrez",edge_attr="primary_r")
    nodes = pd.read_csv(TABLES/"network_nodes.csv").set_index("entrez_id")
    nx.set_node_attributes(graph,nodes.node_type.to_dict(),"node_type")
    assert topology(graph) == json.loads((TABLES/"network_topology_metrics.json").read_text())
    for group in nx.connected_components(graph):
        assert set(nodes.index[nodes.component == nodes.loc[next(iter(group)),"component"]]) == group
    for node in graph:
        assert nodes.loc[node,"degree"] == graph.degree(node)
        assert nodes.loc[node,"strength"] == pytest.approx(sum(abs(d["primary_r"]) for d in graph[node].values()))


@pytest.mark.parametrize("database", ["Reactome","KEGG","GO"])
def test_global_fdr_and_serialized_hits(database):
    if database == "GO":
        base = Path(os.environ["STAGE13_VERIFY_ROOT"]) / "outputs/tables/stage12"
        full = pd.read_parquet(base/"GO_BP_all_tests.parquet")
        sig = pd.read_csv(base/"GO_BP_significant.csv.gz")
        p,q,term = "p","q","go_id"
    else:
        full = pd.read_parquet(TABLES/f"{database.lower()}_all_tests.parquet")
        sig = pd.read_csv(TABLES/f"{database.lower()}_significant.csv.gz")
        p,q,term = "p_value","q_global","term_id"
    np.testing.assert_allclose(full[q],false_discovery_control(full[p]),rtol=1e-12,atol=0)
    keys = [term,"direction","lnc_entrez"]
    expected = full[full[q]<.05].copy()
    expected["lnc_entrez"] = expected.lnc_entrez.astype(int)
    for col in [p,q,"fold_enrichment","overlap"]:
        np.testing.assert_allclose(expected.sort_values(keys)[col],sig.sort_values(keys)[col],rtol=1e-12,atol=0)
    assert expected.set_index(keys).sort_index().index.equals(sig.set_index(keys).sort_index().index)
