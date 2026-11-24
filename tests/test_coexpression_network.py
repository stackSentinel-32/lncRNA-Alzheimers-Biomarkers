import io

import networkx as nx
import pandas as pd
import pytest

from scripts.biology.coexpression_network import build_candidate_network, topology


def edges():
    return pd.DataFrame([
        (1,"A",3,"S",.6,.55,.01,.01,"positive"),
        (1,"A",4,"S",.4,.45,.01,.01,"positive"),
        (1,"A",5,"T",-.8,-.79,.01,.01,"negative"),
        (2,"B",5,"T",.7,.72,.01,.01,"positive"),
        (2,"B",6,"V",-.9,.8,.01,.01,"negative"),
        (2,"B",7,"W",.98,.97,.01,.05,"positive"),
        (8,"C",9,"X",.5,.6,.01,.01,"positive"),
    ], columns=["lnc_entrez","lnc_symbol","coding_entrez","coding_symbol","primary_r",
                "sensitivity_r","primary_q","sensitivity_q","direction"])


def test_graph_selection_and_serialization():
    graph, selected = build_candidate_network(edges(), ["A","B","C"], k=1)
    assert set(graph) == {"1","2","3","5","8","9"}
    assert graph.nodes["1"]["node_type"] == "lncRNA"
    assert graph.nodes["1"]["strength"] == pytest.approx(1.4)
    assert graph.nodes["1"]["degree"] == 2
    assert graph["1"]["5"]["direction"] == "negative"
    assert topology(graph) == dict(nodes=6,edges=4,lncrnas=3,coding=3,components=2,
        density=4/15,bipartite_density=4/9,average_triangle_clustering=0,max_degree=2)
    saved = pd.read_csv(io.StringIO(selected.to_csv(index=False)),dtype={"lnc_entrez":str,"coding_entrez":str})
    rebuilt = nx.from_pandas_edgelist(saved,"lnc_entrez","coding_entrez")
    assert nx.utils.graphs_equal(nx.Graph(graph.edges), nx.Graph(rebuilt.edges))
    graph, _ = build_candidate_network(edges(), ["A"], k=2)
    assert graph.has_node("3") and graph.has_node("4")


def test_duplicate_pair_rejected():
    with pytest.raises(ValueError,match="Duplicate"):
        build_candidate_network(pd.concat([edges(),edges()]), ["A"])
