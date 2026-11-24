"""Illustrative Entrez-keyed bipartite graph of selected Stage 12 brain associations."""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from scripts.biology.functional_enrichment import entrez

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "outputs/tables/biology"
FIGURES = ROOT / "outputs/figures/biology"
CANDIDATES = ["ZFAS1", "LY86-AS1", "PCED1B-AS1", "GIHCG", "SMARCA5-AS1",
              "OIP5-AS1", "PEG13", "ANXA2R-AS1", "PSMD6-AS2"]


def build_candidate_network(edges, candidates, k=3):
    """Select up to k stable associations per Entrez ID and sign, preserving upstream ties."""
    frame = edges.loc[edges.lnc_symbol.isin(candidates)].copy()
    for col in ["lnc_entrez", "coding_entrez"]:
        frame[col] = frame[col].map(entrez)
    if frame.duplicated(["lnc_entrez", "coding_entrez"]).any():
        raise ValueError("Duplicate edge")
    if set(frame.lnc_entrez) & set(frame.coding_entrez):
        raise ValueError("Conflicting node types")
    for role in ["lnc", "coding"]:
        if frame.groupby(role + "_entrez")[role + "_symbol"].nunique().gt(1).any():
            raise ValueError("Conflicting symbols")
    keep = pd.Series(True, index=frame.index)
    for model in ["primary", "sensitivity"]:
        r, q = frame[model + "_r"], frame[model + "_q"]
        if not (r.between(-1, 1) & q.between(0, 1)).all():
            raise ValueError("Invalid r or q")
        keep &= q.lt(.05) & ((r.gt(0) & frame.direction.eq("positive")) |
                             (r.lt(0) & frame.direction.eq("negative")))
    selected = (frame[keep].sort_values("primary_r", key=abs, ascending=False, kind="stable")
                .groupby(["lnc_entrez", "direction"], sort=False).head(k))
    graph = nx.Graph()
    for row in selected.itertuples():
        graph.add_node(row.lnc_entrez, symbol=row.lnc_symbol, node_type="lncRNA")
        graph.add_node(row.coding_entrez, symbol=row.coding_symbol, node_type="protein_coding")
        graph.add_edge(row.lnc_entrez, row.coding_entrez, weight=abs(row.primary_r),
                       direction=row.direction, primary_r=row.primary_r, sensitivity_r=row.sensitivity_r,
                       primary_q=row.primary_q, sensitivity_q=row.sensitivity_q)
    for i, group in enumerate(sorted(nx.connected_components(graph), key=min), 1):
        for node in group:
            graph.nodes[node].update(component=i, degree=graph.degree(node),
                                     strength=graph.degree(node, weight="weight"))
    return graph, selected


def topology(graph):
    n = len(graph)
    lnc = sum(x["node_type"] == "lncRNA" for _, x in graph.nodes(data=True))
    return dict(nodes=n, edges=graph.size(), lncrnas=lnc, coding=n-lnc,
                components=nx.number_connected_components(graph), density=nx.density(graph),
                bipartite_density=graph.size()/(lnc*(n-lnc)) if lnc and n>lnc else 0,
                average_triangle_clustering=nx.average_clustering(graph) if n else 0,
                max_degree=max(dict(graph.degree()).values(), default=0))


def plot_network(graph):
    groups = sorted(nx.connected_components(graph), key=min)
    fig, axes = plt.subplots(max(1,(len(groups)+1)//2), 2, figsize=(13, 2.6*((len(groups)+1)//2)), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, group in zip(axes.flat, groups):
        sub = graph.subgraph(sorted(group))
        pos = nx.spring_layout(sub, seed=42, weight=None)
        for kind, color, shape in [("lncRNA","#e59558","D"),("protein_coding","#90b9cf","o")]:
            nx.draw_networkx_nodes(sub,pos,nodelist=[n for n in sub if sub.nodes[n]["node_type"]==kind],
                                   node_color=color,node_shape=shape,node_size=240,ax=ax)
        for sign, color, style in [("positive","#b03b42","solid"),("negative","#365ca5","dashed")]:
            nx.draw_networkx_edges(sub,pos,edgelist=[(u,v) for u,v,d in sub.edges(data=True) if d["direction"]==sign],
                                   edge_color=color,style=style,ax=ax)
        nx.draw_networkx_labels(sub,pos,labels=nx.get_node_attributes(sub,"symbol"),font_size=8,ax=ax)
        ax.margins(.22)
    fig.suptitle("GSE125583 statistical associations | diamonds show lncRNAs\nSolid red: positive   Dashed blue: negative   Panels: connected components")
    fig.tight_layout(rect=(0,0,1,.94))
    fig.savefig(FIGURES/"coexpression_network.png",dpi=180)
    plt.close(fig)


def main():
    source = Path(os.environ.get("LNCSIGNAL_INPUT_ROOT", ROOT)) / "outputs/tables/stage12/edge_summary.csv"
    graph, edges = build_candidate_network(pd.read_csv(source, dtype={"lnc_entrez":str,"coding_entrez":str}), CANDIDATES)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    nodes = pd.DataFrame([dict(entrez_id=n, **d) for n,d in graph.nodes(data=True)])
    brain = Path(os.environ.get("LNCSIGNAL_INPUT_ROOT", ROOT)) / "outputs/tables/stage11/GSE125583_differential_expression.csv"
    de = pd.read_csv(brain, dtype={"feature_id":str}, usecols=["feature_id","effect","fdr"]).rename(
        columns={"feature_id":"entrez_id","effect":"brain_log2_fold_change","fdr":"brain_de_q"})
    nodes = nodes.merge(de, on="entrez_id", how="left", validate="one_to_one")
    nodes["closeness_unweighted"] = nodes.entrez_id.map(nx.closeness_centrality(graph))
    nodes["betweenness_unweighted"] = nodes.entrez_id.map(nx.betweenness_centrality(graph,weight=None))
    nodes.to_csv(TABLES/"network_nodes.csv",index=False)
    edges.to_csv(TABLES/"network_edges_biology.csv",index=False)
    (TABLES/"network_topology_metrics.json").write_text(json.dumps(topology(graph), indent=2)+"\n")
    plot_network(graph)
    data = pd.read_csv(TABLES/"top_candidate_pathway_enrichments.csv")
    data = data.sort_values("q_global").drop_duplicates("lnc_symbol").sort_values("q_global",ascending=False)
    fig, ax = plt.subplots(figsize=(12,5))
    ax.barh([f"{r.lnc_symbol} ({r.direction}) | {r.database}: {r.term_name}" for r in data.itertuples()],
            -np.log10(data.q_global),color="#387088")
    ax.set_xlabel("Strongest pathway association per candidate, -log10(database-wide BH q)")
    fig.tight_layout()
    fig.savefig(FIGURES/"pathway_enrichment.png",dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
