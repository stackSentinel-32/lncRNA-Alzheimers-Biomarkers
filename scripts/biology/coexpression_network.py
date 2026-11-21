"""Stage 13: Co-expression Network Construction and Visualization using NetworkX.

Constructs multi-lncRNA, multi-module biological networks from audited brain co-expression
evidence, computes graph topological metrics (centrality, degree, clustering),
and generates publication-quality network visualizations with interpretable node and edge attributes.
"""
from collections import defaultdict
import json
import logging
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = ROOT / "outputs/tables/biology"
FIG_DIR = ROOT / "outputs/figures/biology"
STAGE12_DIR = ROOT / "outputs/tables/stage12"
STAGE11_DIR = ROOT / "outputs/tables/stage11"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("coexpression_network")


def load_network_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict]]:
    """Load co-expression edges, background genes, and brain differential expression statistics."""
    # 1. Load Background Universe
    bg_path = STAGE12_DIR / "enrichment_background.csv"
    bg_df = pd.read_csv(bg_path, dtype={"coding_entrez": str})
    symbol_map = dict(zip(bg_df["coding_entrez"], bg_df["symbol"]))

    # 2. Load Brain Differential Expression
    de_path = STAGE11_DIR / "GSE125583_differential_expression.csv"
    de_dict = {}
    if de_path.exists():
        de_df = pd.read_csv(de_path, dtype={"feature_id": str})
        for _, row in de_df.iterrows():
            fid = str(row["feature_id"])
            log2fc = float(row["effect"]) if pd.notna(row["effect"]) else 0.0
            fdr = float(row["fdr"]) if pd.notna(row["fdr"]) else 1.0
            status = "not_significant"
            if fdr < 0.05:
                status = "AD_up" if log2fc > 0 else "AD_down"
            de_dict[fid] = {
                "symbol": str(row["symbol"]) if pd.notna(row["symbol"]) else fid,
                "log2fc": log2fc,
                "fdr": fdr,
                "de_status": status,
            }

    # 3. Load edge summaries from Stage 12
    edge_summary_path = STAGE12_DIR / "edge_summary.csv"
    if not edge_summary_path.exists():
        raise FileNotFoundError(f"Missing edge summary table: {edge_summary_path}")
    edges_df = pd.read_csv(edge_summary_path, dtype={"lnc_entrez": str, "coding_entrez": str})

    # 4. Load lncRNA symbols
    edges_per_lnc_path = STAGE12_DIR / "edges_per_lncrna.csv"
    lnc_symbols = {}
    if edges_per_lnc_path.exists():
        epl_df = pd.read_csv(edges_per_lnc_path, dtype={"lnc_entrez": str})
        lnc_symbols = dict(zip(epl_df["lnc_entrez"], epl_df["symbol"]))

    return edges_df, bg_df, de_df if de_path.exists() else pd.DataFrame(), de_dict


def build_candidate_network(
    edges_df: pd.DataFrame,
    de_dict: dict[str, dict],
    candidate_symbols: list[str],
    max_partners_per_lnc: int = 15,
) -> tuple[nx.Graph, pd.DataFrame, pd.DataFrame]:
    """Build a NetworkX Graph around selected candidate lncRNAs with interpretable node and edge attributes."""
    G = nx.Graph()

    # Module definitions based on verified pathway enrichment
    module_assignments = {
        "ZFAS1": "Translation / Ribosome",
        "LY86-AS1": "Synaptic Transmission",
        "PCED1B-AS1": "Innate Immune / Microglia",
        "GIHCG": "Mitochondrial Respiration",
        "SMARCA5-AS1": "Mitochondrial Respiration",
        "OIP5-AS1": "Ubiquitination / Neuroprotection",
        "PEG13": "Synaptic / Calcium Signaling",
        "ANXA2R-AS1": "Vesicle Trafficking",
        "PSMD6-AS2": "Proteasome Complex",
    }

    # Filter edges to candidates
    cand_edges = edges_df[edges_df["lnc_symbol"].isin(candidate_symbols)].copy()

    # Take top N positive and negative partners per lncRNA by absolute primary_r
    selected_edges = []
    for (lnc_sym, direction), grp in cand_edges.groupby(["lnc_symbol", "direction"]):
        top_grp = grp.sort_values(by="primary_r", key=abs, ascending=False).head(max_partners_per_lnc)
        selected_edges.append(top_grp)

    selected_edges_df = pd.concat(selected_edges, ignore_index=True)

    # Populate nodes and edges into NetworkX
    node_data_list = []
    seen_nodes = set()

    for _, row in selected_edges_df.iterrows():
        lnc_id = str(row["lnc_entrez"])
        lnc_sym = str(row["lnc_symbol"])
        coding_id = str(row["coding_entrez"])
        coding_sym = str(row["coding_symbol"])
        r_val = float(row["primary_r"])
        sens_r = float(row["sensitivity_r"])
        q_val = float(row["primary_q"])
        direction = str(row["direction"])

        # Determine module
        mod = module_assignments.get(lnc_sym, "Exploratory / Unassigned")

        # Add lncRNA node
        if lnc_sym not in seen_nodes:
            seen_nodes.add(lnc_sym)
            de_info = de_dict.get(lnc_id, {"log2fc": 0.0, "fdr": 1.0, "de_status": "not_tested"})
            G.add_node(
                lnc_sym,
                entrez_id=lnc_id,
                node_type="lncRNA",
                symbol=lnc_sym,
                module=mod,
                log2fc=de_info["log2fc"],
                fdr=de_info["fdr"],
                de_status=de_info["de_status"],
            )
            node_data_list.append({
                "node_id": lnc_sym,
                "entrez_id": lnc_id,
                "symbol": lnc_sym,
                "node_type": "lncRNA",
                "module": mod,
                "brain_log2fc": de_info["log2fc"],
                "brain_fdr": de_info["fdr"],
                "brain_de_status": de_info["de_status"],
            })

        # Add coding mRNA node
        if coding_sym not in seen_nodes:
            seen_nodes.add(coding_sym)
            de_info = de_dict.get(coding_id, {"log2fc": 0.0, "fdr": 1.0, "de_status": "not_tested"})
            G.add_node(
                coding_sym,
                entrez_id=coding_id,
                node_type="protein_coding",
                symbol=coding_sym,
                module=mod,
                log2fc=de_info["log2fc"],
                fdr=de_info["fdr"],
                de_status=de_info["de_status"],
            )
            node_data_list.append({
                "node_id": coding_sym,
                "entrez_id": coding_id,
                "symbol": coding_sym,
                "node_type": "protein_coding",
                "module": mod,
                "brain_log2fc": de_info["log2fc"],
                "brain_fdr": de_info["fdr"],
                "brain_de_status": de_info["de_status"],
            })

        # Add edge
        G.add_edge(
            lnc_sym,
            coding_sym,
            primary_r=r_val,
            sensitivity_r=sens_r,
            q_val=q_val,
            direction=direction,
            weight=abs(r_val),
        )

    # Compute graph topological metrics
    degree_dict = dict(G.degree())
    strength_dict = dict(G.degree(weight="weight"))
    betweenness_dict = nx.betweenness_centrality(G, weight="weight")
    closeness_dict = nx.closeness_centrality(G)
    clustering_dict = nx.clustering(G, weight="weight")

    nodes_df = pd.DataFrame(node_data_list)
    nodes_df["degree"] = nodes_df["node_id"].map(degree_dict)
    nodes_df["strength"] = nodes_df["node_id"].map(strength_dict)
    nodes_df["betweenness_centrality"] = nodes_df["node_id"].map(betweenness_dict)
    nodes_df["closeness_centrality"] = nodes_df["node_id"].map(closeness_dict)
    nodes_df["clustering_coefficient"] = nodes_df["node_id"].map(clustering_dict)

    # Update G nodes with topological metrics
    for _, row in nodes_df.iterrows():
        G.nodes[row["node_id"]]["degree"] = row["degree"]
        G.nodes[row["node_id"]]["strength"] = row["strength"]
        G.nodes[row["node_id"]]["betweenness"] = row["betweenness_centrality"]

    return G, nodes_df, selected_edges_df


def plot_modular_network(G: nx.Graph, nodes_df: pd.DataFrame, out_path: Path):
    """Render publication-quality network visualization of biological modules."""
    plt.figure(figsize=(16, 12), dpi=300)

    # Group positions by module using layout
    pos = nx.spring_layout(G, k=0.45, seed=42, iterations=80)

    # Visual encodings:
    # lncRNAs = diamonds, mRNAs = circles
    lnc_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "lncRNA"]
    mrna_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "protein_coding"]

    # Color map for DE status
    de_colors = {
        "AD_up": "#d73027",      # red
        "AD_down": "#4575b4",    # blue
        "not_significant": "#bdbdbd",  # gray
        "not_tested": "#969696",
    }

    lnc_colors = [de_colors.get(G.nodes[n].get("de_status", "not_tested"), "#bdbdbd") for n in lnc_nodes]
    mrna_colors = [de_colors.get(G.nodes[n].get("de_status", "not_tested"), "#bdbdbd") for n in mrna_nodes]

    # Node sizes based on degree
    lnc_sizes = [300 + 40 * G.degree(n) for n in lnc_nodes]
    mrna_sizes = [120 + 25 * G.degree(n) for n in mrna_nodes]

    # Edges: positive = crimson, negative = royalblue, alpha by weight
    pos_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("direction") == "positive"]
    neg_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("direction") == "negative"]

    pos_weights = [G[u][v].get("weight", 0.5) * 2.2 for u, v in pos_edges]
    neg_weights = [G[u][v].get("weight", 0.5) * 2.2 for u, v in neg_edges]

    # Draw edges
    nx.draw_networkx_edges(G, pos, edgelist=pos_edges, width=pos_weights, edge_color="#d73027", alpha=0.55)
    nx.draw_networkx_edges(G, pos, edgelist=neg_edges, width=neg_weights, edge_color="#313695", alpha=0.55, style="dashed")

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, nodelist=mrna_nodes, node_shape="o", node_color=mrna_colors,
        node_size=mrna_sizes, edgecolors="#252525", linewidths=0.8
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=lnc_nodes, node_shape="D", node_color=lnc_colors,
        node_size=lnc_sizes, edgecolors="#000000", linewidths=2.0
    )

    # Labels: label all lncRNAs and key partner mRNAs
    labels = {}
    for n in G.nodes():
        if G.nodes[n].get("node_type") == "lncRNA" or G.degree(n) >= 2 or G.nodes[n].get("symbol") in [
            "RPL34", "RPL36A", "SYN2", "ENC1", "CD68", "ITGB2", "NDUFA2", "NDUFB1", "USP33", "ATP2B3"
        ]:
            labels[n] = n

    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight="bold", font_family="sans-serif")

    # Legend
    shape_lnc = mpatches.Patch(color="#525252", label="lncRNA (Diamond)")
    shape_mrna = mpatches.Patch(color="#969696", label="Protein-coding mRNA (Circle)")
    de_up = mpatches.Patch(color="#d73027", label="AD Upregulated (FDR < 0.05)")
    de_down = mpatches.Patch(color="#4575b4", label="AD Downregulated (FDR < 0.05)")
    de_ns = mpatches.Patch(color="#bdbdbd", label="Unchanged / Not tested")
    edge_pos = plt.Line2D([0], [0], color="#d73027", lw=2, label="Positive co-expression (r >= 0.75)")
    edge_neg = plt.Line2D([0], [0], color="#313695", lw=2, linestyle="--", label="Negative co-expression (r <= -0.75)")

    plt.legend(
        handles=[shape_lnc, shape_mrna, de_up, de_down, de_ns, edge_pos, edge_neg],
        loc="upper right",
        frameon=True,
        framealpha=0.9,
        fontsize=9,
        title="Network Key",
        title_fontsize=10,
    )

    plt.title(
        "lncSignal-AD Brain Co-expression Modules in Fusiform Gyrus (GSE125583)\n"
        "Partial Pearson correlations adjusted for diagnosis, age, sex | Stage 13",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Saved modular network visualization: {out_path}")


def plot_subgraphs_grid(edges_df: pd.DataFrame, de_dict: dict, out_path: Path):
    """Render a 4-panel focused subgraph for the top candidate lncRNAs."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 14), dpi=300)

    subgraph_configs = [
        ("ZFAS1", "Panel A · ZFAS1 Cytoplasmic Translation Module", axes[0, 0]),
        ("LY86-AS1", "Panel B · LY86-AS1 Chemical Synaptic Module", axes[0, 1]),
        ("PCED1B-AS1", "Panel C · PCED1B-AS1 Microglial Immune Module", axes[1, 0]),
        ("GIHCG", "Panel D · GIHCG Mitochondrial Respiration Module", axes[1, 1]),
    ]

    for lnc_sym, title, ax in subgraph_configs:
        sub_edges = edges_df[edges_df["lnc_symbol"] == lnc_sym].sort_values(
            by="primary_r", key=abs, ascending=False
        ).head(16)

        sub_g = nx.Graph()
        sub_g.add_node(lnc_sym, node_type="lncRNA")

        for _, r in sub_edges.iterrows():
            csym = str(r["coding_symbol"])
            sub_g.add_node(csym, node_type="mRNA")
            sub_g.add_edge(lnc_sym, csym, r=float(r["primary_r"]), direction=str(r["direction"]))

        pos = nx.spring_layout(sub_g, k=0.6, seed=42)

        # Draw edges
        pos_edges = [(u, v) for u, v, d in sub_g.edges(data=True) if d.get("direction") == "positive"]
        neg_edges = [(u, v) for u, v, d in sub_g.edges(data=True) if d.get("direction") == "negative"]

        nx.draw_networkx_edges(sub_g, pos, edgelist=pos_edges, edge_color="#d73027", width=2.0, alpha=0.7, ax=ax)
        nx.draw_networkx_edges(sub_g, pos, edgelist=neg_edges, edge_color="#313695", width=2.0, alpha=0.7, style="dashed", ax=ax)

        # Draw nodes
        mrna_nodes = [n for n in sub_g.nodes() if n != lnc_sym]
        nx.draw_networkx_nodes(sub_g, pos, nodelist=[lnc_sym], node_shape="D", node_color="#fc8d59", node_size=600, edgecolors="black", linewidths=1.5, ax=ax)
        nx.draw_networkx_nodes(sub_g, pos, nodelist=mrna_nodes, node_shape="o", node_color="#91bfdb", node_size=350, edgecolors="#4575b4", linewidths=1.0, ax=ax)

        nx.draw_networkx_labels(sub_g, pos, font_size=8, font_weight="bold", ax=ax)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.axis("off")

    plt.suptitle("Focused Co-expression Subgraphs of Prioritized lncRNAs", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Saved candidate subgraphs grid: {out_path}")


def plot_pathway_enrichment_dotplot(enrichment_path: Path, out_path: Path):
    """Plot multi-database pathway enrichment dot plot comparing Reactome, KEGG, and GO BP."""
    if not enrichment_path.exists():
        logger.warning(f"Enrichment file {enrichment_path} not found, skipping dot plot.")
        return

    df = pd.read_csv(enrichment_path)
    # Focus on prominent candidates
    cands = ["ZFAS1", "LY86-AS1", "PCED1B-AS1", "GIHCG", "SMARCA5-AS1"]
    sub_df = df[df["lnc_symbol"].isin(cands)].copy()

    # Pick top 3 pathways per lncRNA and database
    top_rows = []
    for (lnc, db), grp in sub_df.groupby(["lnc_symbol", "database"]):
        top_rows.append(grp.sort_values("q_global").head(3))

    if not top_rows:
        return

    plot_df = pd.concat(top_rows, ignore_index=True)
    plot_df["neg_log_q"] = -np.log10(np.clip(plot_df["q_global"], 1e-150, 1.0))
    plot_df["label"] = plot_df["term_name"] + " (" + plot_df["database"] + ")"
    plot_df = plot_df.sort_values(by=["lnc_symbol", "fold_enrichment"], ascending=[True, True])

    plt.figure(figsize=(12, 10), dpi=300)
    scatter = plt.scatter(
        plot_df["fold_enrichment"],
        plot_df["label"],
        s=plot_df["overlap"] * 12 + 40,
        c=plot_df["neg_log_q"],
        cmap="viridis",
        alpha=0.85,
        edgecolors="black",
        linewidths=0.8,
    )

    cbar = plt.colorbar(scatter)
    cbar.set_label("-log10(Global FDR q)", fontsize=10)

    # Annotate with lncRNA symbol
    for _, row in plot_df.iterrows():
        plt.text(
            row["fold_enrichment"] + 1.2,
            row["label"],
            f"{row['lnc_symbol']} ({row['direction']})",
            va="center",
            fontsize=8,
            color="#333333",
        )

    plt.xlabel("Fold Enrichment", fontsize=11, fontweight="bold")
    plt.title(
        "Functional Pathway Enrichment across Reactome, KEGG & GO BP\n"
        "Strictly evaluated within GSE125583 measurable coding universe (16,799 genes)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Saved pathway enrichment dot plot: {out_path}")


def main():
    """Execute Stage 13 Co-expression Network Pipeline."""
    logger.info("Starting Stage 13 Co-expression Network Pipeline...")

    edges_df, bg_df, de_df, de_dict = load_network_data()

    # Prioritized lncRNAs from differential expression and co-expression analysis
    candidate_symbols = [
        "ZFAS1", "LY86-AS1", "PCED1B-AS1", "GIHCG", "SMARCA5-AS1",
        "OIP5-AS1", "PEG13", "ANXA2R-AS1", "PSMD6-AS2"
    ]

    G, nodes_df, selected_edges_df = build_candidate_network(
        edges_df, de_dict, candidate_symbols, max_partners_per_lnc=12
    )

    logger.info(f"Built network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Save network data
    nodes_df.to_csv(TABLES_DIR / "network_nodes.csv", index=False)
    selected_edges_df.to_csv(TABLES_DIR / "network_edges_biology.csv", index=False)

    # Compute and save topology metrics
    topology_summary = {
        "nodes_count": G.number_of_nodes(),
        "edges_count": G.number_of_edges(),
        "density": nx.density(G),
        "is_connected": nx.is_connected(G),
        "connected_components": nx.number_connected_components(G),
        "average_clustering": nx.average_clustering(G),
        "lncrna_nodes": int((nodes_df["node_type"] == "lncRNA").sum()),
        "mrna_nodes": int((nodes_df["node_type"] == "protein_coding").sum()),
        "top_degree_hubs": nodes_df.sort_values(by="degree", ascending=False)[["symbol", "node_type", "degree", "strength", "module"]].head(8).to_dict(orient="records"),
    }
    with open(TABLES_DIR / "network_topology_metrics.json", "w") as f:
        json.dump(topology_summary, f, indent=2)
    logger.info("Saved network topology metrics.")

    # Generate Figures
    plot_modular_network(G, nodes_df, FIG_DIR / "coexpression_network_modules.png")
    plot_subgraphs_grid(edges_df, de_dict, FIG_DIR / "subgraph_top_candidates.png")
    plot_pathway_enrichment_dotplot(TABLES_DIR / "top_candidate_pathway_enrichments.csv", FIG_DIR / "pathway_enrichment_dotplot.png")

    logger.info("Co-expression Network Pipeline Completed Successfully.")


if __name__ == "__main__":
    main()
