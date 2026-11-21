"""Stage 13: Functional Enrichment Analysis across Reactome, KEGG, and GO databases.

Extends Stage 12 local GO BP enrichment to Reactome and KEGG pathway collections,
strictly maintaining the GSE125583 measurable protein-coding universe (16,799 genes)
as the statistical background.
"""
from collections import defaultdict
import hashlib
import json
import logging
from pathlib import Path
import sys
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = ROOT / "outputs/tables/biology"
REF_DIR = TABLES_DIR / "reference"
STAGE12_DIR = ROOT / "outputs/tables/stage12"
STAGE11_DIR = ROOT / "outputs/tables/stage11"

REF_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("functional_enrichment")


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def bh_adjust(p_values: np.ndarray | pd.Series) -> np.ndarray:
    """Compute Benjamini-Hochberg FDR q-values."""
    p = np.asarray(p_values, dtype=float)
    valid = np.isfinite(p)
    q = np.full_like(p, np.nan)
    if valid.any():
        q[valid] = multipletests(p[valid], method="fdr_bh")[1]
    return q


def fetch_reactome_mappings(cache_path: Path | None = None) -> tuple[dict[str, dict], dict[str, str]]:
    """Fetch or load cached NCBI Entrez to Reactome pathway mappings for Homo sapiens.

    Returns
    -------
    tuple[dict[str, dict], dict[str, str]]
        - terms: mapping of pathway_id -> {'name': str, 'url': str, 'genes': set[str]}
        - metadata: metadata dict with source url, date/status, and sha256 checksum
    """
    if cache_path is None:
        cache_path = REF_DIR / "NCBI2Reactome_Homo_sapiens.tsv"

    if not cache_path.exists():
        logger.info("Downloading NCBI2Reactome from reactome.org...")
        url = "https://reactome.org/download/current/NCBI2Reactome.txt"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Bioinformatics Pipeline)"})
        records = []
        with urllib.request.urlopen(req, timeout=60) as resp:
            for line in resp:
                parts = line.decode("utf-8", errors="ignore").strip().split("\t")
                if len(parts) >= 6 and parts[5] == "Homo sapiens":
                    # EntrezID, PathwayID, URL, PathwayName, Evidence, Species
                    records.append((parts[0], parts[1], parts[2], parts[3]))
        df = pd.DataFrame(records, columns=["gene_id", "pathway_id", "url", "pathway_name"])
        df.to_csv(cache_path, sep="\t", index=False)
        logger.info(f"Saved {len(df)} Reactome mappings to {cache_path}")
    else:
        logger.info(f"Loading cached Reactome mappings from {cache_path}")
        df = pd.read_csv(cache_path, sep="\t", dtype=str)

    terms = defaultdict(lambda: {"name": "", "url": "", "genes": set()})
    for _, row in df.iterrows():
        pid = row["pathway_id"]
        terms[pid]["name"] = row["pathway_name"]
        terms[pid]["url"] = row["url"]
        terms[pid]["genes"].add(str(row["gene_id"]))

    sha256 = compute_sha256(cache_path)
    metadata = {
        "source": "https://reactome.org/download/current/NCBI2Reactome.txt",
        "file": str(cache_path.relative_to(ROOT)),
        "sha256": sha256,
        "total_mappings": len(df),
        "distinct_pathways": len(terms),
    }
    return dict(terms), metadata


def fetch_kegg_mappings(cache_dir: Path | None = None) -> tuple[dict[str, dict], dict[str, str]]:
    """Fetch or load cached KEGG human pathway mappings from KEGG REST API.

    Returns
    -------
    tuple[dict[str, dict], dict[str, str]]
        - terms: mapping of pathway_id -> {'name': str, 'genes': set[str]}
        - metadata: metadata dict with source url, date/status, and sha256 checksum
    """
    if cache_dir is None:
        cache_dir = REF_DIR

    link_path = cache_dir / "kegg_hsa_gene_pathway.tsv"
    name_path = cache_dir / "kegg_hsa_pathway_names.tsv"

    if not link_path.exists():
        logger.info("Fetching KEGG pathway gene links from rest.kegg.jp...")
        link_url = "https://rest.kegg.jp/link/pathway/hsa"
        req = urllib.request.Request(link_url, headers={"User-Agent": "Mozilla/5.0 (Bioinformatics Pipeline)"})
        links = []
        with urllib.request.urlopen(req, timeout=60) as resp:
            for line in resp:
                parts = line.decode("utf-8").strip().split("\t")
                if len(parts) >= 2:
                    gene = parts[0].replace("hsa:", "")
                    pathway = parts[1].replace("path:", "")
                    links.append((gene, pathway))
        df_links = pd.DataFrame(links, columns=["gene_id", "pathway_id"])
        df_links.to_csv(link_path, sep="\t", index=False)
    else:
        logger.info(f"Loading cached KEGG links from {link_path}")
        df_links = pd.read_csv(link_path, sep="\t", dtype=str)

    if not name_path.exists():
        logger.info("Fetching KEGG pathway names from rest.kegg.jp...")
        name_url = "https://rest.kegg.jp/list/pathway/hsa"
        req = urllib.request.Request(name_url, headers={"User-Agent": "Mozilla/5.0 (Bioinformatics Pipeline)"})
        names = []
        with urllib.request.urlopen(req, timeout=60) as resp:
            for line in resp:
                parts = line.decode("utf-8").strip().split("\t")
                if len(parts) >= 2:
                    pid = parts[0].replace("path:", "")
                    pname = parts[1].split(" - ")[0]  # strip " - Homo sapiens (human)"
                    names.append((pid, pname))
        df_names = pd.DataFrame(names, columns=["pathway_id", "pathway_name"])
        df_names.to_csv(name_path, sep="\t", index=False)
    else:
        logger.info(f"Loading cached KEGG pathway names from {name_path}")
        df_names = pd.read_csv(name_path, sep="\t", dtype=str)

    name_map = dict(zip(df_names["pathway_id"], df_names["pathway_name"]))
    terms = defaultdict(lambda: {"name": "", "genes": set()})
    for _, row in df_links.iterrows():
        pid = row["pathway_id"]
        terms[pid]["name"] = name_map.get(pid, pid)
        terms[pid]["genes"].add(str(row["gene_id"]))

    metadata = {
        "source": "https://rest.kegg.jp",
        "link_file": str(link_path.relative_to(ROOT)),
        "link_sha256": compute_sha256(link_path),
        "name_file": str(name_path.relative_to(ROOT)),
        "name_sha256": compute_sha256(name_path),
        "total_mappings": len(df_links),
        "distinct_pathways": len(terms),
    }
    return dict(terms), metadata


def run_hypergeometric_enrichment(
    foreground_df: pd.DataFrame,
    background_genes: set[str],
    symbol_map: dict[str, str],
    lnc_symbol_map: dict[str, str],
    database_terms: dict[str, dict],
    db_name: str,
    min_term_size: int = 10,
    max_term_size: int = 500,
    min_foreground_size: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run one-sided hypergeometric overrepresentation testing against the background universe.

    Parameters
    ----------
    foreground_df : pd.DataFrame
        Must have columns ['lnc_entrez', 'direction', 'coding_entrez'].
    background_genes : set[str]
        The 16,799 measurable protein-coding genes.
    symbol_map : dict[str, str]
        coding_entrez -> gene_symbol.
    lnc_symbol_map : dict[str, str]
        lnc_entrez -> lncRNA symbol.
    database_terms : dict[str, dict]
        pathway_id -> {'name': str, 'genes': set[str], ...}
    db_name : str
        Name of database ('Reactome' or 'KEGG').
    min_term_size : int
        Minimum number of background genes annotated to term.
    max_term_size : int
        Maximum number of background genes annotated to term.
    min_foreground_size : int
        Minimum foreground genes per list.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, dict]
        - all_tests: pd.DataFrame of all conducted tests with unadjusted p and FDR.
        - significant_tests: pd.DataFrame of tests with q_global < 0.05.
        - summary: summary dictionary with test statistics.
    """
    universe_size = len(background_genes)

    # Filter terms to genes present in the background universe and check size bounds
    eligible_terms = {}
    for term_id, term_data in database_terms.items():
        bg_overlap = term_data["genes"] & background_genes
        if min_term_size <= len(bg_overlap) <= max_term_size:
            eligible_terms[term_id] = {
                "name": term_data["name"],
                "genes": bg_overlap,
                "size": len(bg_overlap),
            }

    logger.info(
        f"{db_name}: {len(eligible_terms)} terms eligible (size {min_term_size}-{max_term_size} in background)"
    )

    term_ids = list(eligible_terms.keys())
    term_sizes = np.array([eligible_terms[t]["size"] for t in term_ids], dtype=int)
    term_names = [eligible_terms[t]["name"] for t in term_ids]

    # Pre-build reverse index for fast hit calculation: gene -> list of term indices
    gene_to_term_indices = defaultdict(list)
    for idx, term_id in enumerate(term_ids):
        for gene in eligible_terms[term_id]["genes"]:
            gene_to_term_indices[gene].append(idx)

    test_rows = []
    eligible_lists_count = 0

    # Group foreground by lncRNA and direction
    for (lnc_entrez, direction), group in foreground_df.groupby(["lnc_entrez", "direction"]):
        fg_genes = set(group["coding_entrez"].astype(str))
        if not fg_genes.issubset(background_genes):
            raise ValueError(f"Foreground for lncRNA {lnc_entrez} contains genes outside measurable universe!")

        fg_size = len(fg_genes)
        if fg_size < min_foreground_size or not term_ids:
            continue

        eligible_lists_count += 1
        lnc_sym = lnc_symbol_map.get(str(lnc_entrez), str(lnc_entrez))

        # Count overlaps
        term_hit_counts = np.bincount(
            [idx for g in fg_genes for idx in gene_to_term_indices.get(g, [])],
            minlength=len(term_ids),
        )

        p_values = np.ones(len(term_ids), dtype=float)
        positive_hits = term_hit_counts > 0

        # One-sided hypergeometric test: P(X >= hits) = sf(hits - 1, universe, term_size, fg_size)
        p_values[positive_hits] = hypergeom.sf(
            term_hit_counts[positive_hits] - 1,
            universe_size,
            term_sizes[positive_hits],
            fg_size,
        )

        fold_enrichments = np.zeros(len(term_ids), dtype=float)
        fold_enrichments[positive_hits] = (term_hit_counts[positive_hits] / fg_size) / (
            term_sizes[positive_hits] / universe_size
        )

        # Per-list FDR
        q_list = bh_adjust(p_values)

        for i, tid in enumerate(term_ids):
            hits = int(term_hit_counts[i])
            # Only record detailed gene overlap if hits > 0
            overlap_symbols = ""
            if hits > 0:
                overlapping_genes = fg_genes & eligible_terms[tid]["genes"]
                overlap_symbols = ";".join(
                    sorted(symbol_map.get(g, g) for g in overlapping_genes)
                )

            test_rows.append({
                "database": db_name,
                "term_id": tid,
                "term_name": term_names[i],
                "term_size": int(term_sizes[i]),
                "lnc_entrez": str(lnc_entrez),
                "lnc_symbol": lnc_sym,
                "direction": direction,
                "foreground_size": fg_size,
                "overlap": hits,
                "fold_enrichment": round(float(fold_enrichments[i]), 4),
                "p_value": float(p_values[i]),
                "q_list": float(q_list[i]),
                "overlap_genes": overlap_symbols,
            })

    all_tests_df = pd.DataFrame(test_rows)
    if not all_tests_df.empty:
        # Global Benjamini-Hochberg adjustment across ALL tested list-term pairs
        all_tests_df["q_global"] = bh_adjust(all_tests_df["p_value"])
    else:
        all_tests_df["q_global"] = []

    sig_tests_df = (
        all_tests_df[all_tests_df["q_global"].lt(0.05) & (all_tests_df["overlap"] > 0)]
        .sort_values(["q_global", "p_value"])
        .reset_index(drop=True)
    )

    summary = {
        "database": db_name,
        "background_universe": universe_size,
        "eligible_terms": len(eligible_terms),
        "eligible_foreground_lists": eligible_lists_count,
        "total_tests": len(all_tests_df),
        "significant_tests_q_global_05": len(sig_tests_df),
        "distinct_significant_terms": int(sig_tests_df["term_id"].nunique()) if not sig_tests_df.empty else 0,
        "distinct_significant_lncrnas": int(sig_tests_df["lnc_entrez"].nunique()) if not sig_tests_df.empty else 0,
    }

    logger.info(
        f"{db_name}: {len(sig_tests_df)} significant associations at global BH q < 0.05 "
        f"({summary['distinct_significant_terms']} distinct pathways across {summary['distinct_significant_lncrnas']} lncRNAs)"
    )

    return all_tests_df, sig_tests_df, summary


def main():
    """Execute Stage 13 functional enrichment pipeline."""
    logger.info("Starting Stage 13 Functional Enrichment Analysis...")

    # 1. Load Background Universe (measurable protein-coding genes)
    bg_path = STAGE12_DIR / "enrichment_background.csv"
    if not bg_path.exists():
        raise FileNotFoundError(f"Missing background table: {bg_path}")
    bg_df = pd.read_csv(bg_path, dtype={"coding_entrez": str})
    background_genes = set(bg_df["coding_entrez"])
    symbol_map = dict(zip(bg_df["coding_entrez"], bg_df["symbol"]))
    logger.info(f"Loaded background universe: {len(background_genes)} coding genes")

    # 2. Load Foregrounds
    fg_path = STAGE12_DIR / "enrichment_foreground.csv.gz"
    if not fg_path.exists():
        raise FileNotFoundError(f"Missing foreground table: {fg_path}")
    fg_df = pd.read_csv(fg_path, dtype={"lnc_entrez": str, "coding_entrez": str})
    logger.info(f"Loaded {len(fg_df)} foreground memberships across {fg_df['lnc_entrez'].nunique()} lncRNAs")

    # 3. Load lncRNA symbol mappings
    edges_per_lnc_path = STAGE12_DIR / "edges_per_lncrna.csv"
    if edges_per_lnc_path.exists():
        edges_df = pd.read_csv(edges_per_lnc_path, dtype={"lnc_entrez": str})
        lnc_symbol_map = dict(zip(edges_df["lnc_entrez"], edges_df["symbol"]))
    else:
        lnc_symbol_map = {}

    # Supplement with candidate_lncrna_summary if available
    cand_path = STAGE11_DIR / "candidate_lncrna_summary.csv"
    if cand_path.exists():
        cand_df = pd.read_csv(cand_path, dtype={"feature_id": str})
        for _, r in cand_df.iterrows():
            if str(r["feature_id"]) not in lnc_symbol_map and pd.notna(r["symbol"]):
                lnc_symbol_map[str(r["feature_id"])] = str(r["symbol"])

    # 4. Fetch / Run Reactome Enrichment
    reactome_terms, reactome_meta = fetch_reactome_mappings()
    reactome_all, reactome_sig, reactome_summary = run_hypergeometric_enrichment(
        fg_df, background_genes, symbol_map, lnc_symbol_map, reactome_terms, "Reactome"
    )

    reactome_all.to_parquet(TABLES_DIR / "reactome_all_tests.parquet", compression="zstd", index=False)
    reactome_sig.to_csv(TABLES_DIR / "reactome_significant.csv", index=False)

    # 5. Fetch / Run KEGG Enrichment
    kegg_terms, kegg_meta = fetch_kegg_mappings()
    kegg_all, kegg_sig, kegg_summary = run_hypergeometric_enrichment(
        fg_df, background_genes, symbol_map, lnc_symbol_map, kegg_terms, "KEGG"
    )

    kegg_all.to_parquet(TABLES_DIR / "kegg_all_tests.parquet", compression="zstd", index=False)
    kegg_sig.to_csv(TABLES_DIR / "kegg_significant.csv", index=False)

    # 6. Audit existing GO BP results from Stage 12
    go_sig_path = STAGE12_DIR / "GO_BP_significant.csv.gz"
    go_meta = {}
    if go_sig_path.exists():
        go_sig = pd.read_csv(go_sig_path, dtype={"lnc_entrez": str})
        go_meta = {
            "source": "Human.GRCh38.p13.annot.tsv.gz (Stage 12 snapshot)",
            "total_significant": len(go_sig),
            "distinct_terms": int(go_sig["go_id"].nunique()),
            "distinct_lncrnas": int(go_sig["lnc_entrez"].nunique()),
        }

    # 7. Curate top candidate pathway enrichments
    # Priority lncRNAs: ZFAS1, LY86-AS1, PCED1B-AS1, GIHCG, SMARCA5-AS1, OIP5-AS1, PEG13, ANXA2R-AS1, PSMD6-AS2
    priority_symbols = [
        "ZFAS1", "LY86-AS1", "PCED1B-AS1", "GIHCG", "SMARCA5-AS1",
        "OIP5-AS1", "PEG13", "ANXA2R-AS1", "PSMD6-AS2", "LINC02937", "TAGAP-AS1", "KDM7A-DT"
    ]

    combined_sig_rows = []
    # Add Reactome significant
    for _, r in reactome_sig[reactome_sig["lnc_symbol"].isin(priority_symbols)].iterrows():
        combined_sig_rows.append(r.to_dict())

    # Add KEGG significant
    for _, r in kegg_sig[kegg_sig["lnc_symbol"].isin(priority_symbols)].iterrows():
        combined_sig_rows.append(r.to_dict())

    # Add GO BP significant for these symbols
    if go_sig_path.exists():
        go_sig["lnc_symbol"] = go_sig["lnc_entrez"].map(lnc_symbol_map)
        go_priority = go_sig[go_sig["lnc_symbol"].isin(priority_symbols)]
        for _, r in go_priority.iterrows():
            combined_sig_rows.append({
                "database": "GO_BP",
                "term_id": r["go_id"],
                "term_name": r["term"],
                "term_size": int(r["term_size"]),
                "lnc_entrez": str(r["lnc_entrez"]),
                "lnc_symbol": r["lnc_symbol"],
                "direction": r["direction"],
                "foreground_size": int(r["foreground_size"]),
                "overlap": int(r["overlap"]),
                "fold_enrichment": round(float(r["fold_enrichment"]), 4),
                "p_value": float(r["p"]),
                "q_list": np.nan,
                "overlap_genes": "",
                "q_global": float(r["q"]),
            })

    combined_df = pd.DataFrame(combined_sig_rows).sort_values(["lnc_symbol", "q_global", "p_value"])
    combined_df.to_csv(TABLES_DIR / "top_candidate_pathway_enrichments.csv", index=False)
    logger.info(f"Saved {len(combined_df)} top candidate pathway enrichments across databases.")

    # 8. Save overall summary manifest
    summary_manifest = {
        "stage": "Stage 13 - Functional Enrichment and Biological Interpretation",
        "background_universe": {
            "source": "GSE125583 fusiform gyrus measurable protein-coding genes",
            "file": str(bg_path.relative_to(ROOT)),
            "universe_size": len(background_genes),
            "sha256": compute_sha256(bg_path),
        },
        "foregrounds": {
            "source": "GSE125583 co-expression pairs (|r| >= 0.75 in both primary and sensitivity fits, q < 0.05)",
            "file": str(fg_path.relative_to(ROOT)),
            "memberships": len(fg_df),
            "distinct_lncrnas": int(fg_df["lnc_entrez"].nunique()),
            "sha256": compute_sha256(fg_path),
        },
        "databases": {
            "Reactome": {**reactome_meta, **reactome_summary},
            "KEGG": {**kegg_meta, **kegg_summary},
            "GO_BP_Stage12_Audit": go_meta,
        },
    }

    with open(TABLES_DIR / "functional_enrichment_summary.json", "w") as f:
        json.dump(summary_manifest, f, indent=2)
    logger.info("Saved functional_enrichment_summary.json")

    logger.info("Functional Enrichment Pipeline Completed Successfully.")


if __name__ == "__main__":
    main()
