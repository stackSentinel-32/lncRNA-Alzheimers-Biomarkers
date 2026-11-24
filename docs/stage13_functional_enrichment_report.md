# GSE125583 brain co-expression and pathway interpretation

**Conditional.** The Stage 13 analysis and its real-data checks ran in the independent worktree. Numbered releases for the contributor's pathway snapshots are unknown. The isolated worktree lacks the earlier-stage raw and derived files required by 18 legacy tests. All research results below concern the fusiform gyrus cohort.

## Inputs and selection

Base commit `6762762e6361d1f6e2dbf0aca5c6664213536126`, contributor PR 2 commit `e81713af7079f763362c62d0c0a3a45a4b55bf2f`. Stage 12 adjusted partial Pearson correlations for diagnosis, age and sex in 288 complete cases and 281 after severe-QC exclusions. Its two all-pair BH families each contain 36,252,242 potential pairs. Stage 13 read the original outputs through `LNCSIGNAL_INPUT_ROOT`. It used Entrez IDs throughout. The original matrix was streamed for verification.

The enrichment foreground requires the same correlation sign, both global pairwise q values below 0.05 and both absolute correlations at least 0.75. The 88,089 memberships span 623 lncRNAs before size filtering. Separating positive and negative partners and requiring at least 10 unique coding genes leaves 560 lists. The background is all 16,799 measured, eligible coding genes. Reactome maps 9,799 and KEGG maps 7,438 before pathway-size filtering. The unmapped 7,000 and 9,361 genes respectively remain in the denominator. Annotation coverage can affect enrichment.

For each library, intersect unique pathway members with the measured background, retain pathways of size 10 through 500, and test every eligible list against every eligible pathway. For an overlap of x genes, a list of n, a pathway of K, and a universe of N, the one-sided test uses P(X >= x) under the hypergeometric distribution. Fold enrichment is (x/n)/(K/N). Zero overlaps enter BH with p=1. The significance criterion is database-wide BH q < 0.05 across all list-by-term hypotheses. Per-list q is supplementary. GO BP uses the existing Stage 12 NCBI annotation snapshot. An independent adjustment of its complete p-value vector reproduces the stored q values.

| Library | Eligible terms | Tests | Significant list-term associations | Distinct significant terms | Distinct lncRNAs |
|---|---:|---:|---:|---:|---:|
| Reactome | 1,159 | 649,040 | 4,911 | 424 | 279 |
| KEGG | 355 | 198,800 | 2,566 | 185 | 259 |
| GO BP | 1,826 | 1,022,560 | 3,258 | 357 | 278 |

The submitted aggregate counts reproduce. KEGG hsa04080 denotes *Neuroactive ligand-receptor interaction*. *Neuroactive ligand signaling* has ID hsa04082. The submitted narrative assigned both labels to hsa04080. The original parser also truncated internal hyphen-separated qualifiers. Comparing all 373 snapshot labels with the official human pathway list identified 27 truncated labels, now restored. The original membership tables were retained losslessly in compressed form. No membership update was applied. Overlapping annotations and shared coding genes limit the independence of findings across libraries. [KEGG hsa04080](https://www.kegg.jp/pathway/hsa04080) and [KEGG hsa04082](https://www.kegg.jp/pathway/hsa04082)

## Biological findings

| Candidate | Direction of coding partners | Example term | Database-wide q |
|---|---|---|---:|
| ZFAS1 | Positive | Cytoplasmic translation, GO BP | 4.38e-143 |
| LY86-AS1 | Positive | Chemical synaptic transmission, GO BP | 4.07e-37 |
| PEG13 | Positive | Chemical synaptic transmission, GO BP | 3.86e-32 |
| GIHCG | Positive | Oxidative phosphorylation, KEGG | 2.16e-27 |
| SMARCA5-AS1 | Negative | Oxidative phosphorylation, KEGG | 1.65e-21 |
| PCED1B-AS1 | Positive | Neutrophil degranulation, Reactome | 5.74e-19 |
| OIP5-AS1 | Positive | Antigen processing involving ubiquitination and proteasome degradation, Reactome R-HSA-983168 | 0.00208 |
| ANXA2R-AS1 | Negative | WNT5A-dependent internalization of FZD2, FZD5 and ROR2, Reactome | 0.0100 |

Coding partners of PCED1B-AS1 carry immune annotations. The cell type responsible for this signal is unresolved. Differences in neuronal or glial abundance, other tissue properties and technical variation could contribute to these bulk-tissue associations. Cell fractions were unavailable in the Stage 12 models. The positive and negative oxidative phosphorylation associations illustrate why the sign of each gene list matters. An OIP5-AS1 association with protein degradation genes does not establish an effect on proteostasis. PSMD6-AS2 is included in the graph and has no significant pathway in these three outputs.

Stage 11 brain DE identified TAGAP-AS1 (log2 fold change -0.545, q 5.26e-17), LINC02937 (+0.709, q 1.12e-16) and KDM7A-DT (-0.536, q 3.01e-15). Their mechanisms remain open. ZFAS1 illustrates the different scientific questions: its partner enrichment is strong while its primary brain DE q is 0.0531. Blood MIR29B2CHG did not pass replication FDR (q 0.109). The brain cohort cannot establish blood biomarker performance.

SLX1B-SULT1A4 and SULT1A4 overlap on chromosome 16 in the upstream annotation. Their correlations are +0.99165 (primary q 3.31e-246) and +0.99178 (sensitivity q 5.89e-241). Shared sequence makes ambiguous read assignment plausible. These estimates cannot distinguish shared reads from biological covariation. The cited nuclease article addresses branched DNA processing.

## Network audit

The nine illustrative candidates came from the contributor's display selection. The Stage 12 summary retains the three strongest primary correlations per lncRNA per sign among edges significant and sign-consistent in both fits. The contributor's later cap of 12 per sign never expanded that input. The reconstruction independently ranks all eligible partners for those nine candidates from the full Parquet. All 54 exported edges match, including both r and q values. There are 27 positive and 27 negative edges. Twelve edges fall below |r|=0.75 in at least one fit. The smallest primary |r| is 0.423.

The undirected simple graph contains 9 lncRNAs, 53 coding genes, 54 edges and 8 connected components. Each candidate reached the display cap of six partners. The ordinary density is 54/1,891 = 0.0285563. The bipartite density is 54/(9 x 53) = 0.113208. Triangle clustering is structurally zero. Strength sums absolute primary correlations, ranging from 2.784 for PSMD6-AS2 to 5.318 for PEG13 among lncRNAs. Closeness and betweenness use unweighted paths. The original code used correlation magnitude as path length for betweenness, symbols as node keys, hand-assigned functional modules and the first candidate's label for shared partners. The corrected exports use Entrez keys and connected-component IDs. Overlapping neighborhoods retain their individual edges. Missing brain DE measurements remain blank. Figure panels correspond to actual components. Degree ranking in this capped display cannot identify cohort-wide biological hubs.

## Literature audit

[The 13-record audit](literature_evidence_table.csv) retains each submitted title and identifier, independent PubMed and Crossref metadata, and the decision about the specific claim. The submitted implementation filled rows from curated strings and did not require a successful bibliographic match. The corrected checker requires agreement on PMID, DOI and normalized title before admitting a curated reference. Evidence scope was assessed separately. Thirteen PMIDs resolved. Eight submitted DOIs resolved in Crossref and five returned no record there. One submitted title/PMID/DOI combination matched. Nine records were excluded from biological support. The five unresolved submitted DOIs belong to the two initial ZFAS1 articles, the proposed ZFAS1 ischemia experiment, the second MIR29B2CHG citation and the three-candidate brain citation. The independently identified mammary-system articles have verified replacement identifiers.

- Hansji et al. observed ribosome-associated ZFAS1 in human breast cancer cells. This supports ribosome association in that system. It does not demonstrate translation fidelity or regulation in AD brain. [PMID 27871336](https://pubmed.ncbi.nlm.nih.gov/27871336/)
- Askarian-Amiri et al. characterized the snoRNA host and studied mammary differentiation. Those experiments provide direct evidence in their mammary system. [PMID 21460236](https://pubmed.ncbi.nlm.nih.gov/21460236/)
- Hébert et al. studied miR-29a/b-1 and BACE1 in AD and experimental systems. The result provides context for the miRNA family. It does not test the MIR29B2CHG host transcript. [PMID 18434550](https://pubmed.ncbi.nlm.nih.gov/18434550/)
- The OIP5-AS1 paper uses human AD hippocampal expression data and computational network inference. It provides independent computational context. Direct regulation and neuroprotection remain untested by that study. [PMID 34856901](https://pubmed.ncbi.nlm.nih.gov/34856901/)

## Reproduction and checks

Run in the independent worktree. The original data directory is used as input only. Ordinary enrichment and graph generation operate offline. The literature checker explicitly refreshes bibliographic metadata over the network. The stored audit can be read offline.

```powershell
Set-Location C:/Projects/lncSignal-AD-stage13-repair
$python = 'C:/Projects/lncSignal-AD/.venv/Scripts/python.exe'
$env:LNCSIGNAL_INPUT_ROOT = 'C:/Projects/lncSignal-AD'
& $python -m scripts.biology.functional_enrichment
& $python -m scripts.biology.coexpression_network
& $python -m scripts.biology.literature_evidence
& $python -m pytest -v tests/test_functional_enrichment.py tests/test_coexpression_network.py tests/test_literature_evidence.py
$env:STAGE13_VERIFY_ROOT = $env:LNCSIGNAL_INPUT_ROOT
& $python -m pytest -q tests/test_stage13_integration.py
Remove-Item Env:STAGE13_VERIFY_ROOT
& $python -m pytest -q --tb=line
& $python -m compileall -q scripts/biology tests
& $python -m pip check
```

Python 3.12.10, pandas 3.1.0, NumPy 2.5.4, SciPy 1.18.1, statsmodels 0.15.0, NetworkX 3.7, Matplotlib 3.11.2, PyArrow 26.0.0, requests 2.34.2 and pytest 9.2.0. Nine focused offline tests and four real-data integration checks passed. The latter compare foreground identities, ranked network edges and numerical values, reconstructed topology and strength, global q values and all serialized significant hits. Compileall and pip check succeeded. The default full suite had 327 passes, 35 skips and 18 failures from absent earlier-stage input paths. Failures were in test_paths and the Stage 10, 11 and 12 integration tests. No earlier stage was rerun.

The enrichment manifest records background, foreground and reference snapshot hashes. References derive from the source commit's Reactome human NCBI2Reactome subset and KEGG REST human link/name tables. The original retrieval did not record numbered releases. Compressed snapshots, checked-in labels and hashes identify the exact inputs used here. Enrichment outputs include full local Parquet statistics (regenerable, ignored), compressed significant tables and a compact candidate table giving the strongest significant association per direction and database. Network outputs include edge/node CSVs, topology JSON and two figures. All are under `outputs/{tables,figures}/biology`.

## Contributor change size

The 25-file submission added 103,975 text lines. Git does not count binary artifact lines. The 93,611 mapping lines account for 90.0% of the text diff. The duplicate literature JSON, redundant focused-network graphic and full all-test tables are omitted from Git. Significant tables and required mapping snapshots are compressed. The 619-line candidate extract is replaced by 28 selected results plus a header.

| Source file | Added lines |
|---|---:|
| `docs/biological_interpretation.md` | 208 |
| `docs/literature_evidence_table.csv` | 14 |
| `outputs/figures/biology/coexpression_network_modules.png` | binary |
| `outputs/figures/biology/pathway_enrichment_dotplot.png` | binary |
| `outputs/figures/biology/subgraph_top_candidates.png` | binary |
| `outputs/tables/biology/functional_enrichment_summary.json` | 56 |
| `outputs/tables/biology/kegg_all_tests.parquet` | binary |
| `outputs/tables/biology/kegg_significant.csv` | 2567 |
| `outputs/tables/biology/literature_evidence_records.json` | 197 |
| `outputs/tables/biology/network_edges_biology.csv` | 55 |
| `outputs/tables/biology/network_nodes.csv` | 63 |
| `outputs/tables/biology/network_topology_metrics.json` | 68 |
| `outputs/tables/biology/reactome_all_tests.parquet` | binary |
| `outputs/tables/biology/reactome_significant.csv` | 4912 |
| `outputs/tables/biology/reference/NCBI2Reactome_Homo_sapiens.tsv` | 53208 |
| `outputs/tables/biology/reference/kegg_hsa_gene_pathway.tsv` | 40029 |
| `outputs/tables/biology/reference/kegg_hsa_pathway_names.tsv` | 374 |
| `outputs/tables/biology/top_candidate_pathway_enrichments.csv` | 619 |
| `pytest.ini` | 2 |
| `scripts/biology/__init__.py` | 1 |
| `scripts/biology/coexpression_network.py` | 462 |
| `scripts/biology/functional_enrichment.py` | 476 |
| `scripts/biology/literature_evidence.py` | 446 |
| `tests/test_coexpression_network.py` | 108 |
| `tests/test_functional_enrichment.py` | 110 |

## Additional upstream checksums

Relative to the original project input root.

| Input | SHA-256 |
|---|---|
| `outputs/tables/stage12/edge_summary.csv` | `13714a824a2fcd9ec87e1260a5089186be5b4dd60ae15d8086521e6bc0487df5` |
| `outputs/tables/stage12/edges_per_lncrna.csv` | `011dcf4cdb36efbb34b2bd226f307b315d50bfde979fa84c1029e5fe8323855b` |
| `outputs/tables/stage12/GO_BP_all_tests.parquet` | `5c2d083cf8d78c5d4b8d69c295d5ac6f8baf27588633d5b451f44843e5de1cee` |
| `outputs/tables/stage12/GO_BP_significant.csv.gz` | `d8bc527107344b2222c56f510d525b7c6a5ea19d49e096166ba0e9d7a2c55b42` |
| `outputs/tables/stage11/GSE125583_differential_expression.csv` | `0a97aaf62d5946773aa263e9d190533033941aeeedf550ef796ddc2c3c4c2b2d` |
