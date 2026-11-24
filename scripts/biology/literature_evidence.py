"""Recheck a curated claim against independent PubMed and Crossref records."""
import csv
import json
import re
from pathlib import Path

import requests

TABLE = Path(__file__).resolve().parents[2] / "docs/literature_evidence_table.csv"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
REPLACEMENTS = {
    "34856901": ("34856901", "10.2174/1567205018666211202143449", "Comprehensive Analysis of Differential Expression Profiles of Long Noncoding RNAs with Associated Co-expression and Competing Endogenous RNA Networks in the Hippocampus of Patients with Alzheimer's Disease", "Computational association", "Human AD hippocampal GEO reanalysis includes OIP5-AS1 in a predicted network. No perturbation experiment confirms those edges or a neuroprotective mechanism."),
    "27687727": ("27871336", "10.1186/s13062-016-0165-y", "ZFAS1: a long noncoding RNA associated with ribosomes in breast cancer cells", "Experimental association", "Human breast cancer cells. ZFAS1 associates with ribosomes. Brain translation fidelity and AD regulation remain untested."),
    "21406540": ("21460236", "10.1261/rna.2528811", "SNORD-host RNA Zfas1 is a regulator of mammary development and a potential marker for breast cancer", "Direct mechanistic evidence", "Mouse mammary differentiation and human breast tissue. Zfas1 hosts snoRNAs. No AD experiment."),
    "18287011": ("18434550", "10.1073/pnas.0710263105", "Loss of microRNA cluster miR-29a/b-1 in sporadic Alzheimer's disease correlates with increased BACE1/beta-secretase expression", "Independent biological context", "Human AD brain and cell experiments on miR-29a/b-1. The MIR29B2CHG host transcript was not tested."),
}


# Topic and system reviewed against the retrieved records for rejected claims.
REJECTIONS = {
    "30286689": "Human social-support psychometrics. No ZFAS1 or ischemic neuronal experiment.",
    "20448464": "Human ICF syndrome and microRNA methylation. No AD or MIR29B2CHG evidence.",
    "32603684": "PMID studies bacterial isolates. DOI studies mesenchymal cells in spinal cord ischemia. Neither establishes OIP5-AS1 neuroprotection in AD.",
    "10745266": "PMID concerns transplant-associated viral reactivation. DOI concerns a human GABA receptor variant. Neither tests LY86-AS1 synaptic regulation in AD.",
    "28872473": "Human peripheral pain safety study and mouse corticostriatal erratum. Neither tests PCED1B-AS1 or AD microglial activation.",
    "24177439": "Preschool cortisol and human prostate cancer angiogenesis studies. Neither tests either lncRNA or mitochondrial control in AD.",
    "25257173": "Dendritic cells during viral infection and intracellular pH in E. coli. Neither tests PEG13 neuronal networks in AD.",
    "31536968": "Systemic glucocorticoid use and branched DNA processing by a nuclease. Neither evaluates read assignment to this human RNA or proves an artifact.",
    "30814729": "Avian behavior study. No human AD cohort or experiment on these three lncRNAs.",
}


def normalized(text):
    text = re.sub(r"<[^>]+>", "", text).replace(chr(946), "beta")
    return re.sub(r"[^a-z0-9]", "", text.casefold())


def matches(pmid, doi, title, pubmed, crossref):
    """Require independent title and identifier agreement. Missing fields fail closed."""
    return bool(title and doi and pubmed.get("pmid") == pmid
                and pubmed.get("doi", "").lower() == doi.lower() == crossref.get("doi", "").lower()
                and normalized(title) == normalized(pubmed.get("title", "")) == normalized(crossref.get("title", "")))


def lookup(url, params=None):
    try:
        response = requests.get(url, params=params, timeout=35)
        if response.status_code == 404:
            return {}, "unresolved identifier"
        response.raise_for_status()
        return response.json(), "retrieved"
    except (requests.RequestException, ValueError) as exc:
        return {}, f"unverified ({type(exc).__name__})"


def crossref_record(doi):
    data, status = lookup("https://api.crossref.org/works/" + doi)
    r = data.get("message", {})
    if not r:
        return {"status": status}
    return dict(doi=r["DOI"], title=r["title"][0], authors=", ".join(
        " ".join(filter(None, [a.get("given"), a.get("family")])) for a in r.get("author", [])),
        journal=r["container-title"][0], year=r["issued"]["date-parts"][0][0], status=status)


def main():
    with TABLE.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ids = [r["submitted_pmid"] for r in rows] + [r[0] for r in REPLACEMENTS.values()]
    data, status = lookup(EUTILS + "/esummary.fcgi", dict(db="pubmed", id=",".join(ids), retmode="json"))
    publications = {}
    for pmid in ids:
        r = data.get("result", {}).get(pmid, {})
        if r.get("title"):
            publications[pmid] = dict(pmid=pmid, title=r["title"], journal=r["fulljournalname"],
                year=r["pubdate"][:4], authors=", ".join(a["name"] for a in r["authors"]),
                doi=next((a["value"] for a in r["articleids"] if a["idtype"] == "doi"), ""))
    for row in rows:
        pmid, doi, title = (row[k] for k in ("submitted_pmid", "submitted_doi", "submitted_title"))
        p = publications.get(pmid, {"status": status})
        c = crossref_record(doi)
        row.update(pubmed_record=json.dumps(p, ensure_ascii=False), doi_record=json.dumps(c, ensure_ascii=False),
                   submitted_match=matches(pmid, doi, title, p, c), evidence="Unsupported hypothesis",
                   interpretation=REJECTIONS.get(pmid, "Citation does not verify the submitted mechanism."),
                   accepted_reference="")
        if pmid in REPLACEMENTS:
            pid, corrected_doi, corrected_title, evidence, interpretation = REPLACEMENTS[pmid]
            record = publications.get(pid, {})
            if matches(pid, corrected_doi, corrected_title, record, crossref_record(corrected_doi)):
                row.update(evidence=evidence, interpretation=interpretation, accepted_reference=json.dumps(record, ensure_ascii=False))
    # Preserve the last successful audit if an authoritative service is unavailable.
    if any(not r["accepted_reference"] for r in rows if r["submitted_pmid"] in REPLACEMENTS) or len(publications) < len(set(ids)) or any("unverified (" in r["doi_record"] for r in rows):
        raise RuntimeError("Incomplete bibliographic lookup; prior audit preserved")
    with TABLE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
