#!/usr/bin/env python3
"""Local Reactome enrichment for the six Cluster-2 attention modules.

The statistical universe is the same 200 shared genes supplied to OncoBERT.
Only public Reactome definitions are read; module assignments never leave the
workspace. One-sided hypergeometric P values are Benjamini-Hochberg corrected
separately for each module across all pathways represented by >=2 input genes.
"""

from __future__ import annotations

from pathlib import Path
import os

import numpy as np
import pandas as pd
from scipy.stats import hypergeom


ROOT = Path(os.environ.get("ONCOBERT_WORK_ROOT", Path(__file__).resolve().parents[2]))
BASE = ROOT / "artifacts/cluster2_attention_modules/spectral_k6"
MEMBERSHIP = BASE / "spectral_module_membership.tsv"
OUTPUT = BASE / "pathway_enrichment"
OUTPUT_PREFIX = "k6"
FUNCTIONAL_HYPOTHESES = {
    "M1": (
        "RAS–MAPK and receptor-kinase signaling",
        "FDR-significant SCF-KIT, NTRK, MET, RAF/MAPK, and RAS-mutant signaling terms",
    ),
    "M2": (
        "Cellular senescence and G1/S checkpoint control",
        "Top-ranked oxidative-stress senescence, SASP, cellular senescence, and RB/G1-S checkpoint terms",
    ),
    "M3": (
        "Transcriptional, epigenetic, and TGF-β tumor-suppressor control",
        "Top-ranked RNA polymerase II transcription, WDR5/histone regulation, TGF-beta/SMAD, and RUNX terms",
    ),
    "M4": (
        "DNA double-strand-break repair and lineage specification",
        "Top-ranked NHEJ/double-strand-break response plus WNT, FOXA1, GATA2, PAX5, and SOX9 lineage factors",
    ),
    "M5": (
        "Chromatin remodeling and genome-maintenance programs",
        "Strongest terms involve chromatin-modifying enzymes, SWI/SNF, telomere maintenance, and autophagy",
    ),
    "M6": (
        "Receptor tyrosine kinase–PI3K/AKT signaling",
        "Top-ranked RTK, aberrant PI3K, IGF1R, ERBB2, and FGFR signaling terms",
    ),
}
GMT = ROOT / "artifacts/reference_gene_sets/reactome/ReactomePathways.gmt"


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted P values with monotonicity enforcement."""
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def read_reactome() -> list[tuple[str, str, set[str]]]:
    pathways = []
    with GMT.open() as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 4:
                pathways.append((fields[0], fields[1], set(fields[2:])))
    return pathways


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    membership = pd.read_csv(MEMBERSHIP, sep="\t").sort_values("display_order")
    background = set(membership.gene.astype(str))
    pathways = read_reactome()
    records: list[dict[str, object]] = []
    background_size = len(background)

    for module, frame in membership.groupby("spectral_module", sort=True):
        genes = set(frame.gene.astype(str))
        local: list[dict[str, object]] = []
        for name, pathway_id, pathway_genes in pathways:
            represented = pathway_genes & background
            if len(represented) < 2:
                continue
            overlap = genes & represented
            p_value = hypergeom.sf(
                len(overlap) - 1, background_size, len(represented), len(genes)
            )
            local.append({
                "module": f"M{module}",
                "module_size": len(genes),
                "pathway_id": pathway_id,
                "pathway_name": name,
                "background_size": background_size,
                "pathway_genes_in_background": len(represented),
                "overlap_size": len(overlap),
                "expected_overlap": len(genes) * len(represented) / background_size,
                "enrichment_ratio": (
                    (len(overlap) / len(genes)) /
                    (len(represented) / background_size)
                ),
                "p_value": p_value,
                "overlap_genes": ";".join(sorted(overlap)),
                "pathway_genes_in_background_list": ";".join(sorted(represented)),
            })
        frame_results = pd.DataFrame(local)
        frame_results["fdr"] = bh_adjust(frame_results.p_value.to_numpy(float))
        records.extend(frame_results.to_dict("records"))

    results = pd.DataFrame(records).sort_values(
        ["module", "fdr", "p_value", "enrichment_ratio"],
        ascending=[True, True, True, False],
    )
    results.to_csv(OUTPUT / f"{OUTPUT_PREFIX}_reactome_enrichment_all_tested.tsv",
                   sep="\t", index=False)
    significant = results.loc[results.fdr.lt(0.05)].copy()
    significant.to_csv(OUTPUT / f"{OUTPUT_PREFIX}_reactome_enrichment_fdr_lt_0p05.tsv",
                       sep="\t", index=False)
    results.groupby("module", sort=False).head(20).to_csv(
        OUTPUT / f"{OUTPUT_PREFIX}_reactome_enrichment_top20_per_module.tsv",
        sep="\t", index=False,
    )
    hypothesis_rows = []
    for module, (title, rationale) in FUNCTIONAL_HYPOTHESES.items():
        local = results.loc[results.module.eq(module)].iloc[0]
        hypothesis_rows.append({
            "module": module,
            "summary_title": title,
            "interpretation_status": (
                "FDR-supported" if (significant.module == module).any()
                else "hypothesis from top-ranked non-FDR-significant pathways"
            ),
            "rationale": rationale,
            "leading_reactome_term": local.pathway_name,
            "leading_term_fdr": local.fdr,
        })
    pd.DataFrame(hypothesis_rows).to_csv(
        OUTPUT / f"{OUTPUT_PREFIX}_module_functional_hypotheses.tsv",
        sep="\t", index=False
    )
    print("Significant Reactome pathways (FDR < 0.05):")
    module_labels = [f"M{x}" for x in sorted(membership.spectral_module.unique())]
    print(significant.groupby("module").size().reindex(module_labels, fill_value=0).to_string())
    print(f"Saved {len(results):,} tests and {len(significant):,} significant results")


if __name__ == "__main__":
    main()
