"""Standardized sample-level TCGA broad, detailed, and ICD-O-3 retrieval.

All tasks use identical sample representations and patient-grouped five-fold
splits. Labels observed in fewer than five eligible samples are retained via a
single ``Other`` category.
"""
from __future__ import annotations

import json
from pathlib import Path
import os

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


REVISION = Path(os.environ.get("ONCOBERT_WORK_ROOT", Path(__file__).resolve().parents[2]))
ROOT = Path(os.environ.get("ONCOBERT_TCGA_MODEL_ROOT", Path(__file__).resolve().parents[2] / "data" / "derived" / "tcga_mc3_three_state_set_transformer"))
CLINICAL = Path(os.environ.get("ONCOBERT_TCGA_CLINICAL", Path(__file__).resolve().parents[2] / "data" / "raw" / "tcga_pancan_clinical_data.csv"))
OUTPUT = REVISION / "artifacts/standardized_sample_level_retrieval/tcga"
REPRESENTATIONS = {
    "NMF": (ROOT / "molecular_subtype_benchmark/representations/nmf_k13_embeddings.h5", "#9c755f"),
    "MutationProjector (GAT)": (ROOT / "molecular_subtype_benchmark/representations/mutation_projector_gat_embeddings.h5", "#59a14f"),
    "OncoBERT (Set Transformer)": (ROOT / "embeddings.h5", "#4e79a7"),
}
TASKS = {
    # Use the original TCGA PanCancer Atlas project labels (33 disease
    # acronyms) for the broad benchmark.  The cBioPortal ``Cancer Type``
    # field merges several projects (for example LUAD/LUSC and COAD/READ)
    # and therefore yields only 30 targets.
    "broad": ("TCGA PanCanAtlas Cancer Type Acronym", "Broad Cancer Type Retrieval"),
    "detailed": ("Oncotree Code", "Detailed Cancer Type Retrieval"),
    "icdo3": ("International Classification of Diseases for Oncology, Third Edition ICD-O-3 Histology Code", "ICD-O-3 Morphological Class Retrieval"),
}
TOP_K = (1, 3, 5, 10)
BOOTSTRAPS = 2_000
SEED = 42
INVALID = {"", "unknown", "not collected", "not applicable", "nan", "none"}


def load_representations() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    reference_path = REPRESENTATIONS["OncoBERT (Set Transformer)"][0]
    with h5py.File(reference_path) as handle:
        sample_ids = handle["sample_ids"].asstr()[:]
        sample_patients = handle["sample_patients"].asstr()[:]
    matrices = {}
    for model, (path, _) in REPRESENTATIONS.items():
        with h5py.File(path) as handle:
            if not np.array_equal(handle["sample_ids"].asstr()[:], sample_ids):
                raise ValueError(f"TCGA sample order mismatch: {path}")
            matrices[model] = handle["sample_embeddings"][:].astype(np.float64)
    clinical = pd.read_csv(CLINICAL, dtype=str, low_memory=False).drop_duplicates("Sample ID").set_index("Sample ID")
    short_ids = pd.Index([value[:15] for value in sample_ids])
    matched = clinical.reindex(short_ids)
    frame = pd.DataFrame({"sample_id": sample_ids, "clinical_sample_id": short_ids, "patient_id": sample_patients})
    for column, _ in TASKS.values():
        frame[column] = matched[column].to_numpy(object)
    return frame, matrices


def standardize(frame: pd.DataFrame, column: str):
    raw = frame[column].astype("string").str.strip()
    eligible = raw.notna() & ~raw.str.lower().isin(INVALID)
    counts = raw[eligible].value_counts()
    rare = set(counts[counts < 5].index.astype(str))
    labels = raw[eligible].astype(str).where(~raw[eligible].astype(str).isin(rare), "Other").to_numpy()
    audit = pd.DataFrame({"original_label":counts.index.astype(str), "sample_count":counts.to_numpy()})
    audit["standardized_label"] = audit.original_label.where(audit.sample_count >= 5, "Other")
    return eligible.to_numpy(), labels, audit


def patient_bootstrap(correct: np.ndarray, patients: np.ndarray) -> tuple[float,float]:
    grouped = pd.DataFrame({"patient":patients,"correct":correct.astype(float)}).groupby("patient",sort=False).correct.agg(["sum","count"])
    num, den = grouped["sum"].to_numpy(), grouped["count"].to_numpy()
    rng = np.random.default_rng(SEED); estimates = np.empty(BOOTSTRAPS)
    for start in range(0,BOOTSTRAPS,100):
        stop=min(start+100,BOOTSTRAPS); draw=rng.integers(0,len(grouped),(stop-start,len(grouped)),dtype=np.int32)
        estimates[start:stop]=num[draw].sum(1)/den[draw].sum(1)
    return tuple(np.quantile(estimates,[.025,.975]))


def evaluate(frame: pd.DataFrame, matrices: dict[str,np.ndarray], task: str):
    column,title=TASKS[task]; eligible,labels,audit=standardize(frame,column)
    local=frame.loc[eligible].reset_index(names="source_row"); local["label"]=labels
    groups=local.patient_id.to_numpy(str); classes=np.sort(np.unique(labels)); class_index={v:i for i,v in enumerate(classes)}
    splitter=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
    folds=list(splitter.split(np.zeros(len(labels)),labels,groups))
    rows=[]; all_predictions=[]
    for model,full in matrices.items():
        matrix=full[local.source_row.to_numpy(int)]; scores=np.full((len(labels),len(classes)),-np.inf,dtype=np.float64)
        fold_id=np.full(len(labels),-1,np.int8)
        for fold,(train,test) in enumerate(folds):
            scaler=StandardScaler().fit(matrix[train])
            probe=RidgeClassifier(alpha=1,class_weight="balanced",solver="cholesky").fit(scaler.transform(matrix[train]),labels[train])
            local_scores=probe.decision_function(scaler.transform(matrix[test]))
            if local_scores.ndim==1: local_scores=np.column_stack([-local_scores,local_scores])
            columns=np.array([class_index[v] for v in probe.classes_])
            scores[np.ix_(test,columns)]=local_scores; fold_id[test]=fold
        if (fold_id<0).any(): raise ValueError("Incomplete grouped CV predictions")
        truth=np.array([class_index[v] for v in labels]); ranking=np.argsort(scores,axis=1)[:,::-1]
        correct={k:(ranking[:,:min(k,len(classes))]==truth[:,None]).any(1) for k in TOP_K}
        row={"model":model,"dimension":matrix.shape[1],"samples":len(labels),"patients":local.patient_id.nunique(),"standardized_class_count":len(classes)}
        for k in TOP_K:
            lo,hi=patient_bootstrap(correct[k],groups)
            row.update({f"top{k}_accuracy":float(correct[k].mean()),f"top{k}_ci_lower":float(lo),f"top{k}_ci_upper":float(hi)})
        rows.append(row)
        guesses=classes[ranking[:,0]]
        for i in range(len(local)):
            all_predictions.append({"model":model,"sample_id":local.sample_id.iloc[i],"patient_id":groups[i],"label":labels[i],"fold":int(fold_id[i]),"top1_prediction":guesses[i],**{f"top{k}_correct":bool(correct[k][i]) for k in TOP_K}})
    metadata={
        "task":task,"title":title,"target_column":column,"eligible_samples":len(local),"eligible_patients":local.patient_id.nunique(),
        "original_class_count":audit.original_label.nunique(),"rare_original_class_count":int((audit.sample_count<5).sum()),
        "rare_samples_mapped_to_other":int(audit.loc[audit.sample_count<5,"sample_count"].sum()),"standardized_class_count":len(classes),
        "protocol":"Sample-level target; labels with <5 eligible samples mapped to Other; patient-grouped stratified five-fold CV; fold-training StandardScaler; class-balanced Ridge alpha=1; 2,000 patient-cluster bootstrap replicates.",
    }
    return pd.DataFrame(rows),pd.DataFrame(all_predictions),audit,metadata


def plot(results:pd.DataFrame,metadata:dict,destination:Path,stem:str):
    local=results.set_index("model").loc[list(REPRESENTATIONS)]; positions=np.arange(len(TOP_K)); width=.24
    figure,axis=plt.subplots(figsize=(11.2,6.7)); all_high=[]
    for index,(model,(_,color)) in enumerate(REPRESENTATIONS.items()):
        values=100*np.array([local.loc[model,f"top{k}_accuracy"] for k in TOP_K]); lower=100*np.array([local.loc[model,f"top{k}_ci_lower"] for k in TOP_K]); upper=100*np.array([local.loc[model,f"top{k}_ci_upper"] for k in TOP_K]); all_high.extend(upper)
        bars=axis.bar(positions+(index-1)*width,values,width,yerr=np.vstack([values-lower,upper-values]),capsize=4,color=color,label=model,error_kw={"elinewidth":1.2,"capthick":1.2})
        for bar,value,high in zip(bars,values,upper): axis.text(bar.get_x()+bar.get_width()/2,high+.45,f"{value:.1f}",ha="center",va="bottom",fontsize=12.5)
    chance=100*np.asarray(TOP_K)/metadata["standardized_class_count"]; axis.hlines(chance,positions-.42,positions+.42,color="black",linestyle="--",linewidth=2.0,zorder=5)
    axis.set_xticks(positions,[f"Top-{k}" for k in TOP_K]); axis.set_ylabel("Five-fold held-out sample accuracy (%)",fontsize=15)
    axis.tick_params(axis="both",labelsize=14)
    axis.set_title(f"TCGA {metadata['title']} ({metadata['standardized_class_count']} Classes)",fontsize=20,pad=15)
    axis.set_ylim(0,max(all_high)+7); axis.grid(axis="y",alpha=.18); axis.spines[["top","right"]].set_visible(False)
    handles,labels=axis.get_legend_handles_labels(); handles.append(Line2D([0],[0],color="black",linestyle="--",linewidth=2.0));labels.append("Random guessing");axis.legend(handles,labels,frameon=False,ncol=2,loc="upper left",fontsize=13)
    figure.text(.5,.022,f"{metadata['eligible_samples']:,} samples from {metadata['eligible_patients']:,} patients; patient-grouped five-fold CV. {metadata['rare_original_class_count']} labels ({metadata['rare_samples_mapped_to_other']} samples) mapped to Other. Error bars: patient-cluster-bootstrap 95% CIs.",ha="center",fontsize=9,color="#444444")
    figure.subplots_adjust(left=.10,right=.98,top=.88,bottom=.16)
    for extension in ("pdf","png","svg"):figure.savefig(destination/f"{stem}.{extension}",dpi=240,bbox_inches="tight")
    plt.close(figure)


def main():
    frame,matrices=load_representations(); OUTPUT.mkdir(parents=True,exist_ok=True); summary={}
    for task in TASKS:
        destination=OUTPUT/task;destination.mkdir(parents=True,exist_ok=True)
        results,predictions,audit,metadata=evaluate(frame,matrices,task)
        results.to_csv(destination/"model_topk_metrics.csv",index=False);predictions.to_csv(destination/"heldout_sample_predictions.csv.gz",index=False,compression="gzip");audit.to_csv(destination/"label_standardization_audit.csv",index=False)
        plot(results,metadata,destination,f"tcga_{task}_three_model_retrieval")
        (destination/"manifest.json").write_text(json.dumps({"complete":True,**metadata,"results":results.to_dict("records")},indent=2)+"\n")
        summary[task]=metadata;print(task,json.dumps(metadata,indent=2));print(results.to_string(index=False))
    (OUTPUT/"manifest.json").write_text(json.dumps({"complete":True,"tasks":summary},indent=2)+"\n")


if __name__=="__main__":main()
