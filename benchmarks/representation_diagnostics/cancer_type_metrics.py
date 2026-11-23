"""Cancer-type segregation and held-out decoding of existing patient embeddings.

Broad GENIE CANCER_TYPE labels are used without merging diagnoses. Unknown and
conflicting patient labels are excluded from scoring, but remain in UMAPs. Main
decoding uses types with >=100 training patients; all types are a sensitivity.
"""
from experiment_paths import relocated_json_loads
import argparse
import json
from pathlib import Path
import shutil

import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler

from diagnostics_utils import MODELS, TITLES, checksum, json_write, eta_squared
from institution_metrics import classification_metrics


def valid_labels(frame):
    return frame.cancer_type.notna() & ~frame.cancer_type.str.strip().str.lower().isin(
        ['unknown', 'multiple cancer types', 'not collected', 'not applicable', ''])


def decode(x, frame, output, name, scope):
    train=frame.split.to_numpy()=='train';test=frame.split.to_numpy()=='test'
    classes=np.sort(frame.cancer_type.unique())
    train_y=frame.cancer_type.to_numpy()[train];test_y=frame.cancer_type.to_numpy()[test]
    scaler=StandardScaler().fit(x[train].astype(np.float64))
    train_x=scaler.transform(x[train].astype(np.float64));test_x=scaler.transform(x[test].astype(np.float64))
    available=np.unique(train_y)
    centers=np.stack([train_x[train_y==label].mean(0) for label in available])
    distances=-2*test_x@centers.T+(centers**2).sum(1)
    centroid=available[distances.argmin(1)]
    model=RidgeClassifier(alpha=1,class_weight='balanced',solver='cholesky').fit(train_x,train_y)
    ridge=model.predict(test_x)
    majority=pd.Series(train_y).value_counts().idxmax()
    baseline,_,_=classification_metrics(test_y,np.repeat(majority,len(test_y)),classes)
    report=dict(train_patients=int(train.sum()),test_patients=int(test.sum()),
                train_classes=len(available),test_classes=len(np.unique(test_y)),
                unseen_test_class_patients=int((~np.isin(test_y,available)).sum()),
                majority_cancer_type=majority,majority_baseline=baseline,
                chance_balanced_accuracy=1/len(np.unique(test_y)))
    per_type=pd.DataFrame(dict(cancer_type=classes,train_patients=[int((train_y==c).sum()) for c in classes],
                               test_patients=[int((test_y==c).sum()) for c in classes]))
    for key,pred in [('nearest_centroid',centroid),('balanced_ridge',ridge)]:
        scores,matrix,recall=classification_metrics(test_y,pred,classes)
        report[key]=scores
        per_type[f'{key}_recall']=np.where(matrix.sum(1)>0,recall,np.nan)
        pd.DataFrame(matrix,index=pd.Index(classes,name='true_cancer_type'),columns=classes).to_csv(
            output/f'{name}_{scope}_{key}_confusion.csv')
    per_type.insert(0,'model',name);per_type.insert(1,'scope',scope)
    per_type.to_csv(output/f'{name}_{scope}_recalls.csv',index=False)
    pd.DataFrame(dict(patient_id=frame.index.to_numpy()[test],true_cancer_type=test_y,
                       nearest_centroid=centroid,balanced_ridge=ridge)).to_csv(
        output/f'{name}_{scope}_test_predictions.csv.gz',index=False)
    return report,per_type


def institution_baseline(frame, output):
    """Train-only cancer frequencies per institution: no embedding features."""
    train=frame[frame.split=='train'];test=frame[frame.split=='test']
    counts=pd.crosstab(train.institution,train.cancer_type)
    classes=np.sort(frame.cancer_type.unique());counts=counts.reindex(columns=classes,fill_value=0)
    totals=counts.sum(0).to_numpy(); assert (totals>0).all()
    scores=counts.reindex(test.institution).fillna(0).to_numpy(dtype=float)
    unknown=scores.sum(1)==0;scores[unknown]=totals
    ordinary=classes[scores.argmax(1)]
    balanced=classes[(scores/totals).argmax(1)]
    truth=test.cancer_type.to_numpy()
    ordinary_metrics,_,_=classification_metrics(truth,ordinary,classes)
    balanced_metrics,_,_=classification_metrics(truth,balanced,classes)
    report=dict(ordinary=ordinary_metrics,balanced=balanced_metrics,test_patients=len(test),
                interpretation='Train cancer frequencies per institution; balanced decision divides by training cancer-class frequencies.')
    pd.DataFrame(dict(patient_id=test.index,true_cancer_type=truth,institution=test.institution,
                       ordinary_prediction=ordinary,balanced_prediction=balanced)).to_csv(
        output/'institution_only_baseline_predictions.csv.gz',index=False)
    json_write(report,output/'institution_only_baseline.json')
    return report


def conditional_chance(graph, cancers, institutions):
    """Exact expected purity under label permutation within institutions."""
    levels,codes=np.unique(institutions,return_inverse=True)
    types,type_codes=np.unique(cancers,return_inverse=True)
    counts=np.zeros((len(levels),len(types)),dtype=np.float64)
    np.add.at(counts,(codes,type_codes),1)
    n=counts.sum(1);prob=counts/n[:,None]
    expectation=prob@prob.T
    diagonal=np.divide((counts*(counts-1)).sum(1),n*(n-1),out=np.zeros(len(n)),where=n>1)
    np.fill_diagonal(expectation,diagonal)
    return float(np.dot(graph.data.astype(np.float64),expectation[codes[graph.row],codes[graph.col]])/
                 graph.data.sum(dtype=np.float64))


def diagnostics(data, frame, ids):
    indices=data.obs_names.get_indexer(ids);assert (indices>=0).all()
    x=np.asarray(data.X)[indices]
    labels=frame.cancer_type.to_numpy()
    eta,adjusted=eta_squared(x,labels)
    graph=data.obsp['connectivities'][indices][:,indices].tocoo()
    raw=float(graph.data[labels[graph.row]==labels[graph.col]].sum(dtype=np.float64)/graph.data.sum(dtype=np.float64))
    _,counts=np.unique(labels,return_counts=True)
    chance=float(np.sum(counts*(counts-1))/(len(labels)*(len(labels)-1)))
    within=conditional_chance(graph,labels,frame.institution.to_numpy())
    return dict(patients=len(ids),cancer_types=len(counts),cancer_eta_squared=eta,
                cancer_df_adjusted_eta_squared=adjusted,weighted_neighbor_cancer_purity=raw,
                random_neighbor_cancer_purity=chance,adjusted_neighbor_cancer_purity=(raw-chance)/(1-chance),
                institution_conditioned_chance_purity=within,
                institution_conditioned_adjusted_cancer_purity=(raw-within)/(1-within)),x


def overlay(covariates, variants, top, colors, output, filename):
    fig,axes=plt.subplots(1,len(variants),figsize=(7*len(variants),10),squeeze=False)
    for ax,name in zip(axes.flat,variants):
        frame=covariates[name]
        labels=frame.cancer_type.where(frame.cancer_type.isin(top),'Other cancer types')
        labels.loc[~valid_labels(frame)]='Unknown / conflicting labels'
        xy=frame[['UMAP1','UMAP2']].to_numpy();order=np.random.default_rng(42).permutation(len(frame))
        ax.scatter(xy[order,0],xy[order,1],c=labels.map(colors).to_numpy()[order],s=.7,alpha=.7,
                   linewidths=0,rasterized=True)
        ax.set_title(f'{TITLES[name]}\nGENIE cancer type · {len(frame):,} patients',fontsize=12)
        ax.set_xlabel('UMAP 1');ax.set_ylabel('UMAP 2');ax.spines[['top','right']].set_visible(False)
    handles=[Line2D([0],[0],marker='o',linestyle='',color=color,label=label,markersize=5) for label,color in colors.items()]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.03),ncol=3 if len(variants)>1 else 2,fontsize=8)
    fig.suptitle('Cancer-type labels on existing patient UMAPs',fontsize=15,y=.99)
    fig.text(.5,.008,'Top 12 types shown separately for display; quantitative metrics retain individual diagnoses.\n'
             'All plotted patients retained. Shared colors; independent projection axes.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.25 if len(variants)>1 else .29,1,.94))
    for extension in ('png','pdf'):
        fig.savefig(output/f'{filename}.{extension}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def plots(table, recalls, baseline, output):
    fig,axes=plt.subplots(2,3,figsize=(18,10))
    labels=['ESM2','GenePT','STRING','Shuffled\ncontrol','GAT']
    colors=['#3572b0','#dd8452','#55a868','#909090','#b46ab5']
    fields=[('cancer_eta_squared','Embedding variance associated with cancer type'),
            ('adjusted_neighbor_cancer_purity','Cancer neighbor purity, adjusted for prevalence'),
            ('institution_conditioned_adjusted_cancer_purity','Cancer neighbor purity beyond institution case mix'),
            ('centroid_test_balanced_accuracy','Held-out nearest-centroid balanced accuracy'),
            ('ridge_test_balanced_accuracy','Held-out balanced-ridge balanced accuracy'),
            ('ridge_test_accuracy','Held-out balanced-ridge ordinary accuracy')]
    chance=table.iloc[0].chance_balanced_accuracy
    for ax,(column,title) in zip(axes.flat,fields):
        values=table[column].to_numpy();ax.bar(range(len(table)),values,color=colors)
        ax.set_xticks(range(len(table)),labels);ax.set_ylim(0,1);ax.set_title(title,fontsize=10)
        if 'balanced_accuracy' in column:
            ax.axhline(chance,color='#555555',ls='--',lw=1,label=f'Chance = {chance:.3f}')
            ax.axhline(baseline['balanced']['balanced_accuracy'],color='#b11c22',ls=':',lw=1,
                       label='Institution-only balanced rule')
            ax.legend(fontsize=7)
        for i,v in enumerate(values):
            ax.text(i,v+.012,f'{v:.3f}',ha='center',fontsize=9)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f'Cancer-type segregation · {int(table.iloc[0].patients):,} unambiguous patients\n'
                 f'Decoding: {int(table.iloc[0].supported_types)} types with ≥100 training patients',fontsize=15)
    fig.text(.5,.012,'Global metrics/decoders: all 256 dimensions. Neighbor purity: saved PCA affinities. '
             'Unknown/conflicting labels excluded from scoring.\n'
             'Fixed decoders fit on original train patients; original test patients for evaluation. '
             'Institution-conditioned purity preserves each institution’s cancer distribution in the null.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.075,1,.93))
    for ext in ('png','pdf'):fig.savefig(output/f'cancer_type_segregation_diagnostics.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    top=recalls[recalls.model=='esm2'].nlargest(20,'train_patients').cancer_type
    matrix=recalls.pivot(index='model',columns='cancer_type',values='balanced_ridge_recall').loc[list(MODELS),top]
    fig,ax=plt.subplots(figsize=(18,8));artist=ax.imshow(matrix,vmin=0,vmax=1,cmap='viridis',aspect='auto')
    ax.set_yticks(range(5),labels);ax.set_xticks(range(len(top)),top,rotation=55,ha='right',fontsize=8)
    for i in range(5):
        for j in range(len(top)):
            v=matrix.iloc[i,j];ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=8,color='white' if v<.55 else 'black')
    fig.colorbar(artist,ax=ax,label='Held-out recall')
    ax.set_title('Cancer-type test recall · top 20 by training prevalence · balanced ridge')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(output/f'cancer_type_test_recall_heatmap.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--covariates',type=Path,default=Path('experiments/patient_assay_coverage_tmb'))
    p.add_argument('--output',type=Path,default=Path('experiments/patient_cancer_type_benchmark'))
    args=p.parse_args();output=args.output;output.mkdir(parents=True,exist_ok=True)
    common=pd.read_csv(args.covariates/'common_diagnostic_patients.csv.gz').patient_id.to_numpy()
    original=pd.read_csv(args.covariates/'sequence_patient_covariates.csv.gz',index_col=0).loc[common]
    frame=original.loc[valid_labels(original)].copy();ids=frame.index.to_numpy()
    train_counts=frame.loc[frame.split=='train'].cancer_type.value_counts()
    supported=train_counts[train_counts>=100].index
    eligible=frame.cancer_type.isin(supported).to_numpy();supported_frame=frame.loc[eligible]
    frame[['cancer_type','institution','assay','split']].to_csv(output/'benchmark_patient_labels.csv.gz')
    pd.crosstab(frame.cancer_type,frame.split).to_csv(output/'cancer_type_split_counts.csv')
    baseline=institution_baseline(supported_frame,output)
    institution_manifest=relocated_json_loads(Path('experiments/patient_institution_benchmark/analysis_manifest.json').read_text())
    paths={name:Path(record['path']) for name,record in institution_manifest['input_h5ad'].items()}
    results=[];recalls=[];all_sensitivity=[]
    print(f'Known labels: {len(frame)} patients, {frame.cancer_type.nunique()} types; supported: {len(supported)} types, {len(supported_frame)} patients',flush=True)
    for name in MODELS:
        print('Cancer-type benchmark:',name,flush=True)
        d=ad.read_h5ad(paths[name]);assert np.array_equal(d.obs.loc[ids,'split'].astype(str),frame.split)
        metrics,x=diagnostics(d,frame,ids)
        decoder,recall=decode(x[eligible],supported_frame,output,name,'supported')
        all_decoder,_=decode(x,frame,output,name,'all_types')
        full_common_eta,_=eta_squared(np.asarray(d.X)[d.obs_names.get_indexer(common)],original.cancer_type.to_numpy())
        metrics.update(model=name,supported_types=len(supported),supported_patients=len(supported_frame),
                       centroid_test_accuracy=decoder['nearest_centroid']['accuracy'],
                       centroid_test_balanced_accuracy=decoder['nearest_centroid']['balanced_accuracy'],
                       ridge_test_accuracy=decoder['balanced_ridge']['accuracy'],
                       ridge_test_balanced_accuracy=decoder['balanced_ridge']['balanced_accuracy'],
                       chance_balanced_accuracy=decoder['chance_balanced_accuracy'],
                       majority_test_accuracy=decoder['majority_baseline']['accuracy'],
                       train_patients=decoder['train_patients'],test_patients=decoder['test_patients'],
                       full_common_cancer_eta_squared=full_common_eta)
        json_write(dict(metrics=metrics,supported_decoder=decoder,all_types_decoder=all_decoder),output/f'{name}_cancer_type_diagnostics.json')
        results.append(metrics);recalls.append(recall)
        all_sensitivity.append(dict(model=name,**{key:all_decoder[key] for key in ['train_patients','test_patients','train_classes','test_classes','unseen_test_class_patients']},
                                    centroid_accuracy=all_decoder['nearest_centroid']['accuracy'],
                                    centroid_balanced_accuracy=all_decoder['nearest_centroid']['balanced_accuracy'],
                                    ridge_accuracy=all_decoder['balanced_ridge']['accuracy'],
                                    ridge_balanced_accuracy=all_decoder['balanced_ridge']['balanced_accuracy']))
        print(metrics,flush=True);del d,x
    table=pd.DataFrame(results);table.to_csv(output/'cancer_type_benchmark.csv',index=False)
    per_type=pd.concat(recalls,ignore_index=True);per_type.to_csv(output/'supported_cancer_type_recalls.csv',index=False)
    pd.DataFrame(all_sensitivity).to_csv(output/'all_types_sensitivity.csv',index=False)
    plots(table,per_type,baseline,output)
    covariates={name:pd.read_csv(args.covariates/f'{name}_umap_covariates.csv.gz',index_col=0) for name in MODELS}
    top=frame.cancer_type.value_counts().head(12).index.tolist()
    palette=dict(zip(top,plt.colormaps['tab20'].colors[:12]));palette.update({'Other cancer types':'#b9bec5','Unknown / conflicting labels':'#111827'})
    for name in MODELS:overlay(covariates,[name],top,palette,output,f'{name}_umap_cancer_type')
    overlay(covariates,['esm2','genept','string'],top,palette,output,'primary_variants_umap_cancer_type')
    overlay(covariates,['shuffled','gat'],top,palette,output,'control_and_comparator_umap_cancer_type')
    manifest=dict(common_patients=len(common),scored_patients=len(frame),cancer_types=frame.cancer_type.nunique(),
                   excluded_label_counts=original.loc[~valid_labels(original)].cancer_type.value_counts().to_dict(),
                   supported_types=supported.tolist(),minimum_training_patients=100,
                   supported_patients=len(supported_frame),input_h5ad=institution_manifest['input_h5ad'],
                   source_sha256=checksum(__file__),covariates_sha256=checksum(args.covariates/'sequence_patient_covariates.csv.gz'),
                   decoder='Original train/test split; float64 train-only StandardScaler. Nearest centroid and RidgeClassifier alpha=1, balanced class weights, Cholesky. Primary types selected only by training counts; all-types sensitivity also reported.',
                   label_policy='Broad GENIE CANCER_TYPE, no synonym merging. All patient samples must have the same known label; Cancer of Unknown Primary remains a valid clinical category. UNKNOWN and conflicting multi-sample labels are excluded from scoring, retained in overlays.',
                   neighbor_null='Global random label permutation, plus exact expected purity when cancer labels are permuted within institution. This preserves institution cancer mix; it is not a causal assay/batch-effect correction.',
                   interpretation='Cancer-type decodability need not imply institution-independent biology. Same-cohort institution-only predictions and institution-conditioned graph purity are supplementary controls.',
                   display_top_types=top)
    json_write(manifest,output/'analysis_manifest.json')
    for source in [__file__,'diagnose_patient_assay_coverage.py','benchmark_patient_institutions.py']:
        shutil.copy2(source,output/Path(source).name)
    print('Complete:',output,flush=True)


if __name__=='__main__':main()
