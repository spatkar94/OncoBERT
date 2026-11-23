"""Matched-cohort summary of four final representations; excludes shuffled control.

Decoding predicts metadata from embeddings on held-out patients. Variance
association instead predicts embedding vectors from each covariate separately.
These quantities answer different questions; variance fractions are non-additive.
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
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler

from diagnostics_utils import checksum, json_write, eta_squared, regression
from institution_metrics import classification_metrics


MODELS=('esm2','genept','string','gat')
LABELS=('ESM2','GenePT','STRING','MutationProjector')
COLORS=('#3572b0','#dd8452','#55a868','#b46ab5')


def decode_categories(x, frame, output, name):
    train=frame.split.to_numpy()=='train';test=frame.split.to_numpy()=='test'
    scale=StandardScaler().fit(x[train].astype(np.float64))
    train_x=scale.transform(x[train].astype(np.float64));test_x=scale.transform(x[test].astype(np.float64))
    report={}
    for target in ('institution','assay'):
        values=frame[target].to_numpy();classes=np.unique(values)
        model=RidgeClassifier(alpha=1,class_weight='balanced',solver='cholesky')
        model.fit(train_x,values[train]);prediction=model.predict(test_x)
        scores,matrix,recall=classification_metrics(values[test],prediction,classes)
        scores.update(train_classes=len(np.unique(values[train])),test_classes=len(np.unique(values[test])),
                      unseen_test_patients=int((~np.isin(values[test],model.classes_)).sum()),
                      train_patients=int(train.sum()),test_patients=int(test.sum()))
        report[target]=scores
        pd.DataFrame(matrix,index=pd.Index(classes,name='true_label'),columns=classes).to_csv(
            output/f'{name}_{target}_confusion.csv')
        pd.DataFrame(dict(patient_id=frame.index.to_numpy()[test],truth=values[test],prediction=prediction)).to_csv(
            output/f'{name}_{target}_test_predictions.csv.gz',index=False)
    return report


def purity(graph, values):
    _,counts=np.unique(values,return_counts=True)
    chance=float(np.sum(counts*(counts-1))/(len(values)*(len(values)-1)))
    raw=float(graph.data[values[graph.row]==values[graph.col]].sum(dtype=np.float64)/graph.data.sum(dtype=np.float64))
    return dict(raw=raw,chance=chance,adjusted=(raw-chance)/(1-chance))


def continuous_variance(centered, target):
    """Trace-weighted R² of embedding ~ intercept + one continuous covariate."""
    y=np.asarray(target,dtype=np.float64);y=y-y.mean()
    covariance=y@centered
    explained=float(covariance@covariance/(y@y))
    total=float(np.sum(centered*centered))
    return explained/total


def figure(table, output, models=MODELS, labels=LABELS, colors=COLORS):
    decoding=relocated_json_loads((output/f'{models[0]}_diagnostics.json').read_text())
    institution_classes=decoding['institution_decoder']['test_classes']
    assay_classes=decoding['assay_decoder']['test_classes']
    fig,axes=plt.subplots(2,2,figsize=(15,11))
    panels=[('A','Held-out balanced accuracy',
             [f'Institution\n({institution_classes} test classes)',f'Assay\n({assay_classes} test classes)'],
             ['institution_balanced_accuracy','assay_balanced_accuracy'],100,'Balanced accuracy (%)'),
            ('B','Chance-adjusted neighborhood purity',
             ['Institution','Assay'],['institution_adjusted_purity','assay_adjusted_purity'],1,'Adjusted purity'),
            ('C','Held-out decoding from embeddings',
             ['log(1 + TMB)','Assay coverage'],['tmb_decoding_r2','coverage_decoding_r2'],1,'Test R²'),
            ('D','Embedding variance explained by each covariate',
             ['log(1 + TMB)','Assay coverage','Institution'],
             ['tmb_embedding_variance','coverage_embedding_variance','institution_embedding_variance'],100,'Embedding variance (%)')]
    width=.72/len(models)
    for ax,(letter,title,groups,columns,multiplier,ylabel) in zip(axes.flat,panels):
        centers=np.arange(len(groups))
        for i,(name,color) in enumerate(zip(models,colors)):
            values=table.loc[name,columns].to_numpy(dtype=float)*multiplier
            positions=centers+(i-(len(models)-1)/2)*width
            ax.bar(positions,values,width=width*.92,color=color)
            for px,value in zip(positions,values):
                ax.text(px,value+(1.2 if multiplier==100 else .012),
                        f'{value:.1f}' if multiplier==100 else (f'{value:.4f}' if .9995<value<1 else f'{value:.3f}'),
                        ha='center',va='bottom',fontsize=8)
        ax.set_xticks(centers,groups,fontsize=11)
        ax.set_ylabel(ylabel,fontsize=11)
        ax.set_title(f'{letter}  {title}',loc='left',fontsize=12,pad=12)
        ax.set_ylim(0,108 if multiplier==100 else 1.08)
        ax.set_yticks(np.linspace(0,100 if multiplier==100 else 1,6))
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle(f'Representation diagnostics · {int(table.iloc[0].patients):,} matched single-assay patients',fontsize=17,y=.995)
    fig.legend(handles=[Patch(color=color,label=label) for color,label in zip(colors,labels)],
               loc='upper center',bbox_to_anchor=(.5,.965),ncol=len(models),frameon=False,fontsize=11)
    fig.text(.5,.018,f'A/C: fixed linear decoders fitted on {int(table.iloc[0].train_patients):,} training patients; '
             f'{int(table.iloc[0].test_patients):,} held-out test patients for evaluation. '
             'B: saved patient PCA affinity graphs.\n'
             'D: descriptive variance association across all patients; separate univariate OLS for continuous covariates and categorical η² for institution.\n'
             'All 256 embedding dimensions; TMB uses log(1 + supplied TMB); coverage counts assay-listed genes in the 1,439-gene vocabulary. '
             'Variance fractions overlap and must not be added.',ha='center',fontsize=9,linespacing=1.5)
    fig.tight_layout(rect=(0,.105,1,.92),h_pad=3,w_pad=3)
    for ext in ('png','pdf','svg'):
        fig.savefig(output/f'merged_representation_diagnostics.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--covariates',type=Path,default=Path('experiments/patient_assay_coverage_tmb'))
    p.add_argument('--output',type=Path,default=Path('experiments/merged_representation_diagnostics'))
    args=p.parse_args();output=args.output;output.mkdir(parents=True,exist_ok=True)
    ids=pd.read_csv(args.covariates/'common_diagnostic_patients.csv.gz').patient_id.to_numpy()
    frame=pd.read_csv(args.covariates/'sequence_patient_covariates.csv.gz',index_col=0).loc[ids]
    frame=frame.loc[frame.assay_count==1].copy();ids=frame.index.to_numpy()
    assert len(ids)==181019 and frame[['log1p_tmb','coverage_genes','institution','assay']].notna().all().all()
    assert set(frame.index[frame.split=='train']).isdisjoint(frame.index[frame.split=='test'])
    frame.to_csv(output/'benchmark_patient_covariates.csv.gz')
    source_manifest=relocated_json_loads(Path('experiments/patient_institution_benchmark/analysis_manifest.json').read_text())
    paths={name:Path(source_manifest['input_h5ad'][name]['path']) for name in MODELS}
    rows=[]
    for name in MODELS:
        print('Merged diagnostics:',name,flush=True)
        data=ad.read_h5ad(paths[name]);indices=data.obs_names.get_indexer(ids);assert (indices>=0).all()
        assert np.array_equal(data.obs.loc[ids,'split'].astype(str).to_numpy(),frame.split.to_numpy())
        x=np.asarray(data.X)[indices]
        categorical=decode_categories(x,frame,output,name)
        tmb=regression(x,frame,frame.log1p_tmb.to_numpy())
        coverage=regression(x,frame,frame.coverage_genes.to_numpy())
        graph=data.obsp['connectivities'][indices][:,indices].tocoo()
        inst_purity=purity(graph,frame.institution.to_numpy());assay_purity=purity(graph,frame.assay.to_numpy())
        centered=x.astype(np.float64);centered-=centered.mean(0)
        tmb_variance=continuous_variance(centered,frame.log1p_tmb)
        raw_tmb_variance=continuous_variance(centered,frame.tmb_per_mb)
        coverage_variance=continuous_variance(centered,frame.coverage_genes)
        institution_variance,_=eta_squared(x,frame.institution.to_numpy())
        result=dict(model=name,patients=len(ids),train_patients=int((frame.split=='train').sum()),
                    test_patients=int((frame.split=='test').sum()),
                    institution_balanced_accuracy=categorical['institution']['balanced_accuracy'],
                    assay_balanced_accuracy=categorical['assay']['balanced_accuracy'],
                    institution_adjusted_purity=inst_purity['adjusted'],assay_adjusted_purity=assay_purity['adjusted'],
                    tmb_decoding_r2=tmb['test_r2'],coverage_decoding_r2=coverage['test_r2'],
                    tmb_embedding_variance=tmb_variance,coverage_embedding_variance=coverage_variance,
                    institution_embedding_variance=institution_variance,raw_tmb_embedding_variance=raw_tmb_variance)
        rows.append(result)
        json_write(dict(summary=result,institution_decoder=categorical['institution'],assay_decoder=categorical['assay'],
                        tmb_decoder=tmb,coverage_decoder=coverage,institution_purity=inst_purity,assay_purity=assay_purity),
                   output/f'{name}_diagnostics.json')
        print(result,flush=True);del data,x,centered,graph
    table=pd.DataFrame(rows).set_index('model');table.to_csv(output/'merged_diagnostics.csv')
    figure(table,output)
    manifest=dict(models=list(MODELS),patients=len(frame),institutions=frame.institution.nunique(),assays=frame.assay.nunique(),
                   cohort='Same complete, sample-matched single-assay patients in all four encoders. Shuffled control excluded.',
                   decoders='Fixed alpha=1, train-only StandardScaler, float64. Balanced RidgeClassifier for both categorical targets; Ridge for continuous targets. No parameter tuning.',
                   variance='In-sample descriptive trace R² for embedding ~ intercept + one continuous covariate; categorical between-institution/total embedding sums of squares (η²). Independent associations, not additive variance partitioning or causal effects.',
                   variance_formula='Continuous: ||y_centered.T @ X_centered||² / (||y_centered||² * ||X_centered||_F²).',
                   tmb='Patient mean supplied TMB, scaled x1e6 based on supplied-bin audit, transformed log1p. Raw-rate variance sensitivity also exported.',
                   coverage='Patient mean assay-listed gene count within 1439-gene vocabulary; not read depth.',
                   source_sha256=checksum(__file__),covariates_sha256=checksum(args.covariates/'sequence_patient_covariates.csv.gz'),
                   input_h5ad={name:dict(path=str(path),sha256=checksum(path)) for name,path in paths.items()})
    json_write(manifest,output/'analysis_manifest.json')
    for source in [__file__,'diagnose_patient_assay_coverage.py','benchmark_patient_institutions.py']:
        shutil.copy2(source,output/Path(source).name)
    print('Complete:',output,flush=True)


if __name__=='__main__':main()
