"""Institution segregation of existing patient representations and UMAPs.

All models use the same common cohort and original train/test patient split.
Institution is validated from GENIE sample IDs, not normalized patient IDs.
No encoder or projection is refit. Associations can reflect assay/case mix.
"""
import argparse
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
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from diagnose_patient_assay_coverage import MODELS, TITLES, checksum, json_write, eta_squared


def classification_metrics(truth, predicted, labels):
    matrix = confusion_matrix(truth, predicted, labels=labels)
    counts = matrix.sum(1)
    recall = np.divide(matrix.diagonal(), counts, out=np.zeros(len(labels), dtype=float), where=counts>0)
    return dict(accuracy=float((truth==predicted).mean()),
                balanced_accuracy=float(recall[counts>0].mean())), matrix, recall


def decode(x, frame, output, name):
    train = frame.split.to_numpy() == 'train'
    test = frame.split.to_numpy() == 'test'
    labels = np.sort(frame.institution.unique())
    scaler = StandardScaler().fit(x[train].astype(np.float64))
    train_x = scaler.transform(x[train].astype(np.float64))
    test_x = scaler.transform(x[test].astype(np.float64))
    train_y, test_y = frame.institution.to_numpy()[train], frame.institution.to_numpy()[test]
    centers = np.stack([train_x[train_y==label].mean(0) for label in labels])
    assert all((train_y==label).any() for label in labels)
    distances = -2 * test_x @ centers.T + (centers**2).sum(1)
    centroid_prediction = labels[distances.argmin(1)]
    # A fixed linear decoder supplements the simple centroid rule, which can
    # miss institutions with several assay-specific embedding modes.
    ridge = RidgeClassifier(alpha=1, class_weight='balanced', solver='cholesky')
    ridge.fit(train_x, train_y)
    ridge_prediction = ridge.predict(test_x)
    majority = pd.Series(train_y).value_counts().idxmax()
    baseline, _, _ = classification_metrics(test_y, np.repeat(majority, len(test_y)), labels)
    recalls = pd.DataFrame({'institution':labels,
                            'train_patients':[(train_y==label).sum() for label in labels],
                            'test_patients':[(test_y==label).sum() for label in labels]})
    report = dict(train_patients=int(train.sum()), test_patients=int(test.sum()),
                  train_institutions=len(np.unique(train_y)), test_institutions=len(np.unique(test_y)),
                  majority_institution=majority, majority_baseline=baseline)
    for decoder, prediction in [('nearest_centroid',centroid_prediction),('balanced_ridge',ridge_prediction)]:
        scores, matrix, recall = classification_metrics(test_y, prediction, labels)
        report[decoder] = scores
        recalls[f'{decoder}_recall'] = recall
        pd.DataFrame(matrix,index=pd.Index(labels,name='true_institution'),columns=labels).to_csv(
            output/f'{name}_{decoder}_confusion.csv')
    recalls.insert(0,'model',name)
    recalls.to_csv(output/f'{name}_institution_recalls.csv',index=False)
    pd.DataFrame(dict(patient_id=frame.index.to_numpy()[test],true_institution=test_y,
                       nearest_centroid=centroid_prediction,balanced_ridge=ridge_prediction)).to_csv(
        output/f'{name}_test_predictions.csv.gz',index=False)
    return report, recalls


def segregation(data, frame, ids):
    indices=data.obs_names.get_indexer(ids)
    assert (indices>=0).all()
    x=np.asarray(data.X)[indices]
    labels=frame.institution.to_numpy()
    eta,adjusted=eta_squared(x,labels)
    graph=data.obsp['connectivities'][indices][:,indices].tocoo()
    raw=float(graph.data[labels[graph.row]==labels[graph.col]].sum(dtype=np.float64)/graph.data.sum(dtype=np.float64))
    _,counts=np.unique(labels,return_counts=True)
    chance=float(np.sum(counts*(counts-1))/(len(labels)*(len(labels)-1)))
    return dict(patients=len(ids),institution_eta_squared=eta,institution_df_adjusted_eta_squared=adjusted,
                weighted_neighbor_institution_purity=raw,random_neighbor_institution_purity=chance,
                adjusted_neighbor_institution_purity=(raw-chance)/(1-chance)),x


def overlay(covariates, variants, colors, output, filename):
    fig,axes=plt.subplots(1,len(variants),figsize=(7*len(variants),8),squeeze=False)
    for ax,name in zip(axes.flat,variants):
        frame=covariates[name]
        xy=frame[['UMAP1','UMAP2']].to_numpy()
        order=np.random.default_rng(42).permutation(len(frame))
        ax.scatter(xy[order,0],xy[order,1],c=frame.institution.map(colors).to_numpy()[order],
                   s=.7,alpha=.7,linewidths=0,rasterized=True)
        ax.set_title(f'{TITLES[name]}\nInstitution · {len(frame):,} patients',fontsize=12)
        ax.set_xlabel('UMAP 1');ax.set_ylabel('UMAP 2')
        ax.spines[['top','right']].set_visible(False)
    handles=[Line2D([0],[0],marker='o',linestyle='',color=color,label=label,markersize=5) for label,color in colors.items()]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.035),ncol=5,fontsize=9)
    fig.suptitle('Institution labels on existing patient UMAPs',fontsize=15,y=.99)
    fig.text(.5,.012,'All original plotted patients retained. Shared institution colors; independent projection axes.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.19,1,.94))
    for extension in ('png','pdf'):
        fig.savefig(output/f'{filename}.{extension}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def plots(table, recalls, output):
    fig,axes=plt.subplots(2,2,figsize=(13,10))
    labels=['ESM2','GenePT','STRING','Shuffled\ncontrol','GAT']
    colors=['#3572b0','#dd8452','#55a868','#909090','#b46ab5']
    panels=[('institution_eta_squared','Embedding variance associated with institution'),
            ('adjusted_neighbor_institution_purity','Institution neighbor purity, adjusted for prevalence'),
            ('centroid_test_balanced_accuracy','Held-out nearest-centroid balanced accuracy'),
            ('ridge_test_balanced_accuracy','Held-out balanced-ridge balanced accuracy')]
    for ax,(column,title) in zip(axes.flat,panels):
        values=table[column].to_numpy()
        ax.bar(range(len(table)),values,color=colors)
        ax.set_xticks(range(len(table)),labels)
        ax.set_ylim(0,1);ax.set_title(title,fontsize=11)
        if 'accuracy' in column:
            ax.axhline(.05,color='#444444',ls='--',lw=1,label='Chance / majority balanced accuracy = 0.05')
            ax.legend(loc='upper left',fontsize=7)
        for i,v in enumerate(values):
            ax.text(i,v+.015,f'{v:.3f}',ha='center',fontsize=10)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f'Institution segregation · {int(table.iloc[0].patients):,} common patients · 20 institutions',fontsize=15)
    fig.text(.5,.015,'Global metrics/decoders use all 256 dimensions; neighbor purity uses saved PCA affinities.\n'
             'Decoders fit on training patients only; fixed settings; original test patients for evaluation. Association may reflect assay and cancer-type differences.',
             ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.94))
    for extension in ('png','pdf'):
        fig.savefig(output/f'institution_segregation_diagnostics.{extension}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    matrix=recalls.pivot(index='model',columns='institution',values='balanced_ridge_recall').loc[list(MODELS)]
    fig,ax=plt.subplots(figsize=(17,5))
    artist=ax.imshow(matrix.to_numpy(),vmin=0,vmax=1,cmap='viridis',aspect='auto')
    ax.set_yticks(range(len(matrix)),labels)
    ax.set_xticks(range(len(matrix.columns)),matrix.columns,rotation=45,ha='right')
    for row in range(len(matrix)):
        for col in range(len(matrix.columns)):
            value=matrix.iloc[row,col]
            ax.text(col,row,f'{value:.2f}',ha='center',va='center',fontsize=8,color='white' if value<.55 else 'black')
    fig.colorbar(artist,ax=ax,label='Held-out recall')
    ax.set_title('Institution decoding: test recall per institution · balanced ridge')
    fig.tight_layout()
    for extension in ('png','pdf'):
        fig.savefig(output/f'institution_test_recall_heatmap.{extension}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--covariates',type=Path,default=Path('experiments/patient_assay_coverage_tmb'))
    parser.add_argument('--output',type=Path,default=Path('experiments/patient_institution_benchmark'))
    args=parser.parse_args();output=args.output;output.mkdir(parents=True,exist_ok=True)
    samples=pd.read_csv(args.covariates/'sample_covariates.csv.gz')
    assert (samples.sample_id.str.split('-',n=2).str[1]==samples.institution).all()
    assert samples.groupby('patient_id').institution.nunique().max()==1
    ids=pd.read_csv(args.covariates/'common_diagnostic_patients.csv.gz').patient_id.to_numpy()
    metadata=pd.read_csv(args.covariates/'sequence_patient_covariates.csv.gz',index_col=0).loc[ids]
    assert metadata.institution.nunique()==20 and not metadata.institution.isna().any()
    assert set(metadata.index[metadata.split=='train']).isdisjoint(metadata.index[metadata.split=='test'])
    metadata[['institution','assay','assay_count','split']].to_csv(output/'benchmark_patient_labels.csv.gz')
    pd.crosstab(metadata.institution,metadata.split).to_csv(output/'institution_split_counts.csv')
    paths={name:Path('experiments/genie_v20/patient_umap_leiden_r0p5_cls_pca90')/f'{name}_patients.h5ad' for name in MODELS[:-1]}
    paths['gat']=Path('experiments/mutation_projector_four_state_v20/patient_umap_leiden_r0p5_epoch313/gat_patients.h5ad')
    results=[];recalls=[];sensitivity=[]
    for name in MODELS:
        print('Institution benchmark:',name,flush=True)
        data=ad.read_h5ad(paths[name])
        assert np.array_equal(data.obs.loc[ids,'split'].astype(str).to_numpy(),metadata.split.to_numpy())
        metrics,x=segregation(data,metadata,ids)
        decoder,recall=decode(x,metadata,output,name)
        metrics.update(model=name,centroid_test_accuracy=decoder['nearest_centroid']['accuracy'],
                       centroid_test_balanced_accuracy=decoder['nearest_centroid']['balanced_accuracy'],
                       ridge_test_accuracy=decoder['balanced_ridge']['accuracy'],
                       ridge_test_balanced_accuracy=decoder['balanced_ridge']['balanced_accuracy'],
                       majority_test_accuracy=decoder['majority_baseline']['accuracy'],
                       majority_test_balanced_accuracy=decoder['majority_baseline']['balanced_accuracy'],
                       train_patients=decoder['train_patients'],test_patients=decoder['test_patients'])
        results.append(metrics);recalls.append(recall)
        json_write(dict(metrics=metrics,decoders=decoder),output/f'{name}_institution_diagnostics.json')
        single=metadata.assay_count.to_numpy()==1
        matched,_=segregation(data,metadata.loc[single],ids[single])
        matched['model']=name;sensitivity.append(matched)
        print(metrics,flush=True)
        del data,x
    table=pd.DataFrame(results);table.to_csv(output/'institution_benchmark.csv',index=False)
    per_class=pd.concat(recalls,ignore_index=True);per_class.to_csv(output/'institution_test_recalls.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(output/'single_assay_institution_sensitivity.csv',index=False)
    plots(table,per_class,output)
    covariates={name:pd.read_csv(args.covariates/f'{name}_umap_covariates.csv.gz',index_col=0) for name in MODELS}
    institutions=sorted(metadata.institution.unique())
    palette=dict(zip(institutions,plt.colormaps['tab20'].colors))
    for name in MODELS:
        overlay(covariates,[name],palette,output,f'{name}_umap_institution')
    overlay(covariates,['esm2','genept','string'],palette,output,'primary_variants_umap_institution')
    overlay(covariates,['shuffled','gat'],palette,output,'control_and_comparator_umap_institution')
    manifest=dict(common_patients=len(ids),institutions=institutions,source_sha256=checksum(__file__),
                   cohort_sha256=checksum(args.covariates/'common_diagnostic_patients.csv.gz'),
                   sample_covariates_sha256=checksum(args.covariates/'sample_covariates.csv.gz'),
                   input_h5ad={name:dict(path=str(path.resolve()),sha256=checksum(path)) for name,path in paths.items()},
                   cohort_definition='Same 185350 complete, sample-matched patients as the assay benchmark; all have one institution, including multi-assay patients.',
                   sensitivity='Institution eta-squared and neighbor purity also evaluated on the 181019 single-assay patients used for assay metrics.',
                   decoder='All 256 dimensions; float64; train-only StandardScaler. Nearest centroid plus RidgeClassifier alpha=1, class_weight=balanced, solver=cholesky. No parameter tuning.',
                   graph='Existing weighted patient PCA affinity graph, restricted to the cohort; prevalence-adjusted same-institution purity.',
                   label_validation='Assay institution prefix equals GENIE sample-ID institution prefix for every sample; canonical patient IDs can have institution prefixes removed.',
                   interpretation='Institutions use different assays and enroll different cancer populations. This measures association and decodability, not an isolated causal institution/batch effect.',
                   institution_palette={key:list(color) for key,color in palette.items()})
    json_write(manifest,output/'analysis_manifest.json')
    shutil.copy2(__file__,output/'analysis_source.py')
    shutil.copy2('diagnose_patient_assay_coverage.py',output/'diagnose_patient_assay_coverage.py')
    print('Complete:',output,flush=True)


if __name__=='__main__':
    main()
