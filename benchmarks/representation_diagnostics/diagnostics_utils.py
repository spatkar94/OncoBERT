"""Overlay assay breadth and supplied GENIE TMB on existing patient UMAPs.

No encoder, UMAP, or Leiden model is refit. Descriptive associations and simple
train-to-test decoding diagnostics use the same complete, sample-matched patient
cohort across encoders. Assay association is not a causal estimate of batch effects.
"""
from experiment_paths import relocated_json_loads
import argparse
import hashlib
import json
from pathlib import Path
import os
import shutil

import anndata as ad
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


MODELS = ('esm2', 'genept', 'string', 'shuffled', 'gat')
TITLES = dict(esm2='ESM2 network', genept='GenePT network', string='STRING network',
              shuffled='Shuffled-order control', gat='GAT set comparator · epoch 313')


def checksum(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def json_write(value, path):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def one_label(values):
    values = np.unique(values)
    return str(values[0]) if len(values) == 1 else 'Multiple assays'


def patient_metadata(samples):
    """Match the averaging cohort used to construct each encoder's patient vectors."""
    group = samples.groupby('patient_id', sort=True)
    frame = group.agg(n_samples=('sample_id', 'size'), coverage_n=('coverage_genes', 'count'),
                      coverage_genes=('coverage_genes', 'mean'), panel_genes=('panel_genes', 'mean'),
                      tmb_n=('tmb_per_mb', 'count'), tmb_per_mb=('tmb_per_mb', 'mean'),
                      assay_count=('assay', 'nunique'), split=('split', 'first'))
    # Unknown panels remain missing, including patients with partial known coverage.
    frame.loc[frame.coverage_n != frame.n_samples, ['coverage_genes', 'panel_genes']] = np.nan
    frame['assay'] = group.assay.agg(one_label)
    frame['institution'] = group.institution.agg(lambda x: x.iloc[0] if x.nunique() == 1 else 'Multiple institutions')
    frame['cancer_type'] = group.cancer_type.agg(lambda x: x.iloc[0] if x.nunique() == 1 else 'Multiple cancer types')
    frame['coverage_fraction'] = frame.coverage_genes / 1439
    frame['log1p_tmb'] = np.log1p(frame.tmb_per_mb)
    return frame


def metadata(args, output):
    original = np.load(args.sequence_root / 'mutations.npz')
    assert len(original['genes']) == 1439
    clinical = pd.read_csv(args.genie / 'data_clinical_sample.txt', sep='\t', comment='#', dtype=str)
    assert not clinical.SAMPLE_ID.duplicated().any()
    clinical = clinical.set_index('SAMPLE_ID')
    tmb_path = args.genie / 'tmb_20.0-public.tsv'
    tmb = pd.read_csv(tmb_path, sep='\t')
    assert not tmb.SAMPLE_ID.duplicated().any()
    assert np.isfinite(tmb.tmb).all() and (tmb.tmb >= 0).all()
    # The file has no units header. The 1e6 conversion is inferred, and audited
    # against EVERY supplied TMB bin, rather than assuming these rates are per Mb.
    converted = tmb.tmb.to_numpy() * 1e6
    bins = tmb.tmb_bin.to_numpy()
    consistent = (((bins == 'Low (<2)') & (converted <= 2 + 1e-6)) |
                  ((bins == 'Mid (2-16)') & (converted >= 2 - 1e-6) & (converted <= 16 + 1e-6)) |
                  ((bins == 'High (>16)') & (converted >= 16 - 1e-6)))
    if not consistent.all():
        raise ValueError('TMB conversion does not agree with all supplied bins')
    tmb['tmb_per_mb'] = converted
    tmb = tmb.set_index('SAMPLE_ID')
    audit_path = args.gat_root / 'profile_states.npz.audit.json'
    audit = relocated_json_loads(audit_path.read_text())
    sizes = {k: v['profiled_genes_in_vocabulary'] for k, v in audit['assays'].items()}
    full_sizes = {k: v['genes_in_definition'] for k, v in audit['assays'].items()}
    clinical = clinical.loc[original['samples']]
    frame = pd.DataFrame(dict(sample_id=original['samples'], patient_id=original['patients'],
                              split=original['split'], assay=clinical.SEQ_ASSAY_ID.to_numpy(),
                              cancer_type=clinical.CANCER_TYPE.fillna('Unknown').to_numpy()))
    frame['institution'] = frame.assay.str.split('-', n=1).str[0]
    frame['coverage_genes'] = frame.assay.map(sizes)
    frame['panel_genes'] = frame.assay.map(full_sizes)
    frame['tmb_raw'] = frame.sample_id.map(tmb.tmb)
    frame['tmb_per_mb'] = frame.sample_id.map(tmb.tmb_per_mb)
    frame['tmb_bin'] = frame.sample_id.map(tmb.tmb_bin)
    assert frame.tmb_per_mb.notna().all()
    with np.load(args.gat_root / 'profile_states.npz') as archive:
        retained = frame.sample_id.isin(archive['samples'])
        assert np.array_equal(frame.loc[retained, 'sample_id'].to_numpy(), archive['samples'])
        assert np.array_equal(frame.loc[retained, 'assay'].to_numpy(), archive['assays'])
    result = {'sequence': patient_metadata(frame), 'gat': patient_metadata(frame.loc[retained])}
    frame.to_csv(output / 'sample_covariates.csv.gz', index=False)
    frame.loc[frame.tmb_per_mb > 1000].to_csv(output / 'supplied_tmb_above_1000_per_mb.csv', index=False)
    for key, value in result.items():
        value.to_csv(output / f'{key}_patient_covariates.csv.gz')
    summary = dict(original_samples=len(frame), original_patients=len(result['sequence']),
                   gat_samples=int(retained.sum()), gat_patients=len(result['gat']),
                   tmb_matched_samples=int(frame.tmb_per_mb.notna().sum()),
                   missing_panel_samples=int(frame.coverage_genes.isna().sum()),
                   patients_with_any_missing_panel=int(result['sequence'].coverage_genes.isna().sum()),
                   missing_panels=sorted(frame.loc[frame.coverage_genes.isna(), 'assay'].unique().tolist()),
                   tmb_conversion_factor=1e6, tmb_units='inferred mutations/Mb',
                   tmb_unit_evidence='All supplied Low (<2), Mid (2-16), High (>16) bins agree after multiplying raw values by 1e6.',
                   tmb_file_rows=len(tmb), tmb_bin_agreement=float(consistent.mean()),
                   supplied_samples_tmb_above_1000=int((frame.tmb_per_mb > 1000).sum()),
                   supplied_sample_tmb_quantiles={str(q):float(frame.tmb_per_mb.quantile(q)) for q in (0,.5,.95,.99,.999,1)},
                   patient_aggregation='Unweighted arithmetic mean across the exact samples averaged for each encoder; assay identity only when all samples use one assay.',
                   coverage_definition='Count of assay-listed genes within the 1439-gene model vocabulary; not state!=0, mutation burden, read depth, or captured megabases.',
                   coverage_sha256=checksum(audit_path), tmb_sha256=checksum(tmb_path),
                   clinical_sha256=checksum(args.genie / 'data_clinical_sample.txt'))
    json_write(summary, output / 'metadata_audit.json')
    return result, summary


def eta_squared(x, labels):
    """Between-label / total centered sum of squares across all vector dimensions."""
    _, codes = np.unique(labels, return_inverse=True)
    counts = np.bincount(codes)
    x = np.asarray(x, dtype=np.float64)
    sums = np.zeros((len(counts), x.shape[1]))
    np.add.at(sums, codes, x)
    mean = x.mean(0)
    between = np.sum(counts[:, None] * (sums / counts[:, None] - mean)**2)
    total = np.sum((x - mean)**2)
    value = float(between / total)
    adjusted = 1 - (1 - value) * (len(x) - 1) / (len(x) - len(counts))
    return value, float(adjusted)


def regression(x, frame, target):
    train = (frame.split == 'train').to_numpy() & np.isfinite(target)
    test = (frame.split == 'test').to_numpy() & np.isfinite(target)
    # Contextual embedding dimensions can be strongly collinear. Accumulate
    # regression in float64 so Cholesky solves are accurate at fixed alpha=1.
    train_x = x[train].astype(np.float64)
    test_x = x[test].astype(np.float64)
    scaler = StandardScaler().fit(train_x)
    model = Ridge(alpha=1, solver='cholesky').fit(scaler.transform(train_x), target[train])
    predicted = model.predict(scaler.transform(test_x))
    return dict(test_r2=float(r2_score(target[test], predicted)),
                test_spearman=float(spearmanr(target[test], predicted).statistic),
                train_patients=int(train.sum()), test_patients=int(test.sum()))


def assay_decoding(x, frame):
    """A fixed nearest-centroid decoder; no classifier hyperparameter selection."""
    train = (frame.split == 'train').to_numpy()
    test = (frame.split == 'test').to_numpy()
    labels, codes = np.unique(frame.assay.to_numpy(), return_inverse=True)
    counts = np.bincount(codes[train], minlength=len(labels))
    sums = np.zeros((len(labels), x.shape[1]), dtype=np.float64)
    np.add.at(sums, codes[train], x[train])
    centers = sums / np.maximum(counts, 1)[:, None]
    scaler = StandardScaler().fit(x[train].astype(np.float64))
    center_scaled = scaler.transform(centers)
    test_scaled = scaler.transform(x[test].astype(np.float64))
    available = np.flatnonzero(counts)
    centers_available = center_scaled[available]
    distance = -2 * test_scaled @ centers_available.T + (centers_available**2).sum(1)
    prediction = available[distance.argmin(1)]
    truth = codes[test]
    test_counts = np.bincount(truth, minlength=len(labels))
    correct = np.bincount(truth, weights=(prediction == truth), minlength=len(labels))
    support = test_counts > 0
    majority = counts.argmax()
    report = dict(test_accuracy=float(np.mean(prediction == truth)),
                  test_balanced_accuracy=float(np.mean(correct[support] / test_counts[support])),
                  majority_baseline_accuracy=float(np.mean(truth == majority)),
                  majority_baseline_balanced_accuracy=float((test_counts[majority] > 0) / support.sum()),
                  train_assays=int((counts > 0).sum()), test_assays=int(support.sum()),
                  test_patients=int(test.sum()), unseen_assay_test_patients=int((counts[truth] == 0).sum()))
    return report, codes, counts, centers


def within_assay_tmb(x, frame, codes, counts, centers):
    """Train-only assay centering; test SSE improvement over train assay-mean TMB."""
    target = frame.log1p_tmb.to_numpy()
    train = (frame.split == 'train').to_numpy() & (counts[codes] >= 20)
    test = (frame.split == 'test').to_numpy() & (counts[codes] >= 20)
    sums = np.bincount(codes[train], weights=target[train], minlength=len(counts))
    n = np.bincount(codes[train], minlength=len(counts))
    means = sums / np.maximum(n, 1)
    x_train = x[train] - centers[codes[train]]
    x_test = x[test] - centers[codes[test]]
    y_train = target[train] - means[codes[train]]
    y_test = target[test] - means[codes[test]]
    scaler = StandardScaler().fit(x_train)
    model = Ridge(alpha=1, solver='cholesky').fit(scaler.transform(x_train), y_train)
    pred = model.predict(scaler.transform(x_test))
    return dict(test_sse_improvement_over_assay_mean=float(1 - np.sum((pred-y_test)**2) / np.sum(y_test**2)),
                test_spearman_residual=float(spearmanr(y_test, pred).statistic),
                train_patients=int(train.sum()), test_patients=int(test.sum()), minimum_train_patients_per_assay=20)


def diagnostics(data, meta, common, name, output):
    frame = meta.loc[common]
    single = frame.assay_count.to_numpy() == 1
    indices = data.obs_names.get_indexer(common)
    x = np.asarray(data.X)[indices]
    pcs = np.asarray(data.obsm['X_pca'])[indices]
    assay_eta, adjusted_eta = eta_squared(x[single], frame.assay.to_numpy()[single])
    cancer_eta, _ = eta_squared(x, frame.cancer_type.to_numpy())
    center_eta, _ = eta_squared(x[single], frame.institution.to_numpy()[single])
    # Same labels and patients for all models; weighted graph homophily measures
    # local assay association in each saved high-dimensional affinity graph.
    single_indices = indices[single]
    graph = data.obsp['connectivities'][single_indices][:, single_indices].tocoo()
    labels = frame.assay.to_numpy()[single]
    purity = float(graph.data[labels[graph.row] == labels[graph.col]].sum(dtype=np.float64) / graph.data.sum(dtype=np.float64))
    _, n = np.unique(labels, return_counts=True)
    random = float(np.sum(n * (n-1)) / (len(labels) * (len(labels)-1)))
    assay, codes, counts, centers = assay_decoding(x[single], frame.loc[single])
    coverage = regression(x, frame, frame.coverage_genes.to_numpy())
    tmb = regression(x, frame, frame.log1p_tmb.to_numpy())
    residual_tmb = within_assay_tmb(x[single], frame.loc[single], codes, counts, centers)
    pc_rows = []
    for i in range(min(10, pcs.shape[1])):
        pc_eta, _ = eta_squared(pcs[single, i:i+1], labels)
        # Correlate residual ranks after removing assay means. This is an
        # association diagnostic, not proof of biological/cancer specificity.
        ranks = pd.DataFrame(dict(pc=rankdata(pcs[single, i]), tmb=rankdata(frame.log1p_tmb.to_numpy()[single]), assay=labels))
        residual = ranks[['pc', 'tmb']] - ranks.groupby('assay')[['pc', 'tmb']].transform('mean')
        pc_rows.append(dict(model=name, pc=i+1, variance_ratio=float(data.uns['pca']['variance_ratio'][i]),
                            assay_eta_squared=pc_eta,
                            spearman_coverage=float(spearmanr(pcs[:, i], frame.coverage_genes).statistic),
                            spearman_log1p_tmb=float(spearmanr(pcs[:, i], frame.log1p_tmb).statistic),
                            assay_adjusted_rank_tmb_correlation=float(residual.pc.corr(residual.tmb))))
    pd.DataFrame(pc_rows).to_csv(output / f'{name}_pc_associations.csv', index=False)
    full = dict(model=name, patients=len(frame), single_assay_patients=int(single.sum()),
                assay_eta_squared=assay_eta, assay_adjusted_eta_squared=adjusted_eta,
                institution_eta_squared=center_eta, cancer_type_eta_squared=cancer_eta,
                weighted_neighbor_assay_purity=purity, random_neighbor_assay_purity=random,
                adjusted_neighbor_assay_purity=(purity-random)/(1-random),
                assay_decoder=assay, coverage_decoder=coverage, log1p_tmb_decoder=tmb,
                within_assay_tmb_decoder=residual_tmb,
                variance_retained=float(np.asarray(data.uns['pca']['variance_ratio']).sum()),
                pca_dimensions=pcs.shape[1])
    json_write(full, output / f'{name}_diagnostics.json')
    flat = dict(model=name, title=TITLES[name], patients=len(frame), single_assay_patients=int(single.sum()),
                assay_eta_squared=assay_eta, assay_adjusted_eta_squared=adjusted_eta,
                institution_eta_squared=center_eta, cancer_type_eta_squared=cancer_eta,
                weighted_neighbor_assay_purity=purity, random_neighbor_assay_purity=random,
                adjusted_neighbor_assay_purity=(purity-random)/(1-random),
                assay_test_accuracy=assay['test_accuracy'], assay_test_balanced_accuracy=assay['test_balanced_accuracy'],
                assay_majority_baseline_accuracy=assay['majority_baseline_accuracy'],
                coverage_test_r2=coverage['test_r2'], coverage_test_spearman=coverage['test_spearman'],
                log1p_tmb_test_r2=tmb['test_r2'], log1p_tmb_test_spearman=tmb['test_spearman'],
                within_assay_tmb_test_sse_improvement=residual_tmb['test_sse_improvement_over_assay_mean'],
                within_assay_tmb_test_spearman=residual_tmb['test_spearman_residual'])
    return flat, pc_rows


def overlay_plot(loaded, metadata_by_model, variants, output, filename, coverage_max, tmb_cap, top_assays):
    fig, axes = plt.subplots(3, len(variants), figsize=(7*len(variants), 16), squeeze=False)
    colors = dict(zip(top_assays, plt.colormaps['tab20'].colors[:len(top_assays)]))
    colors.update({'Other assays':'#c2c7ce', 'Multiple assays':'#111827', 'Unknown panel':'#ff8c00'})
    coverage_norm = Normalize(0, coverage_max)
    tmb_norm = Normalize(0, np.log1p(tmb_cap), clip=True)
    for col, name in enumerate(variants):
        d = loaded[name]; frame = metadata_by_model[name].loc[d.obs_names]
        xy = np.asarray(d.obsm['X_umap'])
        order = np.random.default_rng(42).permutation(len(frame))
        for row, field in enumerate(('coverage_genes', 'tmb_per_mb', 'assay')):
            ax = axes[row, col]
            if row < 2:
                values = frame[field].to_numpy()
                valid = np.isfinite(values)
                ax.scatter(xy[~valid,0], xy[~valid,1], s=.6, c='#aeb4bd', alpha=.5, linewidths=0, rasterized=True)
                idx = order[valid[order]]
                shown = values if row == 0 else np.log1p(values)
                artist = ax.scatter(xy[idx,0], xy[idx,1], c=shown[idx], cmap='viridis' if row == 0 else 'magma',
                                    norm=coverage_norm if row == 0 else tmb_norm,
                                    s=.7, alpha=.75, linewidths=0, rasterized=True)
                bar = fig.colorbar(artist, ax=ax, fraction=.045, pad=.02, extend='max' if row == 1 else 'neither')
                if row == 0:
                    bar.set_label('Assay-listed genes / 1,439', fontsize=9)
                else:
                    ticks = [v for v in (0,1,5,20,100,500) if v <= tmb_cap]
                    bar.set_ticks(np.log1p(ticks), labels=[str(v) for v in ticks])
                    bar.set_label('Mean supplied TMB (inferred mut/Mb; log scale)', fontsize=9)
                subtitle = ('Assay gene coverage' if row == 0 else 'GENIE-supplied TMB')
                subtitle += f' · missing {int((~valid).sum()):,}'
            else:
                labels = frame.assay.where(frame.assay.isin(top_assays), 'Other assays')
                labels.loc[frame.assay_count > 1] = 'Multiple assays'
                labels.loc[frame.coverage_genes.isna()] = 'Unknown panel'
                point_colors = labels.map(colors).to_numpy()
                ax.scatter(xy[order,0], xy[order,1], c=point_colors[order], s=.7, alpha=.7, linewidths=0, rasterized=True)
                subtitle = 'Assay identity · top 12 + other/multiple/unknown'
            ax.set_title(f'{TITLES[name]}\n{subtitle}\n{len(frame):,} patients', fontsize=11)
            ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
            ax.spines[['top','right']].set_visible(False)
    handles = [Line2D([0],[0],marker='o',linestyle='',color=color,markersize=5,label=label) for label,color in colors.items()]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5,.025), ncol=4 if len(variants)==1 else 5, fontsize=8)
    fig.suptitle('Existing patient UMAPs: assay coverage, supplied TMB, and assay identity', fontsize=15, y=.995)
    fig.text(.5,.006,f'Coverage is gene-panel breadth, not read depth. Patient covariates average the same samples as each embedding.\n'
             f'Shared scales; TMB colors saturate at {tmb_cap:.1f} inferred mut/Mb (99th percentile); raw values retained. Axes are independent across encoders.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0,.11,1,.97))
    for suffix in ('png','pdf'):
        fig.savefig(output / f'{filename}.{suffix}', dpi=170, bbox_inches='tight')
    plt.close(fig)


def diagnostic_plot(table, output):
    columns = [('assay_eta_squared','Representation variance associated with assay', (0,1)),
               ('adjusted_neighbor_assay_purity','Assay neighbor purity, adjusted for prevalence', (0,1)),
               ('assay_test_balanced_accuracy','Held-out assay decoding · balanced accuracy', (0,1)),
               ('coverage_test_r2','Held-out assay-coverage decoding · R²', None),
               ('log1p_tmb_test_r2','Held-out log(1 + supplied TMB) decoding · R²', None),
               ('within_assay_tmb_test_sse_improvement','TMB prediction gain beyond assay means', None)]
    fig, axes = plt.subplots(2,3,figsize=(18,11))
    colors = ['#3572b0','#dd8452','#55a868','#909090','#b46ab5']
    labels = ['ESM2','GenePT','STRING','Shuffled\ncontrol','GAT']
    for ax,(column,title,limits) in zip(axes.flat,columns):
        values = table[column].to_numpy()
        ax.bar(np.arange(len(table)),values,color=colors)
        ax.set_xticks(np.arange(len(table)),labels,fontsize=10)
        ax.set_title(title,fontsize=11)
        ax.axhline(0,color='#333333',lw=.7)
        if limits:
            ax.set_ylim(*limits)
        else:
            ax.set_ylim(min(-.05,float(values.min())-.08),max(1.05,float(values.max())+.08))
        for i,value in enumerate(values):
            ax.text(i,value + (.015 if value>=0 else -.015),f'{value:.3f}',ha='center',
                    va='bottom' if value>=0 else 'top',fontsize=9)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f'Assay/coverage diagnostics · common sample-matched cohort ({int(table.iloc[0].patients):,} patients)',fontsize=15)
    fig.text(.5,.015,'Global metrics/decoders use all 256 dimensions; neighbor purity uses saved PCA affinities. Assay metrics use single-assay patients.\n'
             'Ridge α=1 with train-only scaling; assay decoder uses nearest training centroid. Association can reflect cancer/site mix as well as assay effects.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.07,1,.95))
    for suffix in ('png','pdf'):
        fig.savefig(output / f'assay_coverage_diagnostics.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence-root',type=Path,default=Path('experiments/genie_v20'))
    parser.add_argument('--gat-root',type=Path,default=Path('experiments/mutation_projector_four_state_v20'))
    parser.add_argument(
        '--genie', type=Path,
        default=Path(os.environ.get(
            'ONCOBERT_GENIE_ROOT',
            Path(__file__).resolve().parents[2] / 'data' / 'raw' / 'genie')))
    parser.add_argument('--output',type=Path,default=Path('experiments/patient_assay_coverage_tmb'))
    parser.add_argument('--stage',choices=['all','diagnostics','plots'],default='all')
    args=parser.parse_args();output=args.output;output.mkdir(parents=True,exist_ok=True)
    meta,audit=metadata(args,output)
    paths={name:args.sequence_root/'patient_umap_leiden_r0p5_cls_pca90'/f'{name}_patients.h5ad' for name in MODELS[:-1]}
    paths['gat']=args.gat_root/'patient_umap_leiden_r0p5_epoch313/gat_patients.h5ad'
    loaded={name:ad.read_h5ad(path) for name,path in paths.items()}
    model_meta={name:meta['gat' if name=='gat' else 'sequence'] for name in MODELS}
    common=set(meta['sequence'].index[meta['sequence'].coverage_genes.notna()])
    for name,data in loaded.items():
        frame=model_meta[name].loc[data.obs_names]
        assert np.array_equal(frame.n_samples.to_numpy(),data.obs.n_samples.to_numpy())
        assert np.array_equal(frame.split.to_numpy(),data.obs.split.astype(str).to_numpy())
        common &= set(data.obs_names)
    common=np.array(sorted(common))
    for name in MODELS:
        assert np.array_equal(model_meta[name].loc[common].n_samples,meta['sequence'].loc[common].n_samples)
    pd.DataFrame({'patient_id':common}).to_csv(output/'common_diagnostic_patients.csv.gz',index=False)
    audit['common_diagnostic_patients']=len(common)
    audit['common_cohort_definition']='Present in every model, all sample panels defined, identical sample sets/counts across models; therefore excludes patients with any UHN-WGS-V1 sample.'
    audit['diagnostics']='All 256 original dimensions. Decoders fit on original train patients, evaluated on original test patients; validation not used for tuning. Ridge alpha=1; assay nearest centroid after train-only feature standardization. No encoders, clusters, or UMAPs refit.'
    audit['interpretation']='Descriptive assay associations may also reflect cancer type and institution; they are not causal batch-effect estimates. Supplied TMB outliers are retained, including biologically implausible rates; log1p is used for diagnostics and shared 99th-percentile color saturation for plots.'
    json_write(audit,output/'metadata_audit.json')
    if args.stage in ('all','diagnostics'):
        results=[];pc_rows=[]
        for name in MODELS:
            print('Diagnostics:',name,flush=True)
            flat,pc=diagnostics(loaded[name],model_meta[name],common,name,output)
            results.append(flat);pc_rows.extend(pc)
            print(json.dumps(flat),flush=True)
        table=pd.DataFrame(results);table.to_csv(output/'encoder_diagnostics.csv',index=False)
        pd.DataFrame(pc_rows).to_csv(output/'pc_associations.csv',index=False)
        diagnostic_plot(table,output)
    tmb_cap=float(meta['sequence'].tmb_per_mb.quantile(.99))
    coverage_max=float(meta['sequence'].coverage_genes.max())
    top=meta['sequence'].loc[meta['sequence'].assay_count==1,'assay'].value_counts().head(12).index.tolist()
    if args.stage in ('all','plots'):
        for name in MODELS:
            frame=model_meta[name].loc[loaded[name].obs_names].copy()
            coords=np.asarray(loaded[name].obsm['X_umap'])
            frame['UMAP1'],frame['UMAP2']=coords.T
            frame.to_csv(output/f'{name}_umap_covariates.csv.gz')
            overlay_plot(loaded,model_meta,[name],output,f'{name}_umap_assay_coverage_tmb',coverage_max,tmb_cap,top)
        overlay_plot(loaded,model_meta,['esm2','genept','string'],output,'primary_variants_umap_assay_coverage_tmb',coverage_max,tmb_cap,top)
        overlay_plot(loaded,model_meta,['shuffled','gat'],output,'control_and_comparator_umap_assay_coverage_tmb',coverage_max,tmb_cap,top)
    manifest=dict(source_sha256=checksum(__file__),input_h5ad={name:dict(path=str(paths[name].resolve()),sha256=checksum(paths[name])) for name in MODELS},
                  common_patients=len(common),coverage_color_max=coverage_max,tmb_color_max=tmb_cap,
                  top_assays=top,seed=42,models=list(MODELS),diagnostic_dtype='float64',ridge_alpha=1,
                  ridge_solver='cholesky',assay_decoder='standardized nearest centroid')
    json_write(manifest,output/'analysis_manifest.json')
    shutil.copy2(__file__,output/'analysis_source.py')
    print('Complete:',output,flush=True)


if __name__=='__main__':
    main()
