"""CHORD 5x3 nested survival protocol for clean encoders and NMF baseline."""
import argparse,hashlib,json,time,warnings
from pathlib import Path
import os
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.util import Surv
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored,concordance_index_ipcw,cumulative_dynamic_auc
from sksurv.nonparametric import CensoringDistributionEstimator

DEFAULT_ROOT = Path(os.environ.get(
    'ONCOBERT_CHORD_BENCHMARK_ROOT',
    Path(__file__).resolve().parents[2] / 'data' / 'derived' / 'chord_os_benchmark'))
MODELS=['rank_esm2','rank_genept','rank_string','binary_state','nmf','mutation_count']
ALPHAS=np.array([.3,.1,.03,.01,.003,.001])
HORIZONS=[6.,12.,24.]


def checksum(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def atomic_json(value,path):
    path=Path(path);tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def load(root):
    frame=pd.read_csv(root/'prepared/patients.csv.gz')
    event=frame.event.astype(str).map({'True':True,'False':False,'1':True,'0':False})
    assert event.notna().all();frame['event']=event.astype(bool)
    assert frame.PATIENT_ID.is_unique and frame.START_DATE.eq(0).all()
    assert frame.os_months.gt(0).all() and np.isfinite(frame.os_months).all()
    profiles=dict(np.load(root/'prepared/profiles.npz'))
    np.testing.assert_array_equal(frame.PATIENT_ID,profiles['patients'])
    np.testing.assert_array_equal(frame.SAMPLE_ID,profiles['samples'])
    return frame,profiles


def prepare(root):
    frame,_=load(root);audit=json.loads((root/'prepared/audit.json').read_text())
    out=root/'benchmark';out.mkdir(exist_ok=True)
    folds=frame[['PATIENT_ID','SAMPLE_ID','CANCER_TYPE','event','os_months']].copy();folds['fold']=-1
    for cancer in audit['benchmark_cancers']:
        ix=np.flatnonzero(frame.CANCER_TYPE.eq(cancer))
        splitter=StratifiedKFold(5,shuffle=True,random_state=42)
        for fold,(_,test) in enumerate(splitter.split(ix,frame.event.to_numpy()[ix])):folds.loc[ix[test],'fold']=fold
    assert folds.fold.ge(0).all()
    path=out/'fold_assignments.csv'
    if path.exists():pd.testing.assert_frame_equal(pd.read_csv(path).astype({'event':bool}),folds.reset_index(drop=True),check_dtype=False)
    else:folds.to_csv(path,index=False)
    config=dict(cancers=audit['benchmark_cancers'],models=MODELS,outer_folds=5,inner_folds=3,seed=42,
        estimator='CoxnetSurvivalAnalysis; fixed elastic net l1_ratio=0.1; all native dimensions; no encoder fine-tuning',
        alphas=ALPHAS.tolist(),selection='Largest mean inner held-out Harrell C-index; ties prefer stronger penalty; inner-training StandardScaler only',
        endpoint='Reported positive OS_MONTHS / OS_STATUS from first sequencing; all eligible day-zero specimens retained; no treatment groups',
        patient_representation='Arithmetic mean of all eligible day-zero sample embeddings for each patient; post-baseline specimens excluded',
        primary_metric='Harrell C-index on unmodified reported OS times within each held-out cancer fold',
        secondary_metrics='Uno IPCW C-index at24 months; cumulative/dynamic IPCW AUROC at6,12,24 months with training-only censoring distribution',
        auc_support='Evaluation-only administrative censoring at min(36 months, training max followup-epsilon); does not alter OS for model fitting or primary C-index; horizons require >=10 observed cases and controls and G_train(t)>0.05',
        aggregation='Mean and SD of five held-out-fold metrics; fold risk scores are not pooled to compute a global C-index',
        nmf_baseline='128-component NMF fitted only on GENIE training profiles; fixed transform on CHORD; same nested Cox procedure',
        mutation_count_baseline='log1p number of mutated shared200 genes; same nested Cox procedure',
        audit_sha256=checksum(root/'prepared/audit.json'),fold_sha256=checksum(path),source_sha256=checksum(Path(__file__)))
    target=out/'protocol.json'
    if target.exists():assert json.loads(target.read_text())==config
    else:atomic_json(config,target)
    return frame,folds,config


def survival(frame):return Surv.from_arrays(frame.event.to_numpy(dtype=bool),frame.os_months.to_numpy(dtype=float))


def safe_concordance(y,risk):
    assert np.isfinite(risk).all()
    return float(concordance_index_censored(y['event'],y['time'],risk)[0])


def evaluate(y_train,y_test,risk):
    result=dict(harrell_c=safe_concordance(y_test,risk));notes={}
    # Only the horizon metrics need censoring-support restriction. Fit and primary
    # concordance always use the original reported survival times.
    cap=min(36.,float(y_train['time'].max())-1e-6)
    capped=y_test.copy();beyond=capped['time']>=cap;capped['event'][beyond]=False;capped['time'][beyond]=cap
    result['horizon_evaluation_cap_months']=cap
    censor=CensoringDistributionEstimator().fit(y_train)
    for t in HORIZONS:
        key='auc_'+str(int(t));cases=int((capped['event']&(capped['time']<=t)).sum());controls=int((capped['time']>t).sum())
        result[key+'_cases']=cases;result[key+'_controls']=controls
        if t>=cap or cases<10 or controls<10:
            result[key]=None;notes[key]='Insufficient follow-up/cases/controls';continue
        try:
            g=float(censor.predict_proba(np.array([t]))[0])
            if g<=.05:raise ValueError('Training censoring survival <=0.05')
            score,_=cumulative_dynamic_auc(y_train,capped,risk,np.array([t]));result[key]=float(score[0])
        except ValueError as error:result[key]=None;notes[key]=str(error)
    try:result['ipcw_c24']=float(concordance_index_ipcw(y_train,capped,risk,tau=24.)[0])
    except ValueError as error:result['ipcw_c24']=None;notes['ipcw_c24']=str(error)
    result['metric_notes']=notes
    return result


def fit_path(x,y):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model=CoxnetSurvivalAnalysis(alphas=ALPHAS,l1_ratio=.1,normalize=False,max_iter=100000,tol=1e-7).fit(x,y)
    return model,[str(w.message) for w in caught]


def nested_fit(x,y,outer_train,outer_test,fold):
    train_y=y[outer_train];train_x=x[outer_train]
    inner=StratifiedKFold(3,shuffle=True,random_state=4200+fold)
    validation=np.empty((3,len(ALPHAS)));notes=[]
    for i,(fit,val) in enumerate(inner.split(train_x,train_y['event'])):
        scale=StandardScaler().fit(train_x[fit]);model,w=fit_path(scale.transform(train_x[fit]),train_y[fit]);notes.extend(w)
        transformed=scale.transform(train_x[val])
        for j,alpha in enumerate(ALPHAS):validation[i,j]=safe_concordance(train_y[val],model.predict(transformed,alpha=float(alpha)))
    mean=validation.mean(0);chosen=int(np.argmax(mean));alpha=float(ALPHAS[chosen])
    scale=StandardScaler().fit(train_x);model,w=fit_path(scale.transform(train_x),train_y);notes.extend(w)
    prediction=model.predict(scale.transform(x[outer_test]),alpha=alpha)
    assert np.isfinite(prediction).all()
    return prediction,dict(selected_alpha=alpha,inner_mean_c=mean.tolist(),inner_fold_c=validation.tolist(),warnings=notes)


def run(root,cancer_index,smoke=False):
    frame,folds,config=prepare(root);cancer=config['cancers'][cancer_index]
    local=np.flatnonzero(frame.CANCER_TYPE.eq(cancer));cohort=frame.iloc[local].reset_index(drop=True)
    assignments=folds.iloc[local].fold.to_numpy();y=survival(cohort)
    slug=''.join(c.lower() if c.isalnum() else '_' for c in cancer)
    dest=root/'benchmark/groups'/slug;dest.mkdir(parents=True,exist_ok=True)
    features={name:np.load(root/'representations'/f'{name}.npy',mmap_mode='r') for name in MODELS if name!='mutation_count'}
    features['mutation_count']=np.log1p(frame.shared200_mutation_count.to_numpy(dtype=float))[:,None]
    records=[]
    for name in (MODELS[:1] if smoke else MODELS):
        x=np.asarray(features[name][local],dtype=np.float64);assert np.isfinite(x).all()
        for fold in ([0] if smoke else range(5)):
            test=np.flatnonzero(assignments==fold);train=np.flatnonzero(assignments!=fold)
            assert not set(cohort.PATIENT_ID.iloc[train])&set(cohort.PATIENT_ID.iloc[test])
            prefix=f'{name}_fold{fold}';result_path=dest/(prefix+'.json')
            if result_path.exists() and not smoke:
                record=json.loads(result_path.read_text());assert record['protocol_sha256']==checksum(root/'benchmark/protocol.json');records.append(record);continue
            start=time.monotonic();risk,fit=nested_fit(x,y,train,test,fold);metrics=evaluate(y[train],y[test],risk)
            record=dict(cancer=cancer,model=name,fold=fold,train_patients=len(train),test_patients=len(test),test_deaths=int(y['event'][test].sum()),
                **metrics,**fit,seconds=time.monotonic()-start,protocol_sha256=checksum(root/'benchmark/protocol.json'))
            if smoke:
                atomic_json(record,root/'benchmark/smoke_result.json');print('SMOKE',json.dumps(record),flush=True);return
            pd.DataFrame(dict(patient_id=cohort.PATIENT_ID.iloc[test],sample_id=cohort.SAMPLE_ID.iloc[test],cancer=cancer,fold=fold,
                event=y['event'][test],os_months=y['time'][test],risk=risk)).to_csv(dest/(prefix+'_predictions.csv.gz'),index=False)
            atomic_json(record,result_path);records.append(record)
            print(cancer,name,'fold',fold,'C',metrics['harrell_c'],'AUC12',metrics['auc_12'],'seconds',round(record['seconds'],2),flush=True)
    assert len(records)==5*len(MODELS)
    pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in records]).to_csv(dest/'fold_metrics.csv',index=False)
    atomic_json(dict(complete=True,cancer=cancer,folds=5,models=MODELS,protocol_sha256=checksum(root/'benchmark/protocol.json')),dest/'complete.json')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=DEFAULT_ROOT);p.add_argument('--cancer-index',type=int,default=0);p.add_argument('--prepare-only',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.prepare_only:prepare(a.root)
    else:run(a.root,a.cancer_index,a.smoke)
