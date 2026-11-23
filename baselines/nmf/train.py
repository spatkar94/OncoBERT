"""Train-only NMF rank selection and inference for the matched shared-200 cohort."""
import argparse,hashlib,json,time,warnings
from pathlib import Path
import h5py,joblib,numpy as np,sklearn
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning


def checksum(path):
 """Return the SHA-256 checksum of a file without external project imports."""
 digest=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):
   digest.update(block)
 return digest.hexdigest()


def write_json(value,path):
 tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def mse(x,w,h,batch_size=2048):
 """Bounded-memory mean squared reconstruction error, including zero entries."""
 total=0.
 for start in range(0,len(x),batch_size):
  residual=x[start:start+batch_size].astype(np.float64)-w[start:start+batch_size].astype(np.float64)@h.astype(np.float64)
  total+=float(np.sum(residual**2))
 return total/x.size


def canonicalize(w,h):
 """Fix arbitrary component scale with unit-L2 gene loadings, preserving WH."""
 scale=np.linalg.norm(h.astype(np.float64),axis=1).astype(np.float32)
 scale[scale==0]=1
 return w*scale,h/scale[:,None],scale


def fit_candidate(train,validation,rank,seed=42,max_iter=1000,tol=1e-4):
 model=NMF(n_components=rank,init='nndsvda',solver='cd',beta_loss='frobenius',random_state=seed,max_iter=max_iter,tol=tol,shuffle=False)
 begun=time.monotonic()
 with warnings.catch_warnings(record=True) as caught:
  warnings.simplefilter('always',ConvergenceWarning)
  w=model.fit_transform(csr_matrix(train))
  vw=model.transform(csr_matrix(validation))
 record=dict(rank=rank,train_mse=mse(train,w,model.components_),validation_mse=mse(validation,vw,model.components_),iterations=int(model.n_iter_),seconds=time.monotonic()-begun,
  convergence_warning=any(issubclass(v.category,ConvergenceWarning) for v in caught),warnings=[str(v.message) for v in caught])
 return model,record


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--ranks',type=int,nargs='+',default=[16,32,64,128]);p.add_argument('--max-iter',type=int,default=1000);p.add_argument('--tol',type=float,default=1e-4);p.add_argument('--seed',type=int,default=42)
 args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
 if (out/'summary.json').exists():raise RuntimeError('Completed experiment already exists')
 data=dict(np.load(args.prepared/'profiles.npz'));x=data['mutations'].astype(np.float32)
 assert np.isin(x,[0,1]).all() and np.isin(data['states'],[1,2]).all()
 np.testing.assert_array_equal(data['states']==2,x==1)
 parts={s:np.flatnonzero(data['split']==s) for s in ['train','validation','test']}
 for a,b in [('train','test'),('train','validation'),('validation','test')]:assert not set(data['patients'][parts[a]])&set(data['patients'][parts[b]])
 ranks=sorted(set(args.ranks));assert ranks and min(ranks)>0 and max(ranks)<=min(x[parts['train']].shape)
 rows=[]
 for rank in ranks:
  dest=out/f'rank_{rank}';dest.mkdir(exist_ok=True)
  # Test samples are never used for fitting or rank selection.
  model,record=fit_candidate(x[parts['train']],x[parts['validation']],rank,args.seed,args.max_iter,args.tol)
  joblib.dump(model,dest/'nmf.joblib');write_json(record,dest/'metrics.json');rows.append(record)
  write_json(rows,out/'rank_metrics.json');print(json.dumps(record),flush=True)
 best=min(rows,key=lambda r:(r['validation_mse'],r['rank']));rank=best['rank'];dest=out/f'rank_{rank}'
 model=joblib.load(dest/'nmf.joblib');basis=model.components_.copy();w=np.empty((len(x),rank),dtype=np.float32)
 inference_warnings=set()
 for start in range(0,len(x),4096):
  with warnings.catch_warnings(record=True) as caught:
   warnings.simplefilter('always',ConvergenceWarning)
   w[start:start+4096]=model.transform(csr_matrix(x[start:start+4096]))
   inference_warnings.update(str(v.message) for v in caught)
 np.testing.assert_array_equal(model.components_,basis)
 assert np.isfinite(w).all() and (w>=0).all()
 test_mse=mse(x[parts['test']],w[parts['test']],basis)
 w,h,scale=canonicalize(w,basis);np.save(dest/'canonical_gene_components.npy',h);np.save(dest/'component_scales.npy',scale)
 assert np.isfinite(w).all()
 np.testing.assert_array_equal(w[x.sum(1)==0],0)
 ids,inverse,counts=np.unique(data['patients'],return_inverse=True,return_counts=True)
 sums=np.zeros((len(ids),rank),dtype=np.float64);np.add.at(sums,inverse,w);patients=(sums/counts[:,None]).astype(np.float32)
 path=out/'embeddings.h5';tmp=path.with_suffix('.tmp.h5')
 with h5py.File(tmp,'w') as f:
  for key,value in [('sample_ids',data['samples']),('sample_patients',data['patients']),('sample_split',data['split']),('patient_ids',ids),('genes',data['genes'])]:f.create_dataset(key,data=np.asarray(value,dtype=object),dtype=h5py.string_dtype('utf-8'))
  f.create_dataset('patient_n_samples',data=counts);f.create_dataset('sample_embeddings',data=w,compression='lzf');f.create_dataset('patient_embeddings',data=patients,compression='lzf')
  f.attrs.update(rank=rank,model_sha256=checksum(dest/'nmf.joblib'),sample_pooling='NMF coefficient inference with fixed training gene components',patient_pooling='mean_sample_coefficients',factor_scaling='gene components unit L2; sample coefficients rescaled reciprocally')
 tmp.replace(path)
 summary=dict(complete=True,selected_rank=rank,selection='Minimum validation MSE; ties prefer smaller rank; test never used for fitting/selection',test_mse=test_mse,candidates=rows,
  samples=len(x),patients=len(ids),genes=x.shape[1],sample_splits={k:len(v) for k,v in parts.items()},zero_mutation_samples=int((x.sum(1)==0).sum()),
  inference_warnings=sorted(inference_warnings),sklearn_version=sklearn.__version__,seed=args.seed,input_sha256=checksum(args.prepared/'profiles.npz'),
  factor_scaling='Each gene component normalized to unit L2; coefficients multiplied by original component norm; reconstruction unchanged',
  caution='Higher rank usually improves reconstruction; rank selection is not evidence of biological superiority. NMF reconstruction differs from masked neural reconstruction.')
 write_json(summary,out/'summary.json');print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
