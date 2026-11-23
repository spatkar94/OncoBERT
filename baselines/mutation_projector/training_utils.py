"""Four clean shared-200 experiments: learned genes, uniform15% rank/state masking."""
import argparse,csv,hashlib,json,random,time
from contextlib import nullcontext
from pathlib import Path
import h5py,numpy as np,torch
from sklearn.metrics import average_precision_score,roc_auc_score
from .masking import CleanMaskedSetModel, uniform_mask, masked_ce

BASE=Path(__file__).resolve().parent
CODE=['training_utils.py', 'masking.py']


def checksum(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def atomic_json(value,path):
    path=Path(path);tmp=path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def save(value,path):
    tmp=path.with_suffix('.tmp.pth');torch.save(value,tmp);tmp.replace(path)


def amp(device,precision):
    return torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' and precision=='bf16' else nullcontext()


def load_data(root,condition):
    data=dict(np.load(root/'profiles.npz',allow_pickle=False))
    num_genes=data['states'].shape[1]
    assert num_genes==len(data['genes']) and np.isin(data['states'],[1,2]).all()
    assert np.array_equal(data['states']==2,data['mutations'].astype(bool))
    parts={s:np.flatnonzero(data['split']==s) for s in ['train','validation','test']}
    groups={s:set(data['patients'][rows]) for s,rows in parts.items()}
    assert all(not groups[a]&groups[b] for a,b in [('train','validation'),('train','test'),('validation','test')])
    if condition=='binary_state':values=data['states'].astype(np.int64)-1;classes=2
    else:
        ranks=np.load(root/(condition.removeprefix('rank_')+'_ranks.npy'))
        assert ranks.shape==data['states'].shape
        np.testing.assert_array_equal(np.sort(ranks,axis=1),np.broadcast_to(np.arange(1,num_genes+1),ranks.shape))
        values=ranks.astype(np.int64)-1;classes=200
        classes=num_genes
    return data,parts,torch.from_numpy(values),classes


def binary_metrics(labels,scores,gene_ids,genes,prior):
    rows=[]
    for i,gene in enumerate(genes):
        selected=gene_ids==i;y=labels[selected];positive=int(y.sum());n=len(y)
        ap=float(average_precision_score(y,scores[selected])) if 0<positive<n else None
        rows.append(dict(gene=str(gene),masked_entries=n,mutated_entries=positive,prevalence=float(y.mean()) if n else None,average_precision=ap))
    supported=[r for r in rows if r['average_precision'] is not None]
    pred=scores>=.5;pos=labels==1;neg=~pos
    metrics=dict(mutation_micro_auprc=float(average_precision_score(labels,scores)),
        mutation_macro_auprc=float(np.mean([r['average_precision'] for r in supported])),genes_with_ap=len(supported),
        mutation_auroc=float(roc_auc_score(labels,scores)),mutation_prevalence=float(labels.mean()),
        binary_top1=float((pred==pos).mean()),binary_balanced_accuracy=float(.5*((pred[pos]).mean()+(~pred[neg]).mean())),
        gene_frequency_baseline_micro_auprc=float(average_precision_score(labels,prior[gene_ids])),
        gene_frequency_baseline_macro_auprc=float(np.mean([r['prevalence'] for r in supported])))
    return metrics,rows


def write_rows(rows,path):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


@torch.inference_mode()
def evaluate(model,values,masks,condition,genes,prior,batch_size,device,precision):
    model.eval();total=0;ce=0.;sums=np.zeros(5);ys=[];ps=[];gs=[]
    for start in range(0,len(values),batch_size):
        v=values[start:start+batch_size].to(device);m=masks[start:start+batch_size].to(device)
        with amp(device,precision):logits,_=model(v,m)
        selected=logits[m].float();target=v[m];pred=selected.argmax(-1);n=len(target)
        ce+=torch.nn.functional.cross_entropy(selected,target,reduction='sum').item();total+=n
        if condition=='binary_state':
            ys.append(target.cpu().numpy().astype(np.uint8));ps.append(selected.softmax(-1)[:,1].cpu().numpy());gs.append(torch.nonzero(m,as_tuple=False)[:,1].cpu().numpy())
        else:
            diff=pred-target;order=selected.topk(10).indices
            sums+=np.array([(pred==target).sum().item(),(order[:,:5]==target[:,None]).any(1).sum().item(),(order==target[:,None]).any(1).sum().item(),diff.abs().sum().item(),diff.square().sum().item()])
    metrics=dict(reconstruction_ce=ce/total,masked_entries=total)
    if condition=='binary_state':
        labels=np.concatenate(ys);scores=np.concatenate(ps);gene_ids=np.concatenate(gs)
        binary,pergene=binary_metrics(labels,scores,gene_ids,genes,prior);metrics.update(binary)
        predictions=dict(labels=labels,mutation_probabilities=scores,gene_indices=gene_ids)
    else:
        metrics.update(rank_top1=float(sums[0]/total),rank_top5=float(sums[1]/total),rank_top10=float(sums[2]/total),rank_mae=float(sums[3]/total),rank_rmse=float(np.sqrt(sums[4]/total)))
        pergene=predictions=None
    return metrics,pergene,predictions


@torch.inference_mode()
def export_embeddings(model,values,data,out,args,device):
    """Export unmasked best-checkpoint CLS vectors and patient means for later analysis."""
    model.eval();samples=np.empty((len(values),args.dim),dtype=np.float32)
    for start in range(0,len(values),args.batch_size):
        with amp(device,args.precision):_,cls=model(values[start:start+args.batch_size].to(device))
        samples[start:start+len(cls)]=cls.float().cpu().numpy()
    assert np.isfinite(samples).all()
    ids,inverse,counts=np.unique(data['patients'],return_inverse=True,return_counts=True)
    sums=np.zeros((len(ids),args.dim),dtype=np.float64);np.add.at(sums,inverse,samples)
    tmp=out/'embeddings.tmp.h5'
    with h5py.File(tmp,'w') as f:
        for k,v in [('sample_ids',data['samples']),('sample_patients',data['patients']),('sample_split',data['split']),('genes',data['genes']),('patient_ids',ids)]:
            f.create_dataset(k,data=np.asarray(v,dtype=object),dtype=h5py.string_dtype('utf-8'))
        f.create_dataset('sample_embeddings',data=samples,compression='lzf');f.create_dataset('patient_embeddings',data=(sums/counts[:,None]).astype(np.float32),compression='lzf');f.create_dataset('patient_n_samples',data=counts)
        f.attrs.update(checkpoint_sha256=checksum(out/'best.pth'),sample_pooling='unmasked_CLS',patient_pooling='mean_sample_CLS',gene_identity=f'learned nn.Embedding(200,{args.dim})',representation_dim=args.dim,condition=args.condition)
    tmp.replace(out/'embeddings.h5')


def train(args):
    torch.set_num_threads(args.threads);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    device=torch.device(args.device);root=Path(args.prepared).resolve();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    data,parts,values,classes=load_data(root,args.condition)
    num_genes=len(data['genes'])
    masked_genes=max(1,int(num_genes*.15))
    architecture=dict(num_classes=classes,num_genes=num_genes,dim=args.dim,layers=args.layers,heads=8,ff_dim=2048,dropout=.1)
    config={k:v for k,v in vars(args).items() if k!='stop_after'}
    input_names=['profiles.npz']+([] if args.condition=='binary_state' else [args.condition.removeprefix('rank_')+'_ranks.npy'])
    input_names += [n for n in ['validation_masks.npz','test_masks.npz'] if (root/n).exists()]
    config.update(prepared=str(root),output=str(out),architecture=architecture,gene_identity=f'Fresh trainable nn.Embedding(200,{args.dim}); no pretrained vectors',input_order_embeddings=False,
        masking=f'Uniform without replacement, exactly {masked_genes}/{num_genes} rank or binary-state values per sample; gene identity retained',rank_timing='Precomputed full mutation profile; no diffusion recomputation during training' if args.condition.startswith('rank_') else 'Not applicable',
        objective='Standard unweighted CE on masked genes only',selection='Minimum fixed-mask validation reconstruction CE',initialization='Fresh seed42; identical gene table and backbone across all four models',
        input_sha256={n:checksum(root/n) for n in input_names},code_sha256={n:checksum(BASE/n) for n in CODE},sample_counts={s:len(rows) for s,rows in parts.items()})
    if (out/'config.json').exists():assert json.loads((out/'config.json').read_text())==config
    atomic_json(config,out/'config.json')
    model=CleanMaskedSetModel(**architecture).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    prior=data['mutations'][parts['train']].mean(0)
    val_masks=uniform_mask(values[parts['validation']],.15,torch.Generator().manual_seed(args.seed+100000))
    test_masks=uniform_mask(values[parts['test']],.15,torch.Generator().manual_seed(args.seed+200000))
    for split,masks in [('validation',val_masks),('test',test_masks)]:
        path=root/(split+'_masks.npz')
        if path.exists():
            fixed=np.load(path);np.testing.assert_array_equal(fixed['sample_rows'],parts[split]);np.testing.assert_array_equal(fixed['masks'],masks.numpy())
    mask_rng=torch.Generator().manual_seed(args.seed+2);batch_rng=torch.Generator().manual_seed(args.seed+1)
    best=float('inf');stale=start=0;history=[]
    if (out/'latest.pth').exists():
        cp=torch.load(out/'latest.pth',map_location=device,weights_only=True);model.load_state_dict(cp['model_state_dict']);optimizer.load_state_dict(cp['optimizer'])
        start=cp['epoch'];best=cp['best'];stale=cp['stale'];history=cp['history'];torch.set_rng_state(cp['torch_rng'].cpu());mask_rng.set_state(cp['mask_rng'].cpu());batch_rng.set_state(cp['batch_rng'].cpu())
        if device.type=='cuda':torch.cuda.set_rng_state_all([r.cpu() for r in cp['cuda_rng']])
    print(json.dumps(dict(condition=args.condition,architecture=architecture,resume_epoch=start,gene_lookup_trainable=model.gene_embedding.weight.requires_grad)),flush=True)
    invocations=0
    for epoch in range(start,args.epochs):
        if stale>=args.patience:break
        begun=time.monotonic();model.train();loss_sum=0.;total=correct=0
        order=parts['train'][torch.randperm(len(parts['train']),generator=batch_rng).numpy()]
        for step,offset in enumerate(range(0,len(order),args.batch_size)):
            rows=order[offset:offset+args.batch_size];v=values[rows];mask=uniform_mask(v,.15,mask_rng).to(device);v=v.to(device)
            optimizer.zero_grad(set_to_none=True)
            with amp(device,args.precision):logits,_=model(v,mask)
            loss=masked_ce(logits,v,mask)
            if not torch.isfinite(loss):raise ValueError('Nonfinite reconstruction loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            n=int(mask.sum());loss_sum+=loss.item()*n;total+=n;correct+=(logits[mask].argmax(-1)==v[mask]).sum().item()
            if step==0 or (step+1)%200==0:print(f'{args.condition} epoch{epoch+1} batch{step+1} CE={loss.item():.5f}',flush=True)
        val,pergene,_=evaluate(model,values[parts['validation']],val_masks,args.condition,data['genes'],prior,args.batch_size,device,args.precision)
        base=dict(epoch=epoch+1,model_state_dict=model.state_dict(),architecture=architecture,config=config,genes=data['genes'].tolist(),validation=val)
        if val['reconstruction_ce']<best:
            best=val['reconstruction_ce'];stale=0;save(base,out/'best.pth')
            if pergene:write_rows(pergene,out/'validation_best_per_gene_ap.csv')
        else:stale+=1
        record=dict(epoch=epoch+1,train_reconstruction_ce=loss_sum/total,train_top1=correct/total,train_masked_entries=total,validation=val,epochs_without_improvement=stale,seconds=time.monotonic()-begun);history.append(record)
        save(dict(**base,optimizer=optimizer.state_dict(),best=best,stale=stale,history=history,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [],mask_rng=mask_rng.get_state(),batch_rng=batch_rng.get_state()),out/'latest.pth')
        atomic_json(history,out/'history.json')
        flat=[{k:v for k,v in r.items() if k!='validation'}|{'validation_'+k:v for k,v in r['validation'].items()} for r in history];write_rows(flat,out/'epoch_metrics.csv')
        print(json.dumps(record),flush=True);invocations+=1
        if args.stop_after and invocations>=args.stop_after:return
    cp=torch.load(out/'best.pth',map_location=device,weights_only=True);model.load_state_dict(cp['model_state_dict'])
    test,pergene,pred=evaluate(model,values[parts['test']],test_masks,args.condition,data['genes'],prior,args.batch_size,device,args.precision)
    test['best_epoch']=int(cp['epoch']);atomic_json(test,out/'test_metrics.json')
    if pergene:write_rows(pergene,out/'test_per_gene_ap.csv');np.savez_compressed(out/'test_predictions.npz',**pred)
    export_embeddings(model,values,data,out,args,device)
    atomic_json(dict(complete=True,epochs_run=len(history),best_epoch=int(cp['epoch']),validation=cp['validation'],test=test,early_stopped=stale>=args.patience),out/'pretraining_summary.json')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepared',required=True);p.add_argument('--output',required=True)
    p.add_argument('--condition',choices=['rank_esm2','rank_genept','rank_string','binary_state'],required=True)
    for k,v in [('seed',42),('epochs',500),('patience',20),('batch-size',64),('dim',256),('layers',8),('threads',8),('stop-after',0)]:p.add_argument('--'+k,type=int,default=v)
    p.add_argument('--device',default='cuda:0');p.add_argument('--precision',choices=['bf16','fp32'],default='bf16');train(p.parse_args())
