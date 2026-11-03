"""Locked subject-level CV, leakage-free model selection, dummy and modality controls."""
from pathlib import Path
import argparse, json, sys, hashlib, math, time, random, os
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'vendor'), str(ROOT/'model_source')]
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from scipy import stats
from audit_data import ENGLISH, MAXIMA, sha

DEEP_MODELS=['olfactory_only','eye_only','multimodal_no_wavelet','multimodal']
MODELS=['mean_dummy','median_dummy','ridge_eye','ridge_olfactory','ridge_multimodal']+DEEP_MODELS
from latest_model import SubscoreModel, wavelet_inputs

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)

def cohort():
    rows=json.loads((ROOT/'audit/cohort.json').read_text())
    y=np.array([[np.nan if v is None else v for v in r['y']] for r in rows])
    return rows,y,np.array([r['total'] for r in rows])

def splits(n,seed=42):
    order=np.random.RandomState(seed).permutation(n);folds=[]
    for i,test in enumerate(np.array_split(order,5)):
        train=np.setdiff1d(np.arange(n),test)
        shuffled=np.random.RandomState(seed+100+i).permutation(train)
        nv=math.ceil(.2*len(train));val=shuffled[:nv];fit=shuffled[nv:]
        assert not set(test)&set(train) and not set(fit)&set(val)
        folds.append({'fold':i,'train':train.tolist(),'fit':fit.tolist(),'validation':val.tolist(),'test':test.tolist()})
    return folds

def prepare(a):
    rows,y,total=cohort();spec=splits(len(rows),a.seed)
    dump(ROOT/'locked_splits.json',spec)
    cfg={'seed':a.seed,'folds':5,'epochs':120,'patience':20,'lr':3e-4,'weight_decay':1e-3,
         'target_length':256,'d_model':32,'nhead':4,'num_layers':2,'dropout':.25,'mlp_hidden':192,
         'num_bica_blocks':1,'L_smell':64,'batch_sizes':[8]*7,'wavelet_embed_dim':64,'wavelet_dim':980,
         'wavelet':{'wavelet_len':256,'dwt_wavelet':'db4','dwt_level':4,'use_cwt':True,
                    'cwt_wavelet':'morl','cwt_freqs':12,'cwt_freq_min':.5,'cwt_freq_max':15.,'fs':60.},
         'primary_prediction':'class probability expectation; not calibrated after class reweighting',
         'secondary_prediction':'argmax',
         'model_selection':'inner validation expected-score MAE; full outer-train refit replays selected LR schedule',
         'training':'AdamW; inverse-square-root-frequency replacement sampler AND class-weighted cross entropy (server balanced policy)',
         'missing_outcomes':'masked per domain, never zero-filled or imputed',
         'primary_total_reference':'recorded MoCA total, provisional pending source reconciliation',
         'secondary_total_reference':'complete-case sum of seven observed domains',
         'normalization':'training-only stats on per-trial pooled signals and wavelet features; clipped at +/-6; odor training-only zscore',
         'label_status':'provisional pending source-form reconciliation',
         'wavelet_frequency_note':'CWT fs=60 is the server algorithm reference after normalized-time resampling; not verified physical Hz',
         'source_model':'server wavelet enhanced 0409, frozen source snapshot',
         'secondary_controls':'same-backbone multimodal without wavelet; nested-CV ridge on eye summary/odor features',
         'split_note':'subject-level 5 folds seed42; exploratory extension of prior experiments, not independent confirmatory validation'}
    dump(ROOT/'locked_config.json',cfg)
    out=ROOT/'results';out.mkdir(exist_ok=True)
    for name,func in [('mean_dummy',np.nanmean),('median_dummy',np.nanmedian)]:
        pred=np.zeros((len(rows),7));pred_total=np.zeros(len(rows));pred_sum=np.zeros(len(rows));fids=np.zeros(len(rows),int)
        for fold in spec:
            tr=fold['train'];te=fold['test'];pred[te]=func(y[tr],axis=0);pred_total[te]=func(total[tr]);pred_sum[te]=func(y[tr].sum(1));fids[te]=fold['fold']
        save_predictions(out/name,rows,y,total,pred,pred_total,pred,fids,sum_total=pred_sum)
    print('Locked five subject-level folds and computed training-fold-only dummy predictions.',flush=True)

def save_predictions(out,rows,y,total,pred,pt,argmax,fids,sum_total=None):
    out.mkdir(exist_ok=True,parents=True);records=[]
    for i,row in enumerate(rows):
        for j,target in enumerate(ENGLISH):
            records.append({'subject_id':row['subject_id'],'fold':int(fids[i]),'target':target,'true':y[i,j],'pred':pred[i,j],'argmax':argmax[i,j]})
        records.append({'subject_id':row['subject_id'],'fold':int(fids[i]),'target':'TotalRecorded','true':total[i],'pred':pt[i],'argmax':argmax[i].sum()})
        records.append({'subject_id':row['subject_id'],'fold':int(fids[i]),'target':'TotalSubscoreSum','true':y[i].sum(),'pred':pred[i].sum() if sum_total is None else sum_total[i],'argmax':argmax[i].sum()})
    pd.DataFrame(records).to_csv(out/'oof_predictions.csv',index=False)

def pool_inputs(rows,length):
    cache=ROOT/f'cache_pooled_{length}.pt'
    key=sha(ROOT/'audit/cohort.json')+json.loads((ROOT/'audit/audit_summary.json').read_text())['npy_sha256']
    if cache.exists():
        obj=torch.load(cache,weights_only=False,map_location='cpu')
        if obj['key']==key:return obj['eye'],obj['angle'],obj['odor']
    arrays=[]
    for mod in ['eye','angle']:
        people=[]
        for row in rows:
            trials=[]
            for t in range(1,41):
                x=torch.from_numpy(np.load(Path(row['folder'])/f'trial{t}_{mod}.npy').astype('float32'))
                trials.append(F.adaptive_avg_pool1d(x.T[None],length)[0].T)
            people.append(torch.stack(trials))
        arrays.append(torch.stack(people))
    odor=torch.tensor([r['odor'] for r in rows],dtype=torch.float32)
    tmp=cache.with_suffix('.tmp');torch.save({'key':key,'eye':arrays[0],'angle':arrays[1],'odor':odor},tmp);tmp.replace(cache)
    return arrays[0],arrays[1],odor

def normalize(inputs,indices):
    result=[];params=[]
    for mod_index,x in enumerate(inputs):
        dims=(0,1,2) if x.ndim==4 else (0,)
        mu=x[indices].mean(dim=dims,keepdim=True);sd=x[indices].std(dim=dims,keepdim=True,unbiased=False)
        sd=torch.where(sd<1e-6,torch.ones_like(sd),sd)
        z=(x-mu)/sd
        if mod_index!=2: z=z.clamp(-6,6)
        result.append(z);params.append((mu,sd))
    return result,params

def minibatches(indices,size,rng=None):
    ids=np.array(indices,copy=True)
    if rng is not None:rng.shuffle(ids)
    return [ids[i:i+size] for i in range(0,len(ids),size)]

def evaluate(model,inputs,idx,labels,batch,device):
    model.eval();probs=[];loss=0.;count=0
    with torch.no_grad():
        for ids in minibatches(idx,batch):
            logits=model(tuple(x[ids].to(device) for x in inputs));p=logits.softmax(-1)
            probs.append(p.cpu().numpy())
            if labels is not None:
                truth=torch.tensor(labels[ids],device=device,dtype=torch.long)
                loss+=F.cross_entropy(logits,truth,reduction='sum').item();count+=len(ids)
    return (loss/max(count,1)),np.concatenate(probs)

def fit_model(mode,task,cfg,inputs,labels,fit,val,epochs,device,seed,logpath):
    seed_all(seed);model=SubscoreModel(mode,int(MAXIMA[task]+1),cfg).to(device)
    if cfg.get('training_policy')!='natural_frequency':
        raise RuntimeError('This supplement requires natural-frequency sampling and unweighted CE')
    cw=None
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=5,min_lr=1e-6)
    rng=np.random.RandomState(seed);best=math.inf;best_ep=epochs;stale=0;batch=cfg['batch_sizes'][task]
    history=[]
    for epoch in range(epochs):
        model.train();s=0.;n=0
        if not len(val):
            for g in opt.param_groups:g['lr']=cfg['_refit_lrs'][epoch]
        lr=opt.param_groups[0]['lr']
        drawn=rng.permutation(np.array(fit))
        for ids in minibatches(drawn,batch):
            opt.zero_grad(set_to_none=True)
            logits=model(tuple(x[ids].to(device) for x in inputs));target=torch.tensor(labels[ids],dtype=torch.long,device=device)
            loss=F.cross_entropy(logits,target,weight=cw)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();s+=loss.item()*len(ids);n+=len(ids)
        v=None
        if len(val):
            _,pr=evaluate(model,inputs,val,labels,batch,device)
            v=float(np.abs(pr@np.arange(pr.shape[1])-labels[val]).mean())
            scheduler.step(v)
        history.append({'epoch':epoch+1,'train_unweighted_ce':s/n,'validation_MAE':v,'lr':lr,
                        'epoch_subjects':len(drawn),'unique_epoch_subjects':len(set(drawn.tolist()))})
        if v is not None:
            if v<best-1e-4:best=v;best_ep=epoch+1;stale=0
            else:stale+=1
        if epoch%5==0:print(mode,task,'epoch',epoch+1,'train',round(s/n,5),'val_MAE',v,flush=True)
        if len(val) and stale>=cfg['patience']:break
    dump(logpath,history)
    return model,best_ep

def train(a):
    cfg=json.loads((ROOT/'locked_config.json').read_text());spec=json.loads((ROOT/'locked_splits.json').read_text())
    rows,y,total=cohort();torch.set_num_threads(4);inputs=list(pool_inputs(rows,cfg['target_length']))+[wavelet_inputs(rows,cfg)]
    if a.mode=='smoke':
        device='cuda:0' if torch.cuda.is_available() else 'cpu';cfg=dict(cfg,target_length=32,L_smell=16,epochs=1)
        small=[x[:3,:,:32] if x.ndim==4 else x[:3] for x in inputs]
        for mode in DEEP_MODELS:
            seed_all(42);model=SubscoreModel(mode,6,cfg).to(device);xx=tuple(x.to(device) for x in small)
            logits=model(xx);loss=F.cross_entropy(logits,torch.tensor([1,2,3],device=device));loss.backward()
            assert logits.shape==(3,6) and torch.isfinite(loss)
            model.eval()
            with torch.no_grad():
                p=model(xx)
                altered=list(xx)
                if mode=='eye_only':altered[2]=torch.randn_like(altered[2])
                elif mode=='olfactory_only':altered[0]=torch.randn_like(altered[0]);altered[1]=torch.randn_like(altered[1]);altered[3]=torch.randn_like(altered[3])
                elif mode=='multimodal_no_wavelet':altered[3]=torch.randn_like(altered[3])
                else:
                    print('smoke OK',mode,flush=True)
                    continue
                assert torch.equal(p,model(tuple(altered))), 'Ablation leaks excluded modality'
            print('smoke OK',mode,flush=True)
        return
    mode=a.mode;out=ROOT/'results'/mode;out.mkdir(parents=True,exist_ok=True)
    pred=np.full_like(y,np.nan);arg=np.full_like(y,np.nan);fids=np.zeros(len(rows),int)
    device='cuda:0';signature=sha(ROOT/'locked_config.json')+sha(ROOT/'locked_splits.json')+sha(ROOT/'audit/cohort.json')+sha(__file__)+sha(ROOT/'latest_model.py')+sha(ROOT/'audit/audit_summary.json')+''.join(sha(p) for p in sorted((ROOT/'model_source').glob('*.py')))
    for fold in spec:
        f=fold['fold'];te=fold['test'];fids[te]=f
        inner,p1=normalize(inputs,fold['fit']);outer,p2=normalize(inputs,fold['train'])
        for task in range(7):
            dest=out/f'fold{f}_task{task}.npz'
            if dest.exists():
                z=np.load(dest)
                if str(z['signature'])!=signature or not np.array_equal(z['test_indices'],te):raise RuntimeError('Resume signature/fold mismatch')
                pred[te,task]=z['expected'];arg[te,task]=z['argmax'];continue
            observed=np.isfinite(y[:,task]);fit=[i for i in fold['fit'] if observed[i]];val=[i for i in fold['validation'] if observed[i]];tr=[i for i in fold['train'] if observed[i]]
            if not fit or not val:raise RuntimeError('Insufficient observed targets')
            seed=cfg['seed']+f*100+task
            print(f'START {mode} fold={f+1}/5 task={task+1}/7',flush=True)
            dump(out/'status.json',{'state':'running','fold':f,'task':task,'phase':'selection'})
            start=time.time();model,best_ep=fit_model(mode,task,cfg,inner,y[:,task],fit,val,cfg['epochs'],device,seed,out/f'fold{f}_task{task}_selection.json')
            del model;torch.cuda.empty_cache()
            lr_history=json.loads((out/f'fold{f}_task{task}_selection.json').read_text())
            refit_cfg=dict(cfg,_refit_lrs=[r['lr'] for r in lr_history[:best_ep]])
            model,_=fit_model(mode,task,refit_cfg,outer,y[:,task],tr,[],best_ep,device,seed,out/f'fold{f}_task{task}_refit.json')
            _,probs=evaluate(model,outer,te,None,cfg['batch_sizes'][task],device)
            expected=probs@np.arange(probs.shape[1]);hard=probs.argmax(1);pred[te,task]=expected;arg[te,task]=hard
            torch.save({'state_dict':model.state_dict(),'normalization':p2,'config':cfg,'mode':mode,'task':task,'fold':fold,'selected_epochs':best_ep,'signature':signature},out/f'fold{f}_task{task}.pt')
            np.savez(dest,expected=expected,argmax=hard,probabilities=probs,test_indices=te,signature=signature)
            dump(out/'status.json',{'state':'running','completed_fold':f,'completed_task':task,'elapsed_task_seconds':time.time()-start})
            del model;torch.cuda.empty_cache()
    assert np.isfinite(pred).all()
    save_predictions(out,rows,y,total,pred,pred.sum(1),arg,fids)
    dump(out/'status.json',{'state':'complete','subjects':len(rows),'folds':5})

def metric(y,p):
    valid=np.isfinite(y)&np.isfinite(p);y=y[valid];p=p[valid];e=p-y;den=((y-y.mean())**2).sum()
    r=stats.pearsonr(y,p) if np.ptp(p)>0 and np.ptp(y)>0 else (np.nan,np.nan)
    rho=stats.spearmanr(y,p) if np.ptp(p)>0 and np.ptp(y)>0 else (np.nan,np.nan)
    return {'n':len(y),'MAE':np.abs(e).mean(),'RMSE':np.sqrt((e*e).mean()),'R2':1-(e*e).sum()/den if den else np.nan,'Pearson_r':r[0],'Pearson_p_nominal':r[1],'Spearman_rho':rho[0],'Spearman_p_nominal':rho[1]}

def summarize(a):
    records=[];frames={}
    for model in MODELS:
        p=ROOT/'results'/model/'oof_predictions.csv'
        if not p.exists():continue
        df=pd.read_csv(p);frames[model]=df
        if df.duplicated(['subject_id','target']).any():raise ValueError('Duplicate OOF prediction')
        for target,g in df.groupby('target',sort=False):
            records.append(dict(model=model,target=target,scope='pooled',**metric(g.true.to_numpy(),g.pred.to_numpy())))
            for fold,fg in g.groupby('fold'):
                records.append(dict(model=model,target=target,scope=f'fold{fold}',**metric(fg.true.to_numpy(),fg.pred.to_numpy())))
    table=pd.DataFrame(records)
    for col in ['Pearson_p_nominal','Spearman_p_nominal']:
        table[col.replace('_p_nominal','_q_BH_7domains')]=np.nan
        for model in MODELS:
            mask=(table.model==model)&(table.scope=='pooled')&table.target.isin(ENGLISH)&table[col].notna()
            table.loc[mask,col.replace('_p_nominal','_q_BH_7domains')]=bh_adjust(table.loc[mask,col].to_numpy())
    table.to_csv(ROOT/'results/metrics.csv',index=False,float_format='%.10g')
    comparisons=[];rng=np.random.RandomState(20261013)
    if 'multimodal' in frames:
        for baseline in [m for m in MODELS if m!='multimodal']:
            if baseline not in frames:continue
            joined=frames['multimodal'].merge(frames[baseline],on=['subject_id','target','fold'],suffixes=('_multi','_base'),validate='one_to_one')
            for target,g in joined.groupby('target'):
                g=g.dropna(subset=['true_multi','true_base','pred_multi','pred_base'])
                assert np.array_equal(g.true_multi.to_numpy(),g.true_base.to_numpy())
                diff=np.abs(g.pred_multi-g.true_multi).to_numpy()-np.abs(g.pred_base-g.true_base).to_numpy()
                means=diff[rng.randint(0,len(diff),(10000,len(diff)))].mean(1)
                comparisons.append({'baseline':baseline,'target':target,'n':len(diff),'delta_MAE_multi_minus_baseline':diff.mean(),'ci_low':np.quantile(means,.025),'ci_high':np.quantile(means,.975),'interval_scope':'paired subject bootstrap of fixed OOF predictions; excludes refitting uncertainty'})
    pd.DataFrame(comparisons).to_csv(ROOT/'results/paired_comparisons.csv',index=False)
    print(table[table.scope=='pooled'].to_string(index=False),flush=True)

def bh_adjust(p):
    p=np.asarray(p);n=len(p)
    if not n:return p
    order=np.argsort(p);q=np.minimum.accumulate((p[order]*n/np.arange(1,n+1))[::-1])[::-1]
    result=np.empty(n);result[order]=np.minimum(q,1);return result

def ridge_controls(a):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV,KFold
    cfg=json.loads((ROOT/'locked_config.json').read_text());rows,y,total=cohort()
    inputs=pool_inputs(rows,cfg['target_length'])
    # Fixed unsupervised summaries, no population statistics or outcomes used in feature extraction.
    eye=np.concatenate([v for x in inputs[:2] for v in [x.mean((1,2)).numpy(),x.std((1,2),unbiased=False).numpy(),x.amin((1,2)).numpy(),x.amax((1,2)).numpy()]],axis=1)
    smell=inputs[2].numpy();spec=json.loads((ROOT/'locked_splits.json').read_text())
    for name,x in [('ridge_eye',eye),('ridge_olfactory',smell),('ridge_multimodal',np.concatenate([eye,smell],axis=1))]:
        pred=np.full_like(y,np.nan);fids=np.zeros(len(rows),int);meta=[]
        for fold in spec:
            te=fold['test'];fids[te]=fold['fold']
            for task in range(7):
                tr=[i for i in fold['train'] if np.isfinite(y[i,task])]
                search=GridSearchCV(make_pipeline(StandardScaler(),Ridge()),{'ridge__alpha':[.1,1,10,100,1000]},
                    cv=KFold(3,shuffle=True,random_state=cfg['seed']),scoring='neg_mean_absolute_error',n_jobs=1)
                search.fit(x[tr],y[tr,task]);pred[te,task]=np.clip(search.predict(x[te]),0,MAXIMA[task])
                meta.append({'fold':fold['fold'],'task':task,'alpha':search.best_params_['ridge__alpha'],'training_subjects':len(tr)})
        out=ROOT/'results'/name;save_predictions(out,rows,y,total,pred,pred.sum(1),np.rint(pred),fids)
        dump(out/'fit_metadata.json',meta);print('Completed',name,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','smoke','summarize','ridge']+DEEP_MODELS);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    if a.mode=='prepare':prepare(a)
    elif a.mode=='summarize':summarize(a)
    elif a.mode=='ridge':ridge_controls(a)
    else:train(a)
