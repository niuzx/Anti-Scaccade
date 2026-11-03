"""Strict identity alignment and array audit; never infer missing outcome labels."""
from pathlib import Path
import sys, json, hashlib, re, collections, argparse
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor'))
import numpy as np
import pandas as pd
from pypinyin import lazy_pinyin

DOMAINS = ['视空间总分','命名总分','注意总分','语言总分','抽象总分','延迟回忆总分','定向总分']
ENGLISH = ['VisuospatialExecutive','Naming','Attention','Language','Abstraction','DelayedRecall','Orientation']
MAXIMA = np.array([5,3,6,3,2,5,6])

def namekey(value):
    if pd.isna(value): return ''
    return re.sub('[^a-z]', '', ''.join(lazy_pinyin(str(value).strip())).lower())

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def audit(project, out, full_manifest=False):
    out.mkdir(parents=True,exist_ok=True)
    data=project/'Dataset'; eye=data/'eye1_final_prc'
    m=pd.read_excel(data/'Moca_final.xlsx',engine='openpyxl')
    s=pd.read_excel(data/'smellfinal.xls',engine='openpyxl')
    keys=m['姓名'].map(namekey); skeys=s['匹配的Moca姓名'].map(namekey)
    if not keys.is_unique or not skeys.is_unique or '' in set(keys):raise ValueError('Ambiguous subject identity')
    smap=dict(zip(skeys,range(len(s)))); fmap={}
    for folder in sorted(eye.iterdir()):
        if folder.is_dir():
            k=namekey(re.sub(r'^person_\d+_', '', folder.name))
            if k in fmap:raise ValueError('Duplicate folder identity')
            fmap[k]=folder
    if set(keys)!=set(fmap) or set(keys)!=set(skeys):raise ValueError('Cohort identity sets differ')
    y=m[DOMAINS].apply(pd.to_numeric,errors='raise').to_numpy(float)
    if np.any(np.isfinite(y)&((y<0)|(y>MAXIMA)|(y!=np.floor(y)))):raise ValueError('Invalid score range')
    total=pd.to_numeric(m['moca总分'],errors='raise').to_numpy(float)
    if not np.isfinite(total).all() or np.any((total<0)|(total>30)):raise ValueError('Invalid total')
    records=[];issues=[]; digest=hashlib.sha256(); byte_count=0
    for i,k in enumerate(keys):
        sid=f'S{i+1:03d}';row=s.iloc[smap[k]];folder=fmap[k]
        odor=[]
        for j in range(1,17):
            value=str(row[f'识别{j}']).strip()
            if not re.fullmatch(r'[01](?:\.0+)?分?',value):raise ValueError('Invalid odor response')
            odor.append(float(value.rstrip('分')))
        for t in range(1,41):
            lens=[]
            for modality,channels in [('eye',12),('angle',8)]:
                p=folder/f'trial{t}_{modality}.npy'
                raw=p.read_bytes(); byte_count+=len(raw);digest.update(str(p.relative_to(project)).encode());digest.update(raw)
                import io
                a=np.load(io.BytesIO(raw),allow_pickle=False)
                if a.ndim!=2 or a.shape[1]!=channels or a.shape[0]<1 or not np.isfinite(a).all():raise ValueError(f'Invalid array {sid} {t} {modality}')
                lens.append(len(a))
            if lens[0]!=lens[1]:raise ValueError('Eye/angle trial lengths differ')
        missing=[ENGLISH[j] for j in range(7) if not np.isfinite(y[i,j])]
        delta=None if missing else float(total[i]-y[i].sum())
        if missing or delta!=0:issues.append({'subject_id':sid,'excel_row':i+2,'missing_domains':';'.join(missing),'recorded_total':total[i],'sum_subscores':None if missing else float(y[i].sum()),'total_minus_sum':delta})
        records.append({'subject_id':sid,'folder':str(folder),'excel_row':i+2,'smell_excel_row':smap[k]+2,'sex':str(row['性别']),'age':float(row['年龄']),'education':str(row['学历']),'y':[None if not np.isfinite(v) else float(v) for v in y[i]],'total':float(total[i]),'odor':odor})
    pd.DataFrame(issues).to_csv(out/'label_issues.csv',index=False,encoding='utf-8-sig')
    (out/'cohort.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    ages=np.array([r['age'] for r in records]);sex=collections.Counter(r['sex'] for r in records)
    summary={'n':len(records),'male':sex['男'],'female':sex['女'],'sex_unreported':len(records)-sex['男']-sex['女'],'age_mean':ages.mean(),'age_sd':ages.std(ddof=1),'recorded_total_mean':total.mean(),'recorded_total_sd':total.std(ddof=1),'complete_subscores_n':int(np.isfinite(y).all(axis=1).sum()),'missing_cells':int(np.isnan(y).sum()),'complete_score_discrepancies':sum(bool(r['missing_domains']=='') for r in issues),'education_counts':dict(collections.Counter(r['education'] for r in records)),'npy_files_verified':len(records)*80,'npy_bytes_read':byte_count,'npy_sha256':digest.hexdigest(),'input_sha256':{f:sha(data/f) for f in ['Moca_final.xlsx','smellfinal.xls']},'training_arrays_ready':True,'publication_labels_need_review':True}
    if full_manifest:
        mp=project.parent/'saccade_full_20261013/TRANSFER_MANIFEST.json';manifest=json.loads(mp.read_text())
        mismatches=[]
        for f in manifest['files']:
            p=project/f['path']
            if not p.is_file() or p.stat().st_size!=f['size']:mismatches.append(f['path'])
        summary['transfer_manifest']={'expected_files':len(manifest['files']),'missing_or_size_mismatch':len(mismatches),'scope':'all archive members by existence and size; no original per-file hashes in manifest'}
        (out/'transfer_mismatches.json').write_text(json.dumps(mismatches,ensure_ascii=False),encoding='utf-8')
    (out/'audit_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,default=ROOT.parent);p.add_argument('--output',type=Path,default=ROOT/'audit');p.add_argument('--full-manifest',action='store_true');a=p.parse_args();audit(a.project,a.output,a.full_manifest)
