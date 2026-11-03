"""Validate private pseudonymous inputs; never infer or repair clinical labels."""
from pathlib import Path
import argparse,json,hashlib,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--data-root',type=Path,required=True);parser.add_argument('--policy',choices=['balanced','natural'],required=True);args=parser.parse_args()
 rows=json.loads(args.manifest.read_text(encoding='utf-8'));base=args.data_root.resolve();ids=set();output=[];digest=hashlib.sha256();issues=[]
 if len(rows)<10:raise ValueError('At least ten subjects are required for this five-fold/inner-validation workflow')
 for r in rows:
  sid=r['subject_id']
  if not isinstance(sid,str) or not sid or sid in ids:raise ValueError('Subject IDs must be unique nonempty strings')
  ids.add(sid);relative=Path(r['folder'])
  if relative.is_absolute() or '..' in relative.parts:raise ValueError('Use relative folders within data-root')
  folder=(base/relative).resolve()
  if not folder.is_relative_to(base):raise ValueError('Folder escapes data-root')
  if len(r['y'])!=7 or len(r['odor'])!=16:raise ValueError('Invalid input dimensions')
  for v,maximum in zip(r['y'],[5,3,6,3,2,5,6]):
   if v is not None and (not isinstance(v,(int,float)) or not np.isfinite(v) or not 0<=v<=maximum or v!=int(v)):raise ValueError('Invalid domain score')
  if not isinstance(r['total'],(int,float)) or not np.isfinite(r['total']) or not 0<=r['total']<=30 or int(r['total'])!=r['total']:raise ValueError('Invalid recorded total')
  if any(v not in [0,1] for v in r['odor']):raise ValueError('Odor values must be binary')
  for trial in range(1,41):
   lengths=[]
   for mod,channels in [('eye',12),('angle',8)]:
    f=folder/f'trial{trial}_{mod}.npy';raw=f.read_bytes();digest.update(f'{relative.as_posix()}/{f.name}'.encode());digest.update(raw)
    arr=np.load(f,allow_pickle=False)
    if arr.ndim!=2 or arr.shape[0]<1 or arr.shape[1]!=channels or not np.isfinite(arr).all():raise ValueError('Invalid eye array')
    lengths.append(len(arr))
   if lengths[0]!=lengths[1]:raise ValueError('Paired trial lengths differ')
  missing=sum(v is None for v in r['y']);delta=None if missing else r['total']-sum(r['y'])
  if missing or delta:issues.append({'subject_id':sid,'missing_domains':missing,'recorded_total_minus_domain_sum':delta})
  output.append({k:r[k] for k in ['subject_id','y','total','odor']}|{'folder':str(folder)})
 experiment=ROOT/'experiments'/args.policy;audit=experiment/'audit'
 if (audit/'cohort.json').exists() or (experiment/'results').exists():raise RuntimeError('Existing run found; use a fresh checkout to preserve prior outputs')
 audit.mkdir(parents=True,exist_ok=True)
 (audit/'cohort.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
 (audit/'audit_summary.json').write_text(json.dumps({'n':len(output),'npy_sha256':digest.hexdigest(),'publication_labels_need_review':bool(issues),'array_hash_scope':'relative subject folders and raw paired NPY bytes'},indent=2))
 (audit/'label_issues.json').write_text(json.dumps(issues,indent=2))
 subprocess.run([sys.executable,str(experiment/'run_revision.py'),'prepare'],check=True)
 (experiment/'locked_config.json').write_bytes((ROOT/'configs'/(args.policy+'.json')).read_bytes())
 print('Initialized',args.policy,'subjects',len(output),'flagged records',len(issues))
if __name__=='__main__':main()
