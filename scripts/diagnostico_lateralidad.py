import numpy as np, pandas as pd, re
from pathlib import Path
meta=pd.read_csv('out/train_metadata.csv'); X=np.load('out/X_landmarks_features.npy')
rows=[]
for r in meta.itertuples():
    a=np.load(r.path).astype(np.float32); t=len(a)
    pose=a[:,:132].reshape(t,33,4); lh=a[:,1536:1599]; rh=a[:,1599:]; face=a[:,132:1536]
    sw=np.linalg.norm(pose[:,11,:2]-pose[:,12,:2],axis=1)
    rows.append(dict(signer=r.signer,label=r.label,view=r.view,n=t,
      lh=(np.abs(lh).max(1)>1e-6).mean(),rh=(np.abs(rh).max(1)>1e-6).mean(),face=(np.abs(face).max(1)>1e-6).mean(),
      sw=np.median(sw),cx=np.median((pose[:,11,0]+pose[:,12,0])/2),cy=np.median((pose[:,11,1]+pose[:,12,1])/2),
      wrist_y_l=np.median(pose[:,15,1]),wrist_y_r=np.median(pose[:,16,1]),
      vis=np.median(pose[:,[11,12,13,14,15,16],3])))
d=pd.DataFrame(rows)
pd.set_option('display.width',200)
print(d.groupby('signer').mean(numeric_only=True).round(3).T)
print(d.groupby(['signer','view']).agg(n=('n','mean'),sw=('sw','mean'),lh=('lh','mean'),rh=('rh','mean'),cx=('cx','mean')).round(3))
print(d.groupby(['signer','label']).agg(n=('n','mean'),lh=('lh','mean'),rh=('rh','mean')).round(2).unstack(0))
