"""Evaluate an existing classifier using explicitly supplied held-out binary labels."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, brier_score_loss
from inference import Predictor,load_rgb,prepare_image,scalar_score


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['stroke','ckd','hyp'],required=True)
    p.add_argument('--csv',type=Path,required=True,help='CSV containing filename and an explicit 0/1 label')
    p.add_argument('--images',type=Path,required=True)
    p.add_argument('--label-column',default='label')
    p.add_argument('--threshold',type=float,required=True,help='Decision threshold selected on a separate validation set')
    p.add_argument('--output',type=Path,default=Path('reports/evaluation.json'))
    a=p.parse_args()
    if not 0 <= a.threshold <= 1: p.error('Threshold must lie in [0,1].')
    data=pd.read_csv(a.csv)
    if 'filename' not in data or a.label_column not in data: p.error('Required CSV columns are missing.')
    if data.empty or data.filename.duplicated().any(): p.error('CSV must be nonempty with unique filenames.')
    labels=pd.to_numeric(data[a.label_column],errors='coerce')
    if labels.isna().any() or not labels.isin([0,1]).all(): p.error('Labels must be exactly 0 or 1.')
    engine=Predictor(); model=engine.get_model(a.model); scores=[]
    image_root=a.images.resolve()
    for name in data.filename:
        path=(image_root/str(name)).resolve()
        if not path.is_relative_to(image_root): p.error('Image paths must remain inside --images.')
        image=load_rgb(path)
        scores.append(scalar_score(model(prepare_image(image,model,engine.config[a.model]['preprocessing']),training=False)))
    y=labels.to_numpy(dtype=int); s=np.asarray(scores); predicted=(s>=a.threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,predicted,labels=[0,1]).ravel()
    report={'model':a.model,'n':len(y),'threshold':a.threshold,
            'preprocessing':engine.config[a.model]['preprocessing'],
            'confusion_matrix':{'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)},
            'accuracy':float(np.mean(predicted==y)),
            'sensitivity':float(tp/(tp+fn)) if tp+fn else None,
            'specificity':float(tn/(tn+fp)) if tn+fp else None,
            'roc_auc':float(roc_auc_score(y,s)) if len(np.unique(y))==2 else None,
            'average_precision':float(average_precision_score(y,s)) if len(np.unique(y))==2 else None,
            'brier_score':float(brier_score_loss(y,s)),
            'note':'Metrics describe only supplied labels. Use patients excluded from training and threshold selection.'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
    data.assign(model_score=s,predicted_label=predicted).to_csv(a.output.with_suffix('.csv'),index=False)

if __name__=='__main__':
    main()
