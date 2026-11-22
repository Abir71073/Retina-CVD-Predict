"""Evaluate the original K-means vessel-width heuristic on an explicit CSV subset.
The supplied CSV must be held out from model development. No class resampling.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from inference import ROOT,load_rgb
from vessel_pipeline import segment_vessels,measure_width_proxy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',type=Path,required=True)
    p.add_argument('--images',type=Path,required=True)
    p.add_argument('--label-column',default='H')
    p.add_argument('--threshold',type=float,required=True,help='Experimental width cutoff, not a clinical constant')
    p.add_argument('--output',type=Path,default=Path('reports/width_proxy_evaluation.json'))
    a=p.parse_args()
    if not 0<a.threshold<1:p.error('Threshold must be between zero and one.')
    df=pd.read_csv(a.csv)
    if 'filename' not in df or a.label_column not in df:p.error('Missing filename or label column.')
    labels=pd.to_numeric(df[a.label_column],errors='coerce')
    if df.empty or not labels.isin([0,1]).all() or df.filename.duplicated().any():p.error('Require unique files and binary labels.')
    model=tf.keras.models.load_model(ROOT/'unet_vessel_segmentation.keras',compile=False)
    truth=[];pred=[];failed=[];rows=[]
    for i,row in df.iterrows():
        path=(a.images/str(row['filename'])).resolve()
        if not path.is_relative_to(a.images.resolve()):p.error('Filename escapes image directory.')
        try:
            image=load_rgb(path);image=image.resize((512,512),Image.Resampling.BILINEAR)
            rgb=np.asarray(image)
            result=measure_width_proxy(rgb,segment_vessels(model,rgb),'kmeans')
            if result['status']!='available':raise ValueError(result['reason'])
            score=result['width_ratio_proxy'];label=int(score<a.threshold)
            truth.append(int(labels.iloc[i]));pred.append(label)
            rows.append({'filename':row['filename'],'label':int(labels.iloc[i]),'width_ratio_proxy':score,'prediction':label})
        except (ValueError,OSError) as exc: failed.append({'filename':row['filename'],'reason':str(exc)})
    report={'attempted':len(df),'measured':len(truth),'skipped':failed,'threshold':a.threshold,
            'confusion_matrix':confusion_matrix(truth,pred,labels=[0,1]).tolist() if truth else None,
            'accuracy_on_measurable_subset':float(np.mean(np.asarray(truth)==pred)) if truth else None,
            'note':'Exploratory width proxy, not verified AVR. Skipped images are reported, not counted as correct.'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2))
    pd.DataFrame(rows).to_csv(a.output.with_suffix('.csv'),index=False);print(json.dumps(report,indent=2))

from PIL import Image
if __name__=='__main__':main()
