"""Inspect the existing experimental vessel-width proxy; no disease diagnosis."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
from inference import ROOT,load_rgb
from vessel_pipeline import segment_vessels,measure_width_proxy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('image',type=Path)
    p.add_argument('--model',type=Path,default=ROOT/'unet_vessel_segmentation.keras')
    p.add_argument('--method',choices=['halves','kmeans'],default='halves')
    p.add_argument('--output',type=Path,default=Path('reports/vessel_width_proxy.json'))
    a=p.parse_args();rgb=np.asarray(load_rgb(a.image))
    model=tf.keras.models.load_model(a.model,compile=False)
    probability=segment_vessels(model,rgb)
    report=measure_width_proxy(rgb,probability,a.method)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2))
    Image.fromarray(np.uint8((probability>.5)*255)).save(a.output.with_suffix('.png'))
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
