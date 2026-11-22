"""Run the original U-Net on an image using complete overlapping patch coverage."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
from inference import ROOT,load_rgb
from vessel_pipeline import segment_vessels


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('image',type=Path)
    p.add_argument('--model',type=Path,default=ROOT/'unet_vessel_segmentation.keras')
    p.add_argument('--output',type=Path,default=Path('vessel_segmentation_result.png'))
    a=p.parse_args();rgb=np.asarray(load_rgb(a.image))
    model=tf.keras.models.load_model(a.model,compile=False)
    probability=segment_vessels(model,rgb)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(np.uint8((probability>.5)*255)).save(a.output)
    np.save(a.output.with_suffix('.npy'),probability)
    print('Saved binary mask:',a.output)

if __name__=='__main__': main()
