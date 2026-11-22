"""Reproduce the gatekeeper input/polarity sanity check (not a population accuracy test)."""
import argparse
import csv
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
from gatekeeper import FundusGatekeeper,prepare_gate_input


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images',type=Path,default=Path(__file__).resolve().parents[1]/'test')
    p.add_argument('--reference-images',action='store_true',help='Also use scikit-image example photos; requires scikit-image')
    p.add_argument('--output',type=Path,default=Path('reports/gatekeeper_audit.csv'))
    args=p.parse_args()
    gate=FundusGatekeeper();examples=[]
    for path in sorted(args.images.glob('*.jpg')):
        with Image.open(path) as image:examples.append((path.name,'fundus',image.convert('RGB')))
    if not examples:p.error('No retinal JPG samples were found.')
    if args.reference_images:
        try:
            from skimage import data
        except ImportError:
            p.error('Install the optional audit dependency with: python -m pip install scikit-image==0.26.0')
        for name in ['astronaut','coffee','chelsea','rocket','camera','clock','coins','horse','immunohistochemistry','brick','grass','gravel','retina']:
            arr=getattr(data,name)()
            if arr.dtype==bool:arr=np.uint8(arr)*255
            examples.append(('skimage_'+name,'fundus' if name=='retina' else 'nonfundus',Image.fromarray(arr).convert('RGB')))
    controls=[('noise',np.random.default_rng(42).integers(0,256,(224,224,3),dtype=np.uint8)),
              ('black',np.zeros((224,224,3),np.uint8)),('white',np.full((224,224,3),255,np.uint8))]
    for name,arr in controls:examples.append((name,'nonfundus',Image.fromarray(arr)))
    rows=[]
    for name,kind,image in examples:
        configured=gate.check(image)
        row={'image':name,'kind':kind}
        for mode in ['rgb_0_255','rgb_0_1','mobilenet_v2']:
            row[mode]=float(gate.model(prepare_gate_input(image,mode),training=False).numpy()[0,0])
        row.update(fundus_score=configured['fundus_score'],accepted=configured['accepted'])
        rows.append(row)
        print(f"{name}: {kind}; {'accepted' if configured['accepted'] else 'rejected'}; fundus score {configured['fundus_score']:.8f}")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    print('Saved',args.output)
    print('Labels refer only to image type, not disease. This small sample check does not recover the original training contract or establish deployment accuracy.')

if __name__=='__main__':main()
