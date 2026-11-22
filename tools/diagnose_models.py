"""Compare legacy and configured inputs. Unlabelled samples cannot measure accuracy."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from inference import Predictor, load_rgb, prepare_image, scalar_score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, default=Path(__file__).resolve().parents[1] / 'test')
    parser.add_argument('--output', type=Path, default=Path('reports/prediction_audit.csv'))
    args = parser.parse_args()
    engine = Predictor()
    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.tif','.tiff','.bmp','.webp'})
    if not paths:
        parser.error('No sample images found.')
    rows=[]
    for key, spec in engine.config.items():
        model = engine.get_model(key)
        for path in paths:
            image=load_rgb(path)
            raw=scalar_score(model(prepare_image(image,model,'rgb_0_255'), training=False))
            scaled=scalar_score(model(prepare_image(image,model,'rgb_0_1'), training=False))
            row={'image':path.name,'model':key,'original_app_score':raw,
                 'rgb_0_1_score':scaled,'configured_preprocessing':spec['preprocessing'],
                 'configured_score':scaled if spec['preprocessing']=='rgb_0_1' else raw}
            rows.append(row); print(json.dumps(row),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    print('Saved',args.output)
    print('Filenames are not verified ground truth. Do not choose scaling by which result looks plausible.')

if __name__=='__main__':
    main()
