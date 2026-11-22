"""Integration checks against all five supplied checkpoint files (CPU)."""
from pathlib import Path
import numpy as np
import tensorflow as tf
from app import create_app
from conftest import setup_account
from inference import ROOT,Predictor,load_rgb,prepare_image,scalar_score
from vessel_pipeline import segment_vessels


def test_real_specialists_and_web_endpoint(tmp_path):
    engine=Predictor();app=create_app(engine,tmp_path);client=app.test_client()
    client.environ_base['HTTP_X_CSRF_TOKEN']=setup_account(client)
    urls=[];generated=[]
    try:
        for name in ['normal_fundal_wikimedia.jpg','ckd_hyp_fundal.jpg']:
            with (ROOT/'test'/name).open('rb') as stream:
                response=client.post('/predict',data={'file':(stream,'same.jpg')})
            assert response.status_code==200,response.json
            data=response.json;urls.append(data['raw_url'])
            assert data['gatekeeper']['accepted'] is True
            generated.append(tmp_path/'analyses'/data['request_id']/'input.png')
            assert client.get(data['raw_url']).status_code==200
            for key in ['stroke','ckd','hyp']:
                result=data['scores'][key]
                assert 0<=result['score']<=1
                assert result['explanation_status']=='available'
                assert client.get(result['map_url']).status_code==200
                generated.append(tmp_path/'analyses'/data['request_id']/result['map_url'].split('/')[-1])
                assert result['cam_layer']=='top_activation'
                assert client.get(result['heatmap_url']).status_code==200
            vessels=data['vessels']
            assert vessels['status']=='available',vessels
            assert client.get(vessels['mask_url']).status_code==200
            assert client.get(vessels['overlay_url']).status_code==200
            assert 0<vessels['vessel_area_percent']<100
            assert max(vessels['processed_size'])<=768
            assert client.get(data['report_url']).status_code==200
            assert client.get(data['report_download_url']).status_code==200
            assert any(f['title']=='U-Net vessel findings' for f in data['interpretation']['findings'])
            if name=='ckd_hyp_fundal.jpg':
                # Regression for the demonstrated all-zero hypertension input mismatch.
                image=load_rgb(ROOT/'test'/name);model=engine.get_model('hyp')
                old=scalar_score(model(prepare_image(image,model,'rgb_0_255'),training=False))
                assert old<1e-6 and data['scores']['hyp']['score']>.8
        assert urls[0]!=urls[1]
    finally:
        for path in generated:path.unlink(missing_ok=True)


def test_remaining_checkpoints_load_and_unet_is_finite():
    # Gatekeeper contract is checked separately on retinal and non-retinal samples.
    gate=tf.keras.models.load_model(ROOT/'fundus_gatekeeper_v2_massive.keras',compile=False)
    assert gate.input_shape==(None,224,224,3) and gate.output_shape==(None,1)
    unet=tf.keras.models.load_model(ROOT/'unet_vessel_segmentation.keras',compile=False)
    image=load_rgb(ROOT/'test'/'normal_fundal_wikimedia.jpg').resize((97,103))
    mask=segment_vessels(unet,np.asarray(image))
    assert mask.shape==(103,97) and np.isfinite(mask).all()
    assert mask.min()>=0 and mask.max()<=1
