import io
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
import tensorflow as tf
from app import create_app
from conftest import setup_account
from inference import GradCAM, Predictor, load_rgb, prepare_image, scalar_score


def example_bytes():
    a=np.random.default_rng(7).integers(0,256,(80,96,3),dtype=np.uint8)
    stream=io.BytesIO();Image.fromarray(a).save(stream,format='PNG');stream.seek(0);return stream


class StubEngine:
    def analyze_complete(self,image,output_dir,request_id,cam_target="positive"):
        scores = {}
        for key in ['stroke','ckd','hyp']:
            name = f'{request_id}_{key}.png'
            image.save(Path(output_dir)/name)
            scores[key] = dict(score=.42,display_score='42.0%',positive_label='test label',
                map_filename=name,heatmap_filename=None,explanation_status='available',cam_target=cam_target)
        return {'scores':scores, 'vessels':{'status':'unavailable','reason':'Test segmentation unavailable.'}}


def test_input_scale_is_explicit_and_rgb():
    image=Image.new('RGB',(80,80),(255,128,32))
    class Shape: input_shape=(None,40,40,3)
    raw=prepare_image(image,Shape(),'rgb_0_255'); scaled=prepare_image(image,Shape(),'rgb_0_1')
    np.testing.assert_array_equal(raw[0,0,0],[255,128,32])
    np.testing.assert_allclose(scaled,raw/255)
    with pytest.raises(ValueError): prepare_image(image,Shape(),'guess')


def test_supplied_model_contract():
    p=Predictor()
    assert p.config['hyp']['preprocessing']=='rgb_0_1'
    assert p.config['ckd']['preprocessing']=='rgb_0_255'
    assert p.config['stroke']['preprocessing']=='rgb_0_255'


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-.1,1.1])
def test_invalid_model_scores(value):
    with pytest.raises(ValueError): scalar_score([[value]])


def test_upload_validation_and_unique_outputs(tmp_path):
    app=create_app(StubEngine(),tmp_path);client=app.test_client()
    token=setup_account(client)
    client.environ_base['HTTP_X_CSRF_TOKEN']=token
    assert client.get('/').status_code==200
    assert client.post('/predict').status_code==400
    assert client.post('/predict',data={'file':(io.BytesIO(b'bad'),'x.jpg')}).status_code==400
    blank=io.BytesIO();Image.new('RGB',(80,80)).save(blank,format='PNG');blank.seek(0)
    assert client.post('/predict',data={'file':(blank,'blank.png')}).status_code==400
    first=client.post('/predict',data={'file':(example_bytes(),'../../same.png')})
    second=client.post('/predict',data={'file':(example_bytes(),'../../same.png')})
    assert first.status_code==second.status_code==200
    a,b=first.json,second.json
    assert a['raw_url']!=b['raw_url']
    assert a['scores']['ckd']['map_url']!=a['scores']['stroke']['map_url']
    assert len(list((tmp_path/'analyses').glob('*/input.png')))==2
    app.config['MAX_CONTENT_LENGTH']=100
    assert client.post('/predict',data={'file':(example_bytes(),'large.png')}).status_code==413


def test_failed_prediction_clears_saved_files(tmp_path):
    class Broken:
        def analyze_complete(self,*args,**kwargs): raise RuntimeError('private model path')
    app=create_app(Broken(),tmp_path)
    client=app.test_client();token=setup_account(client)
    response=client.post('/predict',headers={'X-CSRF-Token':token},data={'file':(example_bytes(),'sample.png')})
    assert response.status_code==500
    assert 'private model path' not in response.json['error']
    assert not list((tmp_path/'analyses').iterdir())


@pytest.mark.parametrize('nested',[False,True])
def test_gradcam_is_deterministic_and_preserves_probability(nested):
    inputs=tf.keras.Input((12,12,3))
    conv=tf.keras.layers.Conv2D(4,3,activation='relu',kernel_initializer='ones')(inputs)
    base=tf.keras.Model(inputs,conv)
    if nested:
        model=tf.keras.Sequential([base,tf.keras.layers.GlobalAveragePooling2D(),
                                  tf.keras.layers.Dropout(.8),tf.keras.layers.Dense(1,activation='sigmoid')])
    else:
        x=tf.keras.layers.GlobalAveragePooling2D()(conv)
        x=tf.keras.layers.Dropout(.8)(x)
        model=tf.keras.Model(inputs,tf.keras.layers.Dense(1,activation='sigmoid')(x))
    model.layers[-1].set_weights([np.ones((4,1),dtype=np.float32)*.001,np.zeros(1,dtype=np.float32)])
    batch=np.random.default_rng(42).random((1,12,12,3)).astype(np.float32)
    before=model(batch,training=False).numpy()
    explain=GradCAM(model);a=explain(batch);b=explain(batch)
    np.testing.assert_allclose(a,b,atol=1e-7)
    np.testing.assert_array_equal(before,model(batch,training=False).numpy())
    assert np.isfinite(a).all() and a.min()>=0 and a.max()<=1


def test_heatmap_failure_does_not_discard_score(tmp_path):
    p=Predictor();p.config={'hyp':p.config['hyp']}
    x=tf.keras.Input((10,10,3));y=tf.keras.layers.GlobalAveragePooling2D()(x)
    p.models['hyp']=tf.keras.Model(x,tf.keras.layers.Dense(1,activation='sigmoid')(y))
    result=p.analyze(load_rgb(example_bytes()),tmp_path,'case')['hyp']
    assert 0<=result['score']<=1
    assert result['map_filename'] is None and result['explanation_status']=='unavailable'


@pytest.mark.parametrize('shape',[(64,65),(256,256),(301,457)])
def test_vessel_patches_cover_all_edges(shape):
    from vessel_pipeline import segment_vessels
    class ConstantMask:
        input_shape=(None,256,256,3)
        def __call__(self,batch,training=False):
            return np.full((1,256,256,1),.7,np.float32)
    mask=segment_vessels(ConstantMask(),np.zeros((*shape,3),dtype=np.uint8))
    assert mask.shape==shape
    np.testing.assert_allclose(mask,.7,atol=1e-6)


def test_empty_vessels_return_unavailable():
    from vessel_pipeline import measure_width_proxy
    result=measure_width_proxy(np.zeros((128,128,3),np.uint8),np.zeros((128,128),np.float32))
    assert result['status']=='unavailable'
