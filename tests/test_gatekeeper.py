import io
import json
import numpy as np
from PIL import Image
import pytest
from app import create_app
from auth import db
from conftest import setup_account
from gatekeeper import FundusGatekeeper,FundusRejected,GatekeeperUnavailable,prepare_gate_input
from inference import Predictor,ROOT,load_rgb
from clinical_report import build_interpretation
from test_pipeline import example_bytes


def configured_gate(raw):
    gate=FundusGatekeeper()
    gate.config={'preprocessing':'rgb_0_1','fundus_class_index':0,'accept_threshold':.5,
                 'contract_status':'test'}
    class Fake:
        def __call__(self,batch,training=False):
            assert batch.shape==(1,224,224,3)
            assert batch.dtype==np.float32 and 0<=batch.min()<=batch.max()<=1
            assert training is False
            return np.array([[raw]])
    gate.model=Fake()
    return gate


def test_gate_preprocessing_preserves_rgb_and_scale():
    batch=prepare_gate_input(Image.new('RGB',(300,200),(255,128,0)),'rgb_0_1')
    assert batch.shape==(1,224,224,3)
    np.testing.assert_allclose(batch[0,0,0],[1,128/255,0])


@pytest.mark.parametrize('raw,accepted',[(.01,True),(.49,True),(.5,False),(.51,False),(.99,False)])
def test_gate_polarity_and_tie_rejection(raw,accepted):
    result=configured_gate(raw).check(Image.new('RGB',(100,100)))
    assert result['accepted']==accepted
    assert result['fundus_score']==pytest.approx(1-raw)


@pytest.mark.parametrize('raw',[float('nan'),float('inf'),-.1,1.1])
def test_bad_gate_outputs_stop_analysis(raw):
    with pytest.raises(GatekeeperUnavailable):configured_gate(raw).check(Image.new('RGB',(100,100)))


def test_missing_gate_fails_closed(tmp_path):
    with pytest.raises(GatekeeperUnavailable):FundusGatekeeper(tmp_path).check(Image.new('RGB',(100,100)))


def test_gate_runs_before_cnn_and_unet(tmp_path):
    engine=Predictor();engine.gatekeeper=configured_gate(.99)
    def forbidden(*args,**kwargs):pytest.fail('Downstream model was called after rejection')
    engine.analyze=forbidden;engine.analyze_vessels=forbidden
    with pytest.raises(FundusRejected):engine.analyze_complete(Image.new('RGB',(100,100)),tmp_path,'rejected')
    engine.gatekeeper=configured_gate(float('nan'))
    with pytest.raises(GatekeeperUnavailable):engine.analyze_complete(Image.new('RGB',(100,100)),tmp_path,'broken')


@pytest.mark.parametrize('raw,status,code',[(.99,422,'non_fundus_image'),(float('nan'),503,'gatekeeper_unavailable')])
def test_rejected_or_unavailable_gate_creates_no_report(tmp_path,raw,status,code):
    engine=Predictor();engine.gatekeeper=configured_gate(raw)
    def forbidden(*args,**kwargs):pytest.fail('Specialist or U-Net should not run')
    engine.analyze=forbidden;engine.analyze_vessels=forbidden
    app=create_app(engine,tmp_path);client=app.test_client();token=setup_account(client)
    result=client.post('/predict',headers={'X-CSRF-Token':token},data={'file':(example_bytes(),'photo.png')})
    assert result.status_code==status,result.json
    assert result.json['code']==code
    assert 'scores' not in result.json and 'interpretation' not in result.json
    assert list((tmp_path/'analyses').iterdir())==[]
    with app.app_context():assert db().execute('SELECT count(*) FROM analyses').fetchone()[0]==0


def test_rejected_image_cannot_have_clinical_interpretation():
    with pytest.raises(ValueError):build_interpretation({}, {}, {'accepted':False})


def test_actual_checkpoint_accepts_retinas_and_rejects_noise():
    gate=FundusGatekeeper()
    for path in sorted((ROOT/'test').glob('*.jpg')):
        result=gate.check(load_rgb(path))
        assert result['accepted'],(path.name,result)
        assert result['fundus_score']>.99
    noise=Image.fromarray(np.random.default_rng(42).integers(0,256,(224,224,3),dtype=np.uint8))
    assert not gate.check(noise)['accepted']


def test_actual_nonfundus_upload_never_loads_specialists(tmp_path):
    engine=Predictor();app=create_app(engine,tmp_path);client=app.test_client();token=setup_account(client)
    stream=io.BytesIO();Image.fromarray(np.random.default_rng(42).integers(0,256,(224,224,3),dtype=np.uint8)).save(stream,format='PNG');stream.seek(0)
    result=client.post('/predict',headers={'X-CSRF-Token':token},data={'file':(stream,'nonfundus.png')})
    assert result.status_code==422,result.json
    assert engine.models=={} and engine.unet is None and engine.explainers=={}
    assert list((tmp_path/'analyses').iterdir())==[]
