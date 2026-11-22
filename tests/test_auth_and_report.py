import hashlib
import io
import sqlite3
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import tensorflow as tf
from app import create_app
from auth import create_user,db
from clinical_report import build_interpretation
from conftest import csrf,setup_account,login,PASSWORD
from test_pipeline import StubEngine,example_bytes
from inference import GradCAM,save_overlay,Predictor


def test_login_setup_csrf_logout_and_revocation(tmp_path):
    app=create_app(StubEngine(),tmp_path);client=app.test_client()
    assert client.get('/').status_code==302
    assert client.post('/predict').status_code==401
    assert client.post('/setup',data={'username':'owner','password':PASSWORD}).status_code==400
    token=setup_account(client)
    assert client.get('/').status_code==200
    assert client.get('/setup').status_code==302
    assert client.post('/predict',data={'file':(example_bytes(),'one.png')}).status_code==400
    assert client.post('/logout').status_code==400
    old_cookie=client.get_cookie('session').value
    assert client.post('/logout',data={'csrf_token':token}).status_code==302
    assert client.get('/').status_code==302
    # Replaying a signed old cookie after logout must not resurrect its server session.
    client.set_cookie('session',old_cookie)
    assert client.get('/').status_code==302
    assert login(client,password='wrong-password-123').status_code==401
    assert login(client).status_code==302
    assert client.get('/').status_code==200
    with sqlite3.connect(tmp_path/'accounts.sqlite3') as conn:
        stored=conn.execute('SELECT password_hash FROM users').fetchone()[0]
        assert stored!=PASSWORD and stored.startswith('scrypt:')
        conn.execute('UPDATE sessions SET expires_at=0')
    assert client.get('/').status_code==302


def test_rate_limit_and_setup_password_rules(tmp_path):
    app=create_app(StubEngine(),tmp_path);client=app.test_client()
    token=csrf(client,'/setup')
    assert client.post('/setup',data={'csrf_token':token,'username':'owner','password':'short','confirm_password':'short'}).status_code==400
    token=setup_account(client)
    client.post('/logout',data={'csrf_token':token})
    for _ in range(10):
        assert login(client,password='incorrect password').status_code==401
    assert login(client,password='incorrect password').status_code==429


def test_private_media_and_report_isolation(tmp_path):
    app=create_app(StubEngine(),tmp_path);one=app.test_client();two=app.test_client()
    token=setup_account(one)
    with app.app_context():create_user('second',PASSWORD)
    assert login(two,'second').status_code==302
    result=one.post('/predict',headers={'X-CSRF-Token':token},data={'file':(example_bytes(),'../../input.png')})
    assert result.status_code==200,result.json
    data=result.json
    urls=[data['raw_url'],data['scores']['ckd']['map_url'],data['report_url'],data['report_download_url']]
    outsider=app.test_client()
    for url in urls:
        assert one.get(url).status_code==200
        assert two.get(url).status_code==404
        assert outsider.get(url).status_code in {401,302}
        assert one.get(url).headers['Cache-Control']=='no-store, private'
    assert one.get('/static/uploads/ckd_fundal.jpg').status_code==404
    assert one.get('/static/./uploads/ckd_fundal.jpg').status_code==404
    assert one.get('/static/style.css/../uploads/ckd_fundal.jpg').status_code==404
    assert one.get('/static/results/ckd_map.jpg').status_code==404
    assert one.get('/static/heatmaps/cam_effnet_347_right.jpg').status_code==404
    assert one.get('/media/'+data['request_id']+'/report.json').status_code==404
    assert one.get('/media/not-a-valid-id/input.png').status_code==404
    assert one.get('/').headers['X-Frame-Options']=='DENY'


def test_auth_persists_across_app_restart(tmp_path):
    app=create_app(StubEngine(),tmp_path);client=app.test_client();setup_account(client)
    secret=app.config['SECRET_KEY']
    app2=create_app(StubEngine(),tmp_path)
    assert app2.config['SECRET_KEY']==secret
    assert login(app2.test_client()).status_code==302


def test_cli_password_reset_revokes_sessions(tmp_path):
    app=create_app(StubEngine(),tmp_path);client=app.test_client();setup_account(client)
    result=app.test_cli_runner().invoke(args=['reset-password','owner'],input='Replacement password 123!\nReplacement password 123!\n')
    assert result.exit_code==0,result.output
    assert client.get('/').status_code==302
    assert login(client,password='Replacement password 123!').status_code==302


def test_gradcam_target_reversal_and_no_score_mutation():
    x=tf.keras.Input((8,8,3));conv=tf.keras.layers.Conv2D(2,1,use_bias=False)(x)
    pooled=tf.keras.layers.GlobalAveragePooling2D()(conv)
    model=tf.keras.Model(x,tf.keras.layers.Dense(1,activation='sigmoid',use_bias=False)(pooled))
    kernel=np.zeros((1,1,3,2),np.float32);kernel[0,0,0,0]=1;kernel[0,0,1,1]=1
    model.layers[1].set_weights([kernel]);model.layers[-1].set_weights([np.array([[1.],[-1.]],np.float32)])
    batch=np.zeros((1,8,8,3),np.float32);batch[:,:,:4,0]=1;batch[:,:,4:,1]=1
    before=model(batch,training=False).numpy()
    cam=GradCAM(model);pos=cam(batch,'positive');neg=cam(batch,'negative')
    assert pos[:,:4].mean()>.9 and pos[:,4:].max()==0
    assert neg[:,4:].mean()>.9 and neg[:,:4].max()==0
    np.testing.assert_array_equal(before,model(batch,training=False).numpy())
    with pytest.raises(ValueError):cam(batch,'healthy')


def test_overlay_transparency_and_empty_map(tmp_path):
    original=np.full((8,8,3),100,np.uint8);image=Image.fromarray(original)
    heat=np.zeros((8,8),np.float32);heat[:,:4]=1
    assert save_overlay(image,heat,tmp_path/'map.png')
    rendered=np.asarray(Image.open(tmp_path/'map.png'))
    np.testing.assert_array_equal(rendered[:,4:],original[:,4:])
    assert not np.array_equal(rendered[:,:4],original[:,:4])
    assert not save_overlay(image,np.zeros_like(heat),tmp_path/'empty.png')
    assert not (tmp_path/'empty.png').exists()


def test_unet_failure_preserves_classifier_results(tmp_path):
    engine=Predictor()
    class AcceptedGate:
        def require_fundus(self,image):return {'accepted':True}
    engine.gatekeeper=AcceptedGate()
    engine.analyze=lambda *a,**kw:{'hyp':{'score':.4}}
    def broken(*a,**kw):raise OSError('unet file missing')
    engine.analyze_vessels=broken
    result=engine.analyze_complete(Image.new('RGB',(64,64)),tmp_path,'case')
    assert result['scores']['hyp']['score']==.4
    assert result['vessels']['status']=='unavailable'


@pytest.mark.parametrize('score',[0.,.5,1.])
def test_interpretation_never_invents_diagnosis_or_threshold(score):
    scores={key:{'score':score,'explanation_status':'no_target_activation','cam_target':'positive'}
            for key in ['hyp','ckd','stroke']}
    vessels={'status':'available','processed_size':[512,512],'vessel_area_percent':8.5,
             'vessel_pixels':1200,'width_proxy':{'status':'available','width_ratio_proxy':.4}}
    report=build_interpretation(scores,vessels)
    assert 'cannot confirm or exclude' in report['caution']
    assert f'hypertension {score*100:.1f}%' in report['summary']
    assert '8.50%' in report['summary']
    assert any('8.50%' in f['text'] for f in report['findings'])
    assert any('not a clinical AVR' in f['text'] for f in report['findings'])
    assert any('do not infer normality' in s for s in report['limitations'])
    assert len(report['references'])==3
    partial=build_interpretation(scores,{'status':'unavailable'})
    assert any('incomplete' in s for s in partial['limitations'])


def test_non_ascii_csrf_is_rejected_and_background_attribution_is_disclosed(tmp_path):
    app=create_app(StubEngine(),tmp_path);client=app.test_client();setup_account(client)
    assert client.post('/predict',data={'csrf_token':'incorrect-é'}).status_code==400
    scores={key:{'score':.1,'explanation_status':'available','cam_target':'positive',
                 'cam_outside_field_percent':75.0} for key in ['hyp','ckd','stroke']}
    report=build_interpretation(scores,{'status':'unavailable'})
    assert sum('outside the estimated retinal field' in s for s in report['limitations'])==3


def test_image_interpretation_changes_with_actual_outputs():
    scores={key:{'score':.2,'explanation_status':'available','cam_target':'positive'} for key in ['hyp','ckd','stroke']}
    first=build_interpretation(scores,{'status':'unavailable'})
    scores['hyp']['score']=.831
    vessels={'status':'available','processed_size':[512,512],'vessel_area_percent':8.36,
             'vessel_pixels':16952,'width_proxy':{'status':'unavailable'}}
    second=build_interpretation(scores,vessels)
    assert first['summary']!=second['summary']
    assert 'hypertension 83.1%' in second['summary'] and '8.36%' in second['summary']
    assert 'hypertension 20.0%' in first['summary']
    assert 'not calibrated clinical risk' in second['caution']
