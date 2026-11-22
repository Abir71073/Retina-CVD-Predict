"""Authenticated Flask interface for the existing CNN and U-Net streams."""
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import shutil
import time
from uuid import uuid4

from flask import Flask, g, jsonify, render_template, request, send_from_directory, url_for
from werkzeug.exceptions import RequestEntityTooLarge

from auth import db, init_auth, login_required, require_csrf
from clinical_report import build_interpretation
from inference import Predictor, ROOT, load_rgb
from gatekeeper import FundusRejected, GatekeeperUnavailable


def create_app(predictor=None, storage_root=None, test_config=None):
    app = Flask(__name__)
    storage = Path(storage_root or os.environ.get('RETINA_STORAGE_DIR') or ROOT / 'instance').resolve()
    app.config.update(
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,
        DATABASE=str(storage / 'accounts.sqlite3'), ANALYSIS_FOLDER=storage / 'analyses',
        SECRET_KEY=os.environ.get('APP_SECRET_KEY'), SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE='Lax', SESSION_COOKIE_SECURE=os.environ.get('COOKIE_SECURE') == '1',
        PERMANENT_SESSION_LIFETIME=timedelta(hours=2),
    )
    if test_config:
        app.config.update(test_config)
    app.config['ANALYSIS_FOLDER'].mkdir(parents=True, exist_ok=True)
    init_auth(app)
    engine = predictor if predictor is not None else Predictor()
    app.extensions['predictor'] = engine

    @app.after_request
    def security_headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['Referrer-Policy'] = 'same-origin'
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' blob: data:; connect-src 'self'; base-uri 'none'; "
            "frame-ancestors 'none'; form-action 'self'")
        if request.endpoint != 'static':
            response.headers['Cache-Control'] = 'no-store, private'
        return response

    @app.get('/')
    @login_required
    def index():
        return render_template('index.html')

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        return jsonify(error='Upload exceeds the 16 MB limit.'), 413

    def owned_analysis(analysis_id):
        if not re.fullmatch(r'[0-9a-f]{32}', analysis_id):
            return False
        return db().execute('SELECT 1 FROM analyses WHERE id=? AND user_id=?',
                            (analysis_id, g.user['id'])).fetchone() is not None

    @app.get('/media/<analysis_id>/<filename>')
    @login_required
    def media(analysis_id, filename):
        if not owned_analysis(analysis_id):
            return jsonify(error='Analysis not found.'), 404
        if filename != 'input.png' and not re.fullmatch(analysis_id + r'_[a-z_]+\.png', filename):
            return jsonify(error='Image not found.'), 404
        return send_from_directory(app.config['ANALYSIS_FOLDER'] / analysis_id, filename)

    @app.get('/report/<analysis_id>')
    @login_required
    def report(analysis_id):
        if not owned_analysis(analysis_id):
            return jsonify(error='Analysis not found.'), 404
        path = app.config['ANALYSIS_FOLDER'] / analysis_id / 'report.json'
        if not path.is_file():
            return jsonify(error='Report not found.'), 404
        data = json.loads(path.read_text())
        return render_template('report.html', data=data)

    @app.get('/report/<analysis_id>/download')
    @login_required
    def report_download(analysis_id):
        if not owned_analysis(analysis_id):
            return jsonify(error='Analysis not found.'), 404
        return send_from_directory(app.config['ANALYSIS_FOLDER'] / analysis_id, 'report.json',
                                   as_attachment=True, download_name=f'retinal_report_{analysis_id[:8]}.json')

    @app.post('/predict')
    @login_required
    def predict():
        if not require_csrf():
            return jsonify(error='Session token expired. Reload the page and try again.'), 400
        target = request.form.get('cam_target', 'positive')
        if target not in {'positive', 'negative'}:
            return jsonify(error='Choose a valid Grad-CAM explanation target.'), 400
        upload = request.files.get('file')
        if upload is None or not upload.filename:
            return jsonify(error='Please select an image to upload.'), 400
        try:
            image = load_rgb(upload.stream)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        analysis_id = uuid4().hex
        directory = app.config['ANALYSIS_FOLDER'] / analysis_id
        directory.mkdir()
        try:
            image.save(directory / 'input.png')
            analysis = engine.analyze_complete(image, directory, analysis_id, cam_target=target)
            scores, vessels = analysis['scores'], analysis['vessels']
            interpretation = build_interpretation(scores, vessels, analysis.get('gatekeeper'))
            for result in scores.values():
                for source, dest in [('map_filename', 'map_url'), ('heatmap_filename', 'heatmap_url')]:
                    name = result.pop(source, None)
                    result[dest] = url_for('media', analysis_id=analysis_id, filename=name) if name else None
            for source, dest in [('mask_filename', 'mask_url'), ('overlay_filename', 'overlay_url'),
                                 ('probability_filename', 'probability_url')]:
                name = vessels.pop(source, None)
                vessels[dest] = url_for('media', analysis_id=analysis_id, filename=name) if name else None
            response = dict(
                status='success', request_id=analysis_id, scores=scores, vessels=vessels,
                gatekeeper=analysis.get('gatekeeper'),
                raw_url=url_for('media', analysis_id=analysis_id, filename='input.png'),
                cam_target=target, interpretation=interpretation,
                report_url=url_for('report', analysis_id=analysis_id),
                report_download_url=url_for('report_download', analysis_id=analysis_id),
                created_at_utc=datetime.now(timezone.utc).isoformat(),
            )
            (directory / 'report.json').write_text(json.dumps(response, indent=2))
            with db() as connection:
                connection.execute('INSERT INTO analyses VALUES (?,?,?)', (analysis_id, g.user['id'], time.time()))
            return jsonify(response)
        except FundusRejected as exc:
            shutil.rmtree(directory, ignore_errors=True)
            return jsonify(status='rejected', code='non_fundus_image', error=str(exc), gatekeeper=exc.result), 422
        except GatekeeperUnavailable as exc:
            shutil.rmtree(directory, ignore_errors=True)
            return jsonify(status='unavailable', code='gatekeeper_unavailable', error=str(exc)), 503
        except Exception:
            app.logger.exception('Prediction failed for request %s', analysis_id)
            shutil.rmtree(directory, ignore_errors=True)
            return jsonify(error='Analysis failed. Check the server log for model or environment errors.'), 500

    return app


app = create_app()

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=int(os.environ.get('PORT', '5000')), debug=False)
