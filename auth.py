"""Local accounts, server-revocable sessions and CSRF protection for Flask."""
from functools import wraps
import hashlib
import hmac
import os
from pathlib import Path
import re
import secrets
import sqlite3
import time

import click
from flask import current_app, g, jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash


def db():
    if 'db' not in g:
        g.db = sqlite3.connect(current_app.config['DATABASE'], timeout=15)
        g.db.row_factory = sqlite3.Row
        g.db.execute('PRAGMA foreign_keys = ON')
    return g.db


def csrf_token():
    if 'csrf_token' not in session:
        session['csrf_token'] = secrets.token_hex(32)
    return session['csrf_token']


def require_csrf():
    supplied = request.headers.get('X-CSRF-Token') or request.form.get('csrf_token', '')
    expected = session.get('csrf_token', '')
    return bool(expected and supplied and hmac.compare_digest(expected.encode('utf-8'), supplied.encode('utf-8')))


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if g.user is None:
            if request.endpoint == 'predict' or request.endpoint == 'media':
                return jsonify(error='Please log in to continue.', login_url=url_for('login')), 401
            return redirect(url_for('login'))
        return view(*args, **kwargs)
    return wrapped


def validate_account(username, password):
    username = username.strip().lower()
    if not re.fullmatch(r'[a-z0-9_.-]{3,40}', username):
        raise ValueError('Use 3–40 letters, numbers, dots, underscores or hyphens for the username.')
    if not 12 <= len(password) <= 128:
        raise ValueError('Choose a password with 12–128 characters.')
    return username


def create_user(username, password):
    username = validate_account(username, password)
    with db() as connection:
        connection.execute('INSERT INTO users(username,password_hash) VALUES (?,?)',
                           (username, generate_password_hash(password)))
    return username


def establish_session(user_id):
    session.clear()
    token = secrets.token_urlsafe(32)
    session['auth_token'] = token
    session.permanent = True
    now = time.time()
    with db() as connection:
        connection.execute('DELETE FROM sessions WHERE expires_at < ?', (now,))
        connection.execute('INSERT INTO sessions(token_hash,user_id,expires_at) VALUES (?,?,?)',
                           (hashlib.sha256(token.encode()).hexdigest(), user_id, now + 7200))
    csrf_token()


def init_auth(app):
    path = Path(app.config['DATABASE'])
    path.parent.mkdir(parents=True, exist_ok=True)
    if not app.config.get('SECRET_KEY'):
        key_path = path.parent / 'session_secret'
        try:
            fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(fd, 'w') as stream:
                stream.write(secrets.token_hex(32))
        app.config['SECRET_KEY'] = key_path.read_text().strip()
        if len(app.config['SECRET_KEY']) < 32:
            raise RuntimeError('Invalid session secret. Restore the local session_secret file.')
    with sqlite3.connect(path) as connection:
        connection.executescript('''
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id),
                expires_at REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS login_attempts (
                ip TEXT NOT NULL, attempted_at REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS analyses (
                id TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id), created_at REAL NOT NULL);
        ''')
    try:
        path.chmod(0o600)
    except OSError:
        pass
    app.jinja_env.globals['csrf_token'] = csrf_token
    # One dummy hash avoids revealing account existence through fast password checks.
    dummy_hash = generate_password_hash(secrets.token_urlsafe(24))

    @app.teardown_appcontext
    def close_db(_exception):
        connection = g.pop('db', None)
        if connection is not None:
            connection.close()

    @app.before_request
    def identify_user():
        g.user = None
        token = session.get('auth_token')
        if token:
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            g.user = db().execute('''SELECT users.id, users.username FROM sessions
                JOIN users ON users.id=sessions.user_id WHERE token_hash=? AND expires_at>?''',
                (token_hash, time.time())).fetchone()
            if g.user is None:
                session.clear()
        # Previous releases put data under static/. Never expose it through this app.
        if request.endpoint == 'static':
            filename = (request.view_args or {}).get('filename', '')
            if filename not in {'style.css', 'script.js', 'report.js'}:
                return jsonify(error='Static resource not found.'), 404

    @app.route('/setup', methods=['GET', 'POST'])
    def setup():
        if db().execute('SELECT 1 FROM users LIMIT 1').fetchone():
            return redirect(url_for('login'))
        error = None
        if request.method == 'POST':
            if not require_csrf():
                return render_template('login.html', setup=True, error='Session expired. Reload and try again.'), 400
            username = request.form.get('username', '')
            password = request.form.get('password', '')
            try:
                username = validate_account(username, password)
                if password != request.form.get('confirm_password'):
                    raise ValueError('Passwords do not match.')
                # Transaction prevents two simultaneous requests creating separate first owners.
                connection = db()
                connection.execute('BEGIN IMMEDIATE')
                if connection.execute('SELECT 1 FROM users LIMIT 1').fetchone():
                    connection.rollback()
                    return redirect(url_for('login'))
                cursor = connection.execute('INSERT INTO users(username,password_hash) VALUES (?,?)',
                                            (username, generate_password_hash(password)))
                connection.commit()
                establish_session(cursor.lastrowid)
                return redirect(url_for('index'))
            except ValueError as exc:
                error = str(exc)
        return render_template('login.html', setup=True, error=error), 400 if error else 200

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if not db().execute('SELECT 1 FROM users LIMIT 1').fetchone():
            return redirect(url_for('setup'))
        if g.user is not None:
            return redirect(url_for('index'))
        error = None
        if request.method == 'POST':
            if not require_csrf():
                return render_template('login.html', setup=False, error='Session expired. Reload and try again.'), 400
            ip = request.remote_addr or 'local'
            now = time.time()
            # Reserve the attempt in a write transaction so simultaneous requests cannot bypass it.
            connection = db(); connection.execute('BEGIN IMMEDIATE')
            connection.execute('DELETE FROM login_attempts WHERE attempted_at < ?', (now - 900,))
            count = connection.execute('SELECT count(*) FROM login_attempts WHERE ip=?', (ip,)).fetchone()[0]
            if count >= 10:
                connection.commit()
                return render_template('login.html', setup=False,
                    error='Too many login attempts. Try again in 15 minutes.'), 429
            connection.execute('INSERT INTO login_attempts VALUES (?,?)', (ip, now)); connection.commit()
            username = request.form.get('username', '').strip().lower()
            password = request.form.get('password', '')
            user = connection.execute('SELECT * FROM users WHERE username=?', (username,)).fetchone()
            valid = check_password_hash(user['password_hash'] if user else dummy_hash, password[:129])
            if user and valid and len(password) <= 128:
                with connection:
                    connection.execute('DELETE FROM login_attempts WHERE ip=?', (ip,))
                establish_session(user['id'])
                return redirect(url_for('index'))
            error = 'Incorrect username or password.'
        return render_template('login.html', setup=False, error=error), 401 if error else 200

    @app.post('/logout')
    @login_required
    def logout():
        if not require_csrf():
            return jsonify(error='Invalid session token. Reload the page.'), 400
        token_hash = hashlib.sha256(session['auth_token'].encode()).hexdigest()
        with db() as connection:
            connection.execute('DELETE FROM sessions WHERE token_hash=?', (token_hash,))
        session.clear()
        return redirect(url_for('login'))

    @app.cli.command('create-user')
    @click.argument('username')
    @click.password_option(confirmation_prompt=True)
    def create_user_command(username, password):
        """Create another local account from the owner's terminal."""
        try:
            name = create_user(username, password)
        except (ValueError, sqlite3.IntegrityError) as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f'Created account: {name}')

    @app.cli.command('reset-password')
    @click.argument('username')
    @click.password_option(confirmation_prompt=True)
    def reset_password(username, password):
        """Reset a password locally and revoke that account's active sessions."""
        try:
            username = validate_account(username, password)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        user = db().execute('SELECT id FROM users WHERE username=?', (username,)).fetchone()
        if user is None:
            raise click.ClickException('Account not found.')
        with db() as connection:
            connection.execute('UPDATE users SET password_hash=? WHERE id=?',
                               (generate_password_hash(password), user['id']))
            connection.execute('DELETE FROM sessions WHERE user_id=?', (user['id'],))
        click.echo('Password updated and previous sessions revoked.')
