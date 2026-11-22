import re

PASSWORD = 'Test account password 2026!'


def csrf(client, path='/login'):
    response = client.get(path, follow_redirects=True)
    match = re.search(rb'name="csrf-token" content="([^"]+)"', response.data)
    assert match, response.data[:200]
    return match.group(1).decode()


def setup_account(client, username='owner'):
    token = csrf(client, '/setup')
    response = client.post('/setup', data={'csrf_token': token, 'username': username,
        'password': PASSWORD, 'confirm_password': PASSWORD})
    assert response.status_code == 302
    return csrf(client, '/')


def login(client, username='owner', password=PASSWORD):
    token = csrf(client)
    return client.post('/login', data={'csrf_token':token,'username':username,'password':password})
