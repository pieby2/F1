from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

health = client.get('/health')
print('health', health.status_code, health.json())

seasons = client.get('/seasons')
print('seasons_status', seasons.status_code)
print('seasons_payload', seasons.json())

payload = seasons.json()
last_season = payload['seasons'][-1]

events = client.get(f'/events/{last_season}')
print('events_status', events.status_code)
print('events_count', len(events.json().get('events', [])))

next_event = client.get('/events/next')
print('next_status', next_event.status_code)
if next_event.status_code == 200:
    print('next_payload', next_event.json())

items = events.json().get('events', [])
available = [e for e in items if e.get('available_for_prediction')]
chosen = available[-1] if available else (items[-1] if items else None)
if not chosen:
    print('predict_skipped', 'no events')
else:
    req = {'season': int(last_season), 'round': int(chosen['round'])}
    pred = client.post('/predict_race', json=req)
    print('predict_status', pred.status_code)
    if pred.status_code == 200:
        p = pred.json()
        print('predict_event', p.get('event_name'), p.get('event_date'))
        print('predict_drivers', len(p.get('drivers', [])))
        print('winner', p.get('most_likely_winner'))
    else:
        print('predict_error', pred.text)
