from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

h = client.get('/health')
print('health', h.status_code)
print(h.json())

s = client.get('/seasons')
print('seasons', s.status_code)
payload = s.json()
print('seasons_payload', payload)
if not payload.get('seasons'):
    raise SystemExit('No seasons available')

season = int(payload['seasons'][-1])
e = client.get(f'/events/{season}')
print('events', e.status_code)
ev_payload = e.json()
print('events_count', len(ev_payload.get('events', [])))

events = ev_payload.get('events', [])
if not events:
    raise SystemExit('No events found')

available = [x for x in events if x.get('available_for_prediction')]
chosen = available[-1] if available else events[-1]
req = {'season': season, 'round': int(chosen['round'])}
print('predict_req', req)

p = client.post('/predict_race', json=req)
print('predict', p.status_code)
if p.status_code != 200:
    print(p.text)
    raise SystemExit(1)

pred = p.json()
print('event', pred.get('event_name'), pred.get('event_date'))
print('feature_source', pred.get('feature_source'))
print('driver_rows', len(pred.get('drivers', [])))
print('winner', pred.get('most_likely_winner'))
print('podium', pred.get('predicted_podium'))
