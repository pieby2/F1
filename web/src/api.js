const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api";

async function fetchJson(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  if (!response.ok) {
    let message = `Request failed with status ${response.status}`;
    try {
      const payload = await response.json();
      if (payload?.detail) {
        message = String(payload.detail);
      }
    } catch (_err) {
      // Keep fallback message.
    }
    throw new Error(message);
  }

  return response.json();
}

export async function getSeasons() {
  const payload = await fetchJson("/seasons");
  return Array.isArray(payload?.seasons) ? payload.seasons : [];
}

export async function getEvents(season) {
  const payload = await fetchJson(`/events/${season}`);
  return Array.isArray(payload?.events) ? payload.events : [];
}

export async function getNextEvent() {
  return fetchJson("/events/next");
}

export async function getNewsSummary(count = 6) {
  return fetchJson(`/news/summary?count=${count}`);
}

export async function getHistorySummary(season = null) {
  const query = season === null || season === undefined ? "" : `?season=${season}`;
  return fetchJson(`/history/summary${query}`);
}

export async function predictRace(season, round) {
  return fetchJson("/predict_race", {
    method: "POST",
    body: JSON.stringify({ season, round }),
  });
}

export async function uploadModels(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/upload_models`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    let message = `Request failed with status ${response.status}`;
    try {
      const payload = await response.json();
      if (payload?.detail) message = String(payload.detail);
    } catch (_err) {}
    throw new Error(message);
  }

  return response.json();
}

export async function ingestData(season, startRound = 1, endRound = null) {
  return fetchJson("/ingest_data", {
    method: "POST",
    body: JSON.stringify({ season, start_round: startRound, end_round: endRound }),
  });
}
