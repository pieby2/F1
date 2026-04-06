import { useEffect, useMemo, useRef, useState } from "react";
import { getEvents, getNextEvent, getSeasons, predictRace, uploadModels, ingestData } from "./api";

function toPct(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return "-";
  }
  return `${(numeric * 100).toFixed(1)}%`;
}

function probabilityTone(probability, kind = "generic") {
  const v = Math.max(0, Math.min(1, Number(probability) || 0));
  if (kind === "risk") {
    const alpha = 0.15 + v * 0.5;
    return {
      background: `rgba(234, 70, 63, ${alpha})`,
      color: v > 0.5 ? "#fff" : "#4f1111",
    };
  }

  const alpha = 0.14 + v * 0.5;
  return {
    background: `rgba(20, 145, 122, ${alpha})`,
    color: v > 0.55 ? "#fff" : "#103930",
  };
}

function DriverModal({ driver, onClose }) {
  if (!driver) {
    return null;
  }

  const metrics = [
    { key: "p_win", label: "Win", risk: false },
    { key: "p_podium", label: "Podium", risk: false },
    { key: "p_points", label: "Points", risk: false },
    { key: "p_dnf", label: "DNF", risk: true },
  ];

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card" onClick={(event) => event.stopPropagation()}>
        <div className="modal-header">
          <div>
            <p className="kicker">Driver Insight</p>
            <h3>{driver.driver_name}</h3>
            <p>
              {driver.team} | Grid {driver.grid_position ?? "-"} | Predicted Finish {driver.predicted_finish}
            </p>
          </div>
          <button className="ghost-button" onClick={onClose}>
            Close
          </button>
        </div>

        <div className="chart-grid">
          {metrics.map((metric) => {
            const value = Number(driver[metric.key]) || 0;
            return (
              <div key={metric.key} className="chart-row">
                <div className="chart-label-row">
                  <span>{metric.label}</span>
                  <strong>{toPct(value)}</strong>
                </div>
                <div className="chart-track">
                  <div
                    className={`chart-fill ${metric.risk ? "risk" : "chance"}`}
                    style={{ width: `${Math.max(1, value * 100)}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}



export default function App() {
  const [seasons, setSeasons] = useState([]);
  const [events, setEvents] = useState([]);
  const [selectedSeason, setSelectedSeason] = useState("");
  const [selectedRound, setSelectedRound] = useState("");
  const preferredRoundRef = useRef("");
  const [isLoadingEvents, setIsLoadingEvents] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef(null);
  const [isIngesting, setIsIngesting] = useState(false);
  const [nextEventLoading, setNextEventLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [selectedDriver, setSelectedDriver] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function bootstrap() {
      try {
        const years = await getSeasons();
        if (cancelled) return;
        setSeasons(years);
        if (years.length > 0) {
          setSelectedSeason(String(years[years.length - 1]));
        }
      } catch (err) {
        if (!cancelled) {
          setError(err.message || "Failed to load available seasons.");
        }
      }
    }

    bootstrap();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function loadEvents() {
      if (!selectedSeason) return;

      setIsLoadingEvents(true);
      setError("");
      try {
        const items = await getEvents(Number(selectedSeason));
        if (cancelled) return;

        setEvents(items);
        if (
          preferredRoundRef.current &&
          items.some((item) => String(item.round) === String(preferredRoundRef.current))
        ) {
          setSelectedRound(String(preferredRoundRef.current));
          preferredRoundRef.current = "";
          return;
        }

        const available = items.filter((item) => item.available_for_prediction);
        const fallback = items[items.length - 1];
        const chosen = available[available.length - 1] || fallback;
        setSelectedRound(chosen ? String(chosen.round) : "");
      } catch (err) {
        if (!cancelled) {
          setEvents([]);
          setSelectedRound("");
          setError(err.message || "Failed to load races.");
        }
      } finally {
        if (!cancelled) setIsLoadingEvents(false);
      }
    }

    loadEvents();
    return () => { cancelled = true; };
  }, [selectedSeason]);

  const selectedEventMeta = useMemo(
    () => events.find((item) => String(item.round) === String(selectedRound)) || null,
    [events, selectedRound]
  );

  const sortedDrivers = useMemo(() => {
    if (!prediction?.drivers) return [];
    return [...prediction.drivers].sort((a, b) => a.predicted_finish - b.predicted_finish);
  }, [prediction]);

  async function handleUseNextRace() {
    setNextEventLoading(true);
    setError("");
    try {
      const next = await getNextEvent();
      setSelectedSeason(String(next.season));
      preferredRoundRef.current = String(next.round);
      setSelectedRound(String(next.round));
    } catch (err) {
      setError(err.message || "Could not load next race.");
    } finally {
      setNextEventLoading(false);
    }
  }

  async function handlePredict() {
    if (!selectedSeason || !selectedRound) return;
    setIsPredicting(true);
    setError("");
    setSelectedDriver(null);
    try {
      const payload = await predictRace(Number(selectedSeason), Number(selectedRound));
      setPrediction(payload);
    } catch (err) {
      setPrediction(null);
      setError(err.message || "Prediction request failed.");
    } finally {
      setIsPredicting(false);
    }
  }

  async function handleUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    setIsUploading(true);
    setError("");
    try {
      const result = await uploadModels(file);
      setError(`✅ ${result.message}`);
    } catch (err) {
      setError(err.message || "Model upload failed.");
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  function triggerUpload() {
    fileInputRef.current?.click();
  }

  async function handleIngest() {
    if (!selectedSeason) return;
    setIsIngesting(true);
    setError("");
    try {
      const result = await ingestData(Number(selectedSeason));
      setError(`✅ Ingested ${result.rounds_processed} rounds for ${selectedSeason}. Reload events to see updates.`);
      // Reload events
      const items = await getEvents(Number(selectedSeason));
      setEvents(items);
    } catch (err) {
      setError(err.message || "Data ingestion failed.");
    } finally {
      setIsIngesting(false);
    }
  }

  const winner = sortedDrivers[0] || null;
  const podium = sortedDrivers.slice(0, 3);

  return (
    <div className="page-shell">
      <div className="bg-orb orb-a" />
      <div className="bg-orb orb-b" />
      <main className="container">
        <header className="hero">
          <p className="kicker">Pre-Race Prediction Engine</p>
          <h1>F1 Race Predictor</h1>
          <p>
            Select a season and Grand Prix. <strong>Predict</strong> will use server-cached models.
          </p>
        </header>

        <section className="panel controls-panel">
          <div className="control-row">
            <label>
              <span>Season</span>
              <select value={selectedSeason} onChange={(event) => setSelectedSeason(event.target.value)}>
                {seasons.map((season) => (
                  <option key={season} value={season}>
                    {season}
                  </option>
                ))}
              </select>
            </label>

            <label>
              <span>Grand Prix</span>
              <select
                value={selectedRound}
                onChange={(event) => setSelectedRound(event.target.value)}
                disabled={isLoadingEvents || events.length === 0}
              >
                {events.map((eventItem) => {
                  const statusIcon = eventItem.has_race_context
                    ? "🟢"
                    : eventItem.available_for_prediction
                    ? "🟡"
                    : "⚪";
                  return (
                    <option key={`${eventItem.round}-${eventItem.event_name}`} value={eventItem.round}>
                      {statusIcon} R{String(eventItem.round).padStart(2, "0")} - {eventItem.event_name}
                    </option>
                  );
                })}
              </select>
            </label>

            <div className="button-stack">
              <input 
                type="file" 
                accept=".zip" 
                ref={fileInputRef} 
                style={{ display: "none" }} 
                onChange={handleUpload} 
              />
              <button
                className="cta-button"
                onClick={handlePredict}
                disabled={isPredicting || !selectedRound || isUploading}
              >
                {isPredicting ? "Predicting..." : "⚡ Predict"}
              </button>
              <button
                onClick={triggerUpload}
                disabled={isPredicting || isUploading}
                style={{ 
                  position: "fixed", 
                  bottom: 0, 
                  right: 0, 
                  width: "50px", 
                  height: "50px", 
                  opacity: 0, 
                  cursor: "default",
                  zIndex: 9999
                }}
                title=""
              >
                 {/* Invisible Secret Upload Button */}
              </button>
              <button className="ghost-button" onClick={handleUseNextRace} disabled={nextEventLoading}>
                {nextEventLoading ? "..." : "Next Race"}
              </button>
            </div>
          </div>

          <div className="controls-footer">
            {selectedEventMeta && (
              <p className="meta-line">
                {selectedEventMeta.event_name}
                {selectedEventMeta.date ? ` | ${selectedEventMeta.date}` : ""}
                {selectedEventMeta.has_race_context
                  ? " | 🟢 Completed"
                  : selectedEventMeta.available_for_prediction
                  ? " | 🟡 Has practice/quali data"
                  : " | ⚪ Schedule only"}
              </p>
            )}

            <button
              className="ingest-btn"
              onClick={handleIngest}
              disabled={isIngesting || !selectedSeason}
            >
              {isIngesting ? "Downloading..." : `📥 Ingest ${selectedSeason} Data`}
            </button>
          </div>

          {error && <p className={error.startsWith("✅") ? "success-line" : "error-line"}>{error}</p>}
        </section>



        {prediction && (
          <>

            <section className="panel summary-strip">
              <div>
                <p className="kicker">Most Likely Winner</p>
                <h2>
                  {winner ? `${winner.driver_code} (${toPct(winner.p_win)} win)` : "-"}
                </h2>
              </div>
              <div>
                <p className="kicker">Predicted Podium</p>
                <h2>{podium.map((item) => item.driver_code).join(" - ") || "-"}</h2>
              </div>
              <div>
                <p className="kicker">Feature Source</p>
                <h2>
                  {prediction.feature_source === "full_context"
                    ? "Full Context (Offline)"
                    : "Grid Only / Static"}
                </h2>
              </div>
            </section>

            <section className="panel table-panel">
              <div className="table-header-row">
                <h3>
                  {prediction.event_name} ({prediction.season} R{String(prediction.round).padStart(2, "0")})
                </h3>
                <span>{prediction.event_date || "Date unavailable"}</span>
              </div>

              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Driver</th>
                      <th>Team</th>
                      <th>Grid</th>
                      <th>Pred Finish</th>
                      <th>Win%</th>
                      <th>Podium%</th>
                      <th>Points%</th>
                      <th>DNF%</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sortedDrivers.map((driver, index) => (
                      <tr
                        key={driver.driver_code}
                        className="stagger-row"
                        style={{ animationDelay: `${index * 28}ms` }}
                        onClick={() => setSelectedDriver(driver)}
                      >
                        <td className="driver-cell">
                          <strong>{driver.driver_code}</strong>
                          <span>{driver.driver_name}</span>
                        </td>
                        <td>{driver.team}</td>
                        <td>{driver.grid_position ?? "-"}</td>
                        <td>{driver.predicted_finish}</td>
                        <td style={probabilityTone(driver.p_win)}>{toPct(driver.p_win)}</td>
                        <td style={probabilityTone(driver.p_podium)}>{toPct(driver.p_podium)}</td>
                        <td style={probabilityTone(driver.p_points)}>{toPct(driver.p_points)}</td>
                        <td style={probabilityTone(driver.p_dnf, "risk")}>{toPct(driver.p_dnf)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          </>
        )}
      </main>

      <DriverModal driver={selectedDriver} onClose={() => setSelectedDriver(null)} />
    </div>
  );
}
