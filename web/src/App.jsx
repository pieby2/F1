import { useEffect, useMemo, useRef, useState } from "react";
import { getEvents, getNextEvent, getSeasons, predictRace, uploadModels, ingestData } from "./api";

const TEAM_IDS = {
  "Red Bull": "red-bull",
  "Ferrari": "ferrari",
  "Mercedes": "mercedes",
  "McLaren": "mclaren",
  "Aston Martin": "aston-martin",
  "Alpine": "alpine",
  "Williams": "williams",
  "RB": "rb",
  "Visa Cash App RB": "rb",
  "AlphaTauri": "rb",
  "Kick Sauber": "sauber",
  "Alfa Romeo": "sauber",
  "Haas": "haas"
};

function getTeamClass(name) {
  if (!name) return "team-default";
  for (const [key, val] of Object.entries(TEAM_IDS)) {
    if (name.includes(key)) return `team-${val}`;
  }
  return "team-default";
}

function toPct(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "-";
  return `${(Math.min(1, Math.max(0, numeric)) * 100).toFixed(1)}%`;
}

function AnimatedBar({ value, kind = "chance", delay = 0 }) {
  const numericValue = Math.min(100, Math.max(1, value * 100));
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const t = setTimeout(() => setWidth(numericValue), 100 + delay);
    return () => clearTimeout(t);
  }, [numericValue, delay]);

  return (
    <div className="telemetry-bar-wrap">
      <div 
        className={`telemetry-bar ${kind}`} 
        style={{ width: `${width}%` }} 
      />
    </div>
  );
}

function DriverModal({ driver, onClose }) {
  if (!driver) return null;

  const metrics = [
    { key: "p_win", label: "Win Probability" },
    { key: "p_podium", label: "Podium Finish" },
    { key: "p_points", label: "Points Finish" },
  ];

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className={`modal-card f1-borders ${getTeamClass(driver.team)}`} onClick={(e) => e.stopPropagation()}>
        <div className="modal-top">
          <div className="modal-titles">
            <h3 className="f1-font">{driver.driver_name}</h3>
            <p className="subtitle">{driver.team}</p>
          </div>
          <button className="close-btn" onClick={onClose}>&times;</button>
        </div>
        
        <div className="driver-modal-stats">
          <div className="stat-box">
            <span>GRID</span>
            <strong>{driver.grid_position ?? "-"}</strong>
          </div>
          <div className="stat-box">
            <span>PRED FINISH</span>
            <strong>{driver.predicted_finish}</strong>
          </div>
          <div className="stat-box risk-box">
            <span>DNF RISK</span>
            <strong>{toPct(driver.p_dnf)}</strong>
          </div>
        </div>

        <div className="telemetry-section">
          <h4 className="f1-font telemetry-title">
            {driver.isActualResult ? "RACE RESULT DETAILS" : "TELEMETRY PREDICTIONS"}
          </h4>
          {driver.isActualResult ? (
            <div className="telemetry-row">
                 <div className="telemetry-labels">
                   <span>CLASSIFICATION</span>
                   <span className="telemetry-val">{driver.predicted_finish}</span>
                 </div>
            </div>
          ) : (
            metrics.map((m, i) => {
              const v = Number(driver[m.key]) || 0;
              return (
                <div key={m.key} className="telemetry-row">
                  <div className="telemetry-labels">
                    <span>{m.label}</span>
                    <span className="telemetry-val">{toPct(v)}</span>
                  </div>
                  <AnimatedBar value={v} delay={i * 100} />
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}

function DriverCard({ driver, index, onClick, isActualResult }) {
  return (
    <div 
      className={`driver-card f1-borders ${getTeamClass(driver.team)}`} 
      style={{ animationDelay: `${index * 50}ms` }}
      onClick={() => onClick({...driver, isActualResult})}
    >
      <div className="card-rank f1-font">{driver.predicted_finish || (index + 1)}</div>
      <div className="card-info">
        <h3 className="f1-font">{driver.driver_code}</h3>
        <span className="card-name">{driver.driver_name}</span>
        <span className="card-team">{driver.team}</span>
      </div>
      <div className="card-stats">
        <div className="mini-stat">
          <span>Grid</span>
          <strong>{driver.grid_position ?? "-"}</strong>
        </div>
        {isActualResult ? (
          <div className="mini-stat highlight">
            <span>Result</span>
            <strong>{driver.predicted_finish}</strong>
          </div>
        ) : (
          <div className="mini-stat highlight">
            <span>Win%</span>
            <strong>{toPct(driver.p_win)}</strong>
          </div>
        )}
      </div>
      <div className="card-accent" />
    </div>
  );
}

function TeamCard({ team, index, isActualResult }) {
  return (
    <div 
      className={`team-card f1-borders ${getTeamClass(team.team)}`}
      style={{ animationDelay: `${index * 50}ms` }}
    >
      <div className="card-rank f1-font">{index + 1}</div>
      <div className="card-info">
        <h3 className="f1-font">{team.team}</h3>
        <span className="card-name">Drivers: {team.drivers.map(d => d.driver_code).join(" & ")}</span>
      </div>
      <div className="card-stats team-stats">
        {isActualResult ? (
          <div className="mini-stat highlight">
            <span>Best Finish</span>
            <strong>{Math.min(...team.drivers.map(d => (d.predicted_finish === "DNF" || d.predicted_finish === 99) ? 99 : d.predicted_finish))}</strong>
          </div>
        ) : (
          <>
            <div className="mini-stat highlight">
              <span>Constructors Win%</span>
              <strong>{toPct(team.p_win)}</strong>
            </div>
            <div className="mini-stat">
              <span>Exp. Podium</span>
              <strong>{toPct(team.p_podium)}</strong>
            </div>
          </>
        )}
      </div>
      <div className="card-accent" />
      {!isActualResult && (
        <div className="team-bars">
           <div className="telemetry-row">
             <AnimatedBar value={team.p_win} delay={index*50 + 100} />
           </div>
        </div>
      )}
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
  
  const [activeTab, setActiveTab] = useState("drivers"); // 'drivers' | 'teams'

  useEffect(() => {
    let cancelled = false;
    async function bootstrap() {
      try {
        const years = await getSeasons();
        if (cancelled) return;
        setSeasons(years);
        if (years.length > 0) setSelectedSeason(String(years[years.length - 1]));
      } catch (err) {
        if (!cancelled) setError(err.message || "Failed to load seasons.");
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
        if (preferredRoundRef.current && items.some(i => String(i.round) === String(preferredRoundRef.current))) {
          setSelectedRound(String(preferredRoundRef.current));
          preferredRoundRef.current = "";
          return;
        }
        const available = items.filter(i => i.available_for_prediction);
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

  const selectedEventMeta = useMemo(() => events.find(i => String(i.round) === String(selectedRound)) || null, [events, selectedRound]);

  const sortedDrivers = useMemo(() => {
    if (!prediction?.drivers) return [];
    return [...prediction.drivers].sort((a, b) => a.predicted_finish - b.predicted_finish);
  }, [prediction]);

  const teamStats = useMemo(() => {
    if (!prediction?.drivers) return [];
    const teams = {};
    prediction.drivers.forEach(d => {
      if (!teams[d.team]) {
        teams[d.team] = {
           team: d.team,
           p_win: 0,
           p_podium: 0,
           drivers: []
        };
      }
      teams[d.team].p_win += d.p_win;
      teams[d.team].p_podium += d.p_podium; 
      teams[d.team].drivers.push(d);
    });
    return Object.values(teams).sort((a,b) => b.p_win - a.p_win);
  }, [prediction]);

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
      setError(err.message || "Prediction failed.");
    } finally {
      setIsPredicting(false);
    }
  }

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
  
  async function handleUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setIsUploading(true);
    try {
      const res = await uploadModels(file);
      setError(`✅ ${res.message}`);
    } catch(err) {
      setError(err.message || "Upload failed.");
    } finally {
      setIsUploading(false);
      if(fileInputRef.current) fileInputRef.current.value = "";
    }
  }
  
  async function handleIngest() {
    if (!selectedSeason) return;
    setIsIngesting(true);
    setError("");
    try {
      const res = await ingestData(Number(selectedSeason));
      setError(`✅ Ingested ${res.rounds_processed} rounds for ${selectedSeason}.`);
      const items = await getEvents(Number(selectedSeason));
      setEvents(items);
    } catch (err) {
      setError(err.message || "Ingestion failed.");
    } finally {
      setIsIngesting(false);
    }
  }

  return (
    <div className="f1-shell">
      <nav className="f1-navbar">
        <div className="nav-logo f1-font"><span>FASTF1</span> PRED</div>
      </nav>
      
      <main className="f1-container">
        
        <header className="f1-hero">
          <h1 className="f1-font f1-title">RACE PREDICTOR</h1>
          <div className="f1-hero-line"></div>
        </header>

        <section className="f1-controls-panel f1-borders">
          <div className="f1-controls-grid">
            <div className="field">
              <label className="f1-font">SEASON</label>
              <select className="f1-select" value={selectedSeason} onChange={(e) => setSelectedSeason(e.target.value)}>
                {seasons.map(s => <option key={s} value={s}>{s}</option>)}
              </select>
            </div>
            <div className="field">
              <label className="f1-font">GRAND PRIX</label>
              <select className="f1-select" value={selectedRound} onChange={(e) => setSelectedRound(e.target.value)} disabled={isLoadingEvents}>
                {events.map((ev) => {
                  const s = ev.has_race_context ? "🟢" : ev.available_for_prediction ? "🟡" : "⚪";
                  return (
                    <option key={ev.round} value={ev.round}>
                      {s} R{String(ev.round).padStart(2, "0")} - {ev.event_name}
                    </option>
                  );
                })}
              </select>
            </div>
            <div className="field-actions">
              <input type="file" accept=".zip" ref={fileInputRef} style={{display:"none"}} onChange={handleUpload}/>
              <button className="f1-btn primary f1-font" onClick={handlePredict} disabled={isPredicting || !selectedRound || isUploading}>
                {isPredicting ? "CALCULATING..." : selectedEventMeta?.has_race_context ? "VIEW RESULTS >>" : "PREDICT OUTCOME >>"}
              </button>
              <button className="f1-btn secondary f1-font" onClick={handleUseNextRace} disabled={nextEventLoading}>
                NEXT RACE 
              </button>
            </div>
          </div>
          <div className="f1-controls-footer">
             {selectedEventMeta && (
              <span className="event-meta">
                {selectedEventMeta.event_name} | {selectedEventMeta.date} {selectedEventMeta.has_race_context ? "| 🟢 COMPLETED" : ""}
              </span>
             )}
             <button className="f1-btn ghost f1-font" onClick={handleIngest} disabled={isIngesting || !selectedSeason}>
               {isIngesting ? "..." : `📥 INGEST ${selectedSeason}`}
             </button>
          </div>
          {error && <div className={error.startsWith("✅") ? "msg success" : "msg error"}>{error}</div>}
        </section>

        {prediction && (
          <section className="f1-results-section fade-in">
            <div className="results-header">
              <h2 className="f1-font">
                {prediction.season} {prediction.event_name} <span style={{color:"var(--f1-red)"}}>-</span> RESULTS
              </h2>
            </div>
            
            <div className="f1-tabs">
              <button 
                className={`f1-tab f1-font ${activeTab === 'drivers' ? 'active' : ''}`}
                onClick={() => setActiveTab('drivers')}
              >
                STARTING GRID / DRIVERS
              </button>
              <button 
                className={`f1-tab f1-font ${activeTab === 'teams' ? 'active' : ''}`}
                onClick={() => setActiveTab('teams')}
              >
                CONSTRUCTORS
              </button>
            </div>

            <div className="f1-grid-view">
              {activeTab === 'drivers' && (
                <div className="f1-cards-grid">
                  {sortedDrivers.map((driver, idx) => (
                    <DriverCard 
                      key={driver.driver_code} 
                      driver={driver} 
                      index={idx} 
                      onClick={setSelectedDriver} 
                      isActualResult={prediction.feature_source === "actual_race_results"}
                    />
                  ))}
                </div>
              )}

              {activeTab === 'teams' && (
                <div className="f1-cards-grid">
                  {teamStats.map((team, idx) => (
                    <TeamCard 
                      key={team.team} 
                      team={team} 
                      index={idx} 
                      isActualResult={prediction.feature_source === "actual_race_results"}
                    />
                  ))}
                </div>
              )}
            </div>
          </section>
        )}
      </main>

      <DriverModal driver={selectedDriver} onClose={() => setSelectedDriver(null)} />
    </div>
  );
}
