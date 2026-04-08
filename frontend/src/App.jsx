import { useEffect, useMemo, useState } from "react";
import axios from "axios";

import CameraCapture from "./components/CameraCapture";
import PredictionCard from "./components/PredictionCard";
import HistoryPanel from "./components/HistoryPanel";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:4000/api";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [coords, setCoords] = useState(null);

  useEffect(() => {
    loadHistory();
  }, []);

  useEffect(() => {
    if (!navigator.geolocation) {
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (position) => {
        setCoords({
          lat: position.coords.latitude,
          lon: position.coords.longitude,
        });
      },
      () => {
        setCoords(null);
      }
    );
  }, []);

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const locationLabel = useMemo(() => {
    if (!coords) {
      return "Location not shared";
    }
    return `${coords.lat.toFixed(3)}, ${coords.lon.toFixed(3)}`;
  }, [coords]);

  const loadHistory = async () => {
    try {
      const res = await axios.get(`${API_BASE}/history?limit=10`);
      setHistory(res.data);
    } catch (err) {
      // Keep UI usable even if history API is unavailable.
      setHistory([]);
    }
  };

  const setIncomingFile = (file) => {
    if (!file) {
      return;
    }
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
  };

  const handlePredict = async () => {
    if (!selectedFile) {
      setError("Select or capture an image first.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      if (coords) {
        formData.append("lat", String(coords.lat));
        formData.append("lon", String(coords.lon));
      }

      const res = await axios.post(`${API_BASE}/upload`, formData);
      setResult(res.data);
      await loadHistory();
    } catch (err) {
      const message = err.response?.data?.error || "Prediction failed. Check backend and AI API services.";
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <div className="bg-orb orb-a" />
      <div className="bg-orb orb-b" />

      <header className="hero reveal">
        <p className="eyebrow">AI-Based Crop Disease Prediction</p>
        <h1>LeafScan Intelligence Console</h1>
        <p className="subtitle">
          Upload a crop leaf image or use live camera capture for disease classification, treatment guidance,
          severity estimation, and weather-aware risk notes.
        </p>
      </header>

      <main className="content-grid">
        <section className="panel reveal">
          <h2>Upload Leaf Image</h2>
          <div className="uploader-box">
            <input
              type="file"
              accept="image/*"
              onChange={(e) => setIncomingFile(e.target.files?.[0])}
            />
            <button type="button" onClick={handlePredict} className="primary-btn" disabled={loading}>
              {loading ? "Analyzing..." : "Predict Disease"}
            </button>
          </div>

          <p className="location-pill">Weather location: {locationLabel}</p>

          {previewUrl ? <img src={previewUrl} alt="Leaf preview" className="preview-img" /> : null}

          {error ? <p className="error-text">{error}</p> : null}
        </section>

        <CameraCapture onCapture={setIncomingFile} />

        <PredictionCard result={result} />
        <HistoryPanel items={history} />
      </main>
    </div>
  );
}

export default App;
