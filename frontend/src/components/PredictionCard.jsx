function PredictionCard({ result }) {
  if (!result) {
    return null;
  }

  const confidence = (result.confidence * 100).toFixed(2);
  const severity = result.severity || {};
  const severityLevel = severity.level || "unknown";

  return (
    <section className="panel reveal">
      <h2>Prediction Result</h2>

      <div className="result-grid">
        <div>
          <p className="meta-label">Disease</p>
          <p className="meta-value">{result.disease}</p>
        </div>
        <div>
          <p className="meta-label">Confidence</p>
          <p className="meta-value">{confidence}%</p>
        </div>
        <div>
          <p className="meta-label">Severity</p>
          <p className={`severity-badge ${severityLevel}`}>{severityLevel.toUpperCase()}</p>
        </div>
      </div>

      <p className="treatment-text">{result.treatment}</p>
      <p className="severity-note">{severity.note}</p>

      {result.weather ? (
        <div className="weather-box">
          <h3>Field Weather Context</h3>
          {result.weather.warning ? (
            <p>{result.weather.warning}</p>
          ) : (
            <>
              <p>Temperature: {result.weather.temperature_c} C</p>
              <p>Humidity: {result.weather.humidity}%</p>
              <p>Rainfall: {result.weather.rain_mm} mm</p>
              <p>Wind: {result.weather.wind_speed} km/h</p>
            </>
          )}
          <p className="weather-note">{result.weather_note}</p>
        </div>
      ) : null}

      {Array.isArray(result.top_predictions) && result.top_predictions.length ? (
        <div className="top-preds">
          <h3>Top Predictions</h3>
          {result.top_predictions.map((item) => (
            <p key={item.label}>
              {item.label}: {(item.confidence * 100).toFixed(2)}%
            </p>
          ))}
        </div>
      ) : null}
    </section>
  );
}

export default PredictionCard;
