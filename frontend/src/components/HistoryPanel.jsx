function HistoryPanel({ items }) {
  return (
    <section className="panel reveal">
      <h2>Recent Predictions</h2>
      {!items.length ? <p>No history yet.</p> : null}
      <div className="history-list">
        {items.map((entry) => (
          <article className="history-item" key={entry._id}>
            <p className="history-title">{entry.disease}</p>
            <p>Confidence: {(entry.confidence * 100).toFixed(2)}%</p>
            <p>Severity: {entry.severity?.level || "n/a"}</p>
            <p>{new Date(entry.createdAt).toLocaleString()}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

export default HistoryPanel;
