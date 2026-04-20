type TasteTuningPlaceholderProps = {
  onBack: () => void;
};

export default function TasteTuningPlaceholder({ onBack }: TasteTuningPlaceholderProps) {
  return (
    <div className="app">
      <header className="header">
        <button type="button" className="link-back" onClick={onBack}>
          ← Back to profile
        </button>
        <h1>Tune your taste profile</h1>
        <p className="tagline">This section is not built yet.</p>
      </header>

      <div className="card message">
        <p>
          Here you’ll answer a few quick questions so we can build or refine your hidden preference vector. Check back after
          the next release.
        </p>
      </div>
    </div>
  );
}
