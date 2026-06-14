import { useEffect, useMemo, useRef, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const MODES = [
  {
    id: "upload",
    title: "Upload",
    description: "Choose a file and analyze it.",
  },
  {
    id: "webcam",
    title: "Webcam",
    description: "Capture a frame and analyze it.",
  },
];

function formatPercent(value) {
  return `${Number(value || 0).toFixed(1)}%`;
}

function formatTimestamp(value) {
  if (!value) {
    return "N/A";
  }

  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }

  return new Intl.DateTimeFormat("en-US", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(parsed);
}

function shortenFileName(name, maxLength = 24) {
  if (!name || name.length <= maxLength) {
    return name || "N/A";
  }

  const dotIndex = name.lastIndexOf(".");
  const extension = dotIndex > -1 ? name.slice(dotIndex) : "";
  const stem = dotIndex > -1 ? name.slice(0, dotIndex) : name;
  const room = maxLength - extension.length - 1;

  if (room <= 4) {
    return `${stem.slice(0, maxLength - 1)}…`;
  }

  const left = Math.ceil(room * 0.55);
  const right = Math.max(1, room - left);
  return `${stem.slice(0, left)}…${stem.slice(-right)}${extension}`;
}

function StatCard({ label, value, tone }) {
  return (
    <div className={`stat-card stat-${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ProgressRow({ label, value, color }) {
  return (
    <div className="progress-row">
      <div className="progress-row__meta">
        <span>{label}</span>
        <strong>{formatPercent(value)}</strong>
      </div>
      <div className="progress-track" aria-hidden="true">
        <div className="progress-fill" style={{ width: `${value}%`, background: color }} />
      </div>
    </div>
  );
}

function DownloadIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path
        d="M12 3v10m0 0 4-4m-4 4-4-4M5 17.5V19a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-1.5"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function ExportButton({ children, onClick, variant = "soft", disabled = false }) {
  return (
    <button
      className={`secondary-button ${variant === "ghost" ? "secondary-button--ghost" : "secondary-button--soft"}`}
      type="button"
      onClick={onClick}
      disabled={disabled}
    >
      <DownloadIcon />
      {children}
    </button>
  );
}

export default function App() {
  const [mode, setMode] = useState("upload");
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [cameraReady, setCameraReady] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const [saveError, setSaveError] = useState("");
  const [savedNotice, setSavedNotice] = useState("");
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const previewRef = useRef("");

  useEffect(() => {
    return () => {
      if (previewRef.current.startsWith("blob:")) {
        URL.revokeObjectURL(previewRef.current);
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  useEffect(() => {
    if (mode !== "webcam" && streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      setCameraReady(false);
    }
  }, [mode]);

  const reportCards = useMemo(() => {
    if (!result) {
      return [];
    }
    return [
      { label: "Regions", value: result.tomatoCount, tone: "ink" },
      { label: "Ripe", value: result.ripeCount, tone: "green" },
      { label: "Unripe", value: result.unripeCount, tone: "amber" },
      { label: "Defective", value: result.defectiveCount, tone: "red" },
    ];
  }, [result]);

  async function analyzeSelectedFile(selectedFile) {
    if (!selectedFile) {
      setError("Choose an image before testing.");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile, selectedFile.name || "capture.png");

    setLoading(true);
    setError("");
    setSaveError("");
    setSavedNotice("");
    setResult(null);

    try {
      const response = await fetch(`${API_BASE}/api/analyze`, {
        method: "POST",
        body: formData,
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload?.detail || "Analysis failed.");
      }
      setResult(payload);
    } catch (err) {
      setError(err.message || "Unable to analyze the image.");
    } finally {
      setLoading(false);
    }
  }

  function handleFileChange(event) {
    const nextFile = event.target.files?.[0] || null;
    setFile(nextFile);
    setResult(null);
    setError("");

    if (previewRef.current.startsWith("blob:")) {
      URL.revokeObjectURL(previewRef.current);
    }
    const nextPreview = nextFile ? URL.createObjectURL(nextFile) : "";
    previewRef.current = nextPreview;
    setPreview(nextPreview);
  }

  async function startCamera() {
    setCameraError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "environment",
        },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraReady(true);
      setMode("webcam");
    } catch (err) {
      setCameraError("Camera access is blocked or unavailable in this browser.");
    }
  }

  async function captureFrame() {
    if (!videoRef.current) {
      setCameraError("Camera is not ready yet.");
      return;
    }

    const video = videoRef.current;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
    if (!blob) {
      setCameraError("Could not capture a frame from the camera.");
      return;
    }

    const captured = new File([blob], "webcam-capture.png", { type: "image/png" });
    setFile(captured);
    if (previewRef.current.startsWith("blob:")) {
      URL.revokeObjectURL(previewRef.current);
    }
    const nextPreview = canvas.toDataURL("image/png");
    previewRef.current = nextPreview;
    setPreview(nextPreview);
    await analyzeSelectedFile(captured);
  }

  function resetAll() {
    if (previewRef.current.startsWith("blob:")) {
      URL.revokeObjectURL(previewRef.current);
    }
    previewRef.current = "";
    setFile(null);
    setPreview("");
    setResult(null);
    setError("");
    setCameraError("");
    setSaveError("");
    setSavedNotice("");
  }

  async function downloadReport(format) {
    if (!result?.reportId) {
      return;
    }

    try {
      setSaveError("");
      const response = await fetch(`${API_BASE}/api/reports/${result.reportId}/download?format=${format}`);
      if (!response.ok) {
        throw new Error("Could not download the report.");
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      const extension = format === "pdf" ? "pdf" : format === "docx" ? "docx" : "json";
      link.download = `tomato-report-${result.reportId}.${extension}`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      setSavedNotice(`Report downloaded as ${extension.toUpperCase()}.`);
    } catch (err) {
      setSaveError(err.message || "Unable to save the report.");
    }
  }

  const summaryBars = [
    { label: "Ripe", value: result?.ripePct || 0, color: "#2E7D32" },
    { label: "Unripe", value: result?.unripePct || 0, color: "#C47F00" },
    { label: "Defective", value: result?.defectPct || 0, color: "#C62828" },
  ];

  return (
    <div className="app-shell">
      <div className="backdrop backdrop--left" />
      <div className="backdrop backdrop--right" />
      <div className="grain" />

      <div className="app-topbar">
        <div className="brand-lockup">
          <span className="brand-mark">TM</span>
          <div>
            <strong>Tomato Grading Console</strong>
            <span>Inspect. Export.</span>
          </div>
        </div>
      </div>

      <header className="hero">
        <div className="hero-copywrap">
          <p className="eyebrow">Tomato inspection</p>
          <h1>Inspect tomatoes faster.</h1>
          <p className="hero-copy">
            Upload or capture. Review. Export.
          </p>
        </div>

        <div className="hero-note">
          <span>Steps</span>
          <strong>Upload</strong>
          <strong>Inspect</strong>
          <strong>Export</strong>
        </div>
      </header>

      <main className="layout">
        <section className="panel controls">
            <div className="panel__header">
              <h2>Input</h2>
              <button className="ghost-button" onClick={resetAll} type="button">
                Reset
            </button>
          </div>

          <div className="mode-switch">
            {MODES.map((item) => (
              <button
                key={item.id}
                type="button"
                className={`mode-card ${mode === item.id ? "mode-card--active" : ""}`}
                onClick={() => setMode(item.id)}
              >
                <strong>{item.title}</strong>
                <span>{item.description}</span>
              </button>
            ))}
          </div>

          {mode === "upload" ? (
            <div className="upload-box">
              <label className="file-drop">
                <input type="file" accept="image/*" onChange={handleFileChange} />
                <strong>Select image</strong>
                <span>JPG, PNG, WEBP, BMP</span>
              </label>

              {preview ? <img className="media-preview" src={preview} alt="Selected preview" /> : null}

              <button
                className="primary-button"
                type="button"
                disabled={!file || loading}
                onClick={() => analyzeSelectedFile(file)}
              >
                {loading ? "Analyzing..." : "Analyze"}
              </button>
            </div>
          ) : (
            <div className="camera-box">
              <div className="camera-stage">
                <video ref={videoRef} className="camera-feed" autoPlay playsInline muted />
                {!cameraReady ? <div className="camera-placeholder">Camera preview appears here</div> : null}
              </div>

              <div className="camera-actions">
                <button className="primary-button" type="button" onClick={startCamera}>
                  Open Camera
                </button>
                <button className="secondary-button" type="button" onClick={captureFrame} disabled={!cameraReady || loading}>
                  {loading ? "Analyzing..." : "Capture"}
                </button>
              </div>
            </div>
          )}

          {error ? <p className="feedback feedback--error">{error}</p> : null}
          {cameraError ? <p className="feedback feedback--error">{cameraError}</p> : null}
        </section>

        <section className="panel report">
              <div className="panel__header">
                <div>
                  <h2>Report</h2>
                  <p className="panel-kicker">Latest result.</p>
                </div>
            {result ? (
              <span className="status-pill status-pill--live">Ready</span>
            ) : (
              <span className="status-pill">Waiting</span>
            )}
          </div>

          {!result ? (
            <div className="empty-state">
              <strong>No report yet</strong>
              <p>Results will appear here.</p>
            </div>
          ) : (
            <>
              <div className="report-toolbar">
                <div className="report-meta-grid">
                  <div className="report-meta">
                    <span>ID</span>
                    <strong>{result.reportId}</strong>
                  </div>
                  <div className="report-meta">
                    <span>File</span>
                    <strong title={result.fileName}>{shortenFileName(result.fileName, 26)}</strong>
                  </div>
                  <div className="report-meta">
                    <span>Time</span>
                    <strong>{formatTimestamp(result.generatedAt)}</strong>
                  </div>
                </div>
                <div className="report-actions">
                  <ExportButton onClick={() => downloadReport("pdf")}>PDF</ExportButton>
                  <ExportButton onClick={() => downloadReport("docx")}>DOCX</ExportButton>
                </div>
              </div>

              {savedNotice ? <p className="feedback feedback--success">{savedNotice}</p> : null}
              {saveError ? <p className="feedback feedback--error">{saveError}</p> : null}

              <div className="stats-grid">
                {reportCards.map((card) => (
                  <StatCard key={card.label} {...card} />
                ))}
              </div>

              <div className="insight-banner insight-banner--accent">
                <span>Summary</span>
                <p>{result.summary}</p>
              </div>

              <div className="artifacts-grid">
                <figure className="image-card image-card--hero">
                  <figcaption>Original</figcaption>
                  <img src={result.image} alt="Uploaded original" />
                </figure>
                <figure className="image-card image-card--hero">
                  <figcaption>Annotated</figcaption>
                  <img src={result.annotatedImage} alt="Annotated analysis" />
                </figure>
              </div>

              {result.summaryChart ? (
                <figure className="image-card chart-card">
                  <figcaption>Chart</figcaption>
                  <img src={result.summaryChart} alt="Summary chart" />
                </figure>
              ) : null}

              <div className="composition-card">
                <div className="composition-card__header">
                  <h3>Quality</h3>
                  <span>{result.tomatoCount} regions</span>
                </div>
                {summaryBars.map((item) => (
                  <ProgressRow key={item.label} {...item} />
                ))}
              </div>

              <div className="table-card">
                <div className="table-card__header">
                  <h3>Regions</h3>
                  <span>Per region defect</span>
                </div>

                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>Status</th>
                        <th>Label</th>
                        <th>Confidence</th>
                        <th>Defect</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.assessments.map((item) => (
                        <tr key={`${item.number}-${item.status}`}>
                          <td>{item.number}</td>
                          <td>
                            <span className="tag" style={{ background: `${item.color}20`, color: item.color }}>
                              {item.status}
                            </span>
                          </td>
                          <td>{item.label}</td>
                          <td>{Number(item.confidence).toFixed(3)}</td>
                          <td>{formatPercent(item.defectPercent)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {result.detailImages?.length ? (
                <div className="details-grid">
                  <div className="table-card__header">
                    <h3>Region Details</h3>
                    <span>Zoomed crops for each tomato region</span>
                  </div>
                  <div className="detail-gallery">
                    {result.detailImages.map((item) => (
                      <figure className="detail-card" key={`${item.title}-${item.caption}`}>
                        <img src={item.image} alt={item.title} />
                        <figcaption>
                          <strong>{item.title}</strong>
                          <span>{item.caption}</span>
                        </figcaption>
                      </figure>
                    ))}
                  </div>
                </div>
              ) : null}
            </>
          )}
        </section>
      </main>
    </div>
  );
}
