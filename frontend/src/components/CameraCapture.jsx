import { useEffect, useRef, useState } from "react";

function CameraCapture({ onCapture }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  const [cameraOn, setCameraOn] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  const startCamera = async () => {
    try {
      setError("");
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setCameraOn(true);
    } catch (err) {
      setError("Camera access failed. Please allow camera permissions.");
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    setCameraOn(false);
  };

  const captureFrame = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) {
      return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);

    canvas.toBlob(
      (blob) => {
        if (!blob) {
          return;
        }
        const file = new File([blob], "camera-capture.jpg", { type: "image/jpeg" });
        onCapture(file);
      },
      "image/jpeg",
      0.95
    );
  };

  return (
    <div className="camera-card">
      <h3>Live Camera Detection</h3>
      <p>Capture a leaf photo directly from your camera.</p>

      <div className="camera-actions">
        {!cameraOn ? (
          <button type="button" onClick={startCamera} className="ghost-btn">
            Start Camera
          </button>
        ) : (
          <>
            <button type="button" onClick={captureFrame} className="primary-btn">
              Capture Frame
            </button>
            <button type="button" onClick={stopCamera} className="ghost-btn">
              Stop Camera
            </button>
          </>
        )}
      </div>

      {error ? <p className="error-text">{error}</p> : null}

      <video ref={videoRef} autoPlay playsInline className={`camera-view ${cameraOn ? "show" : "hide"}`} />
      <canvas ref={canvasRef} className="camera-canvas" />
    </div>
  );
}

export default CameraCapture;
