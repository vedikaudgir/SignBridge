import React, { useEffect } from 'react';
import { Camera, CircleStop, ScanLine, X, AlertCircle } from 'lucide-react';

function statusTone(streamStatus) {
  if (streamStatus === 'streaming') {
    return 'bg-emerald-500';
  }

  if (streamStatus === 'connecting') {
    return 'bg-amber-500';
  }

  if (streamStatus === 'error') {
    return 'bg-rose-500';
  }

  return 'bg-zinc-400';
}

export function CameraControlToolbar({
  videoRef,
  onStartStreaming,
  onStopStreaming,
  streamStatus,
  connectionMessage,
  websocketUrl,
}) {
  const isStreaming = streamStatus === 'streaming';
  const isConnecting = streamStatus === 'connecting';

  return (
    <section className="camera-control-toolbar" aria-label="Camera controls">
      <div className="camera-control-buttons">
        <button
          type="button"
          onClick={() => onStartStreaming(videoRef.current)}
          disabled={isConnecting || isStreaming}
          className="camera-start-button"
        >
          <Camera className="w-4 h-4" />
          <span>
            {isConnecting
              ? 'Connecting...'
              : isStreaming
                ? 'Streaming Live'
                : 'Start Camera'}
          </span>
        </button>

        <button
          type="button"
          onClick={onStopStreaming}
          disabled={!isStreaming && !isConnecting}
          className="camera-stop-button"
        >
          <CircleStop className="w-4 h-4" />
          <span>Stop Camera</span>
        </button>
      </div>

      <div
        className="backend-status"
        role="status"
        aria-live="polite"
        aria-label={`${connectionMessage}. Backend URL: ${websocketUrl}. Camera access requires HTTPS or localhost.`}
        title={`${connectionMessage} · ${websocketUrl} · Camera access requires HTTPS or localhost.`}
      >
        <span className={`backend-status-dot ${statusTone(streamStatus)}`} />
        <span className="backend-status-label">{connectionMessage}</span>
        <span className="backend-status-url">{websocketUrl}</span>
      </div>
    </section>
  );
}

export default function InputSection({
  videoRef,
  currentInput,
  onClearInput,
  streamStatus,
  connectionMessage,
}) {
  useEffect(() => {
    const videoElement = videoRef.current;
    return () => {
      if (videoElement?.srcObject) {
        const tracks = videoElement.srcObject.getTracks();
        tracks.forEach((track) => track.stop());
        videoElement.srcObject = null;
      }
    };
  }, [videoRef]);

  return (
    <section className="camera-panel live-panel">
      <div className="live-panel-heading">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-600">
          Live Sign Input
        </h2>
        {currentInput && (
          <button
            type="button"
            onClick={onClearInput}
            className="text-xs text-zinc-500 hover:text-zinc-800 flex items-center gap-1 font-medium transition-colors"
          >
            <X className="w-3.5 h-3.5" />
            Clear Stream
          </button>
        )}
      </div>

      <div className="camera-preview relative rounded-md border border-zinc-300 bg-[#FCFCFD] overflow-hidden">
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          aria-label="Live camera preview"
          aria-hidden={!currentInput}
          className={`h-full w-full object-cover ${
            currentInput ? 'opacity-100' : 'opacity-0'
          }`}
        />

        {!currentInput && streamStatus === 'error' && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-center p-6 camera-placeholder-bg">
            <div className="w-12 h-12 rounded-full bg-rose-100 border border-rose-200 flex items-center justify-center mb-4">
              <AlertCircle className="w-6 h-6 text-rose-600" />
            </div>
            <p className="text-sm font-semibold text-rose-700 mb-1">
              {connectionMessage.toLowerCase().includes('permission')
                ? 'Camera permission needs attention'
                : 'Unable to start the live stream'}
            </p>
            <p className="text-xs text-zinc-600 max-w-sm leading-relaxed mb-3">
              {connectionMessage}
            </p>
            <p className="text-xs text-zinc-600 max-w-sm leading-relaxed">
              Check this site’s camera permission and confirm the configured
              backend is reachable, then try starting the camera again.
            </p>
          </div>
        )}

        {!currentInput && streamStatus !== 'error' && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-center p-6 camera-placeholder-bg">
            <div className="w-12 h-12 rounded-full bg-zinc-100 border border-zinc-200 flex items-center justify-center mb-4">
              <ScanLine className="w-6 h-6 text-zinc-700" />
            </div>
            <p className="text-sm font-semibold text-zinc-900 mb-1">
              Start the webcam stream
            </p>
            <p className="text-xs text-zinc-500 max-w-sm leading-relaxed">
              Camera frames are sent over WebSocket to the configured backend
              for landmark extraction and sign prediction.
            </p>
          </div>
        )}

        {currentInput && (
          <div role="status" className="absolute left-4 top-4 flex items-center gap-2 rounded bg-black/75 px-3 py-1.5 text-[11px] font-medium text-white">
            <span className={`h-2.5 w-2.5 rounded-full ${statusTone(streamStatus)}`} />
            <span>{connectionMessage}</span>
          </div>
        )}
      </div>

    </section>
  );
}
