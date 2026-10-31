import React from 'react';
import { Activity, Info, Plus, Waves } from 'lucide-react';

const LOW_CONFIDENCE_WARNING_THRESHOLD = 0.2;

function confidencePercent(prediction) {
  if (!prediction) {
    return '0.0';
  }

  return (prediction.confidence * 100).toFixed(1);
}

export default function RecognitionSection({
  prediction,
  onAddToSentence,
  isAnalyzing,
  latestConfidence,
  frameCount,
  connectionMessage,
}) {
  const percent = confidencePercent(prediction);
  const displayedConfidence = prediction?.confidence ?? latestConfidence;
  const displayedPercent = (displayedConfidence * 100).toFixed(1);
  const isLowConfidence =
    displayedConfidence < LOW_CONFIDENCE_WARNING_THRESHOLD;

  return (
    <section className="recognition-panel live-panel">
      <div className="live-panel-heading">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-600">
          Recognition
        </h2>
        <span className="text-[11px] font-mono text-zinc-500 bg-zinc-100 border border-zinc-200 px-2 py-0.5 rounded">
          Frames processed: {frameCount}
        </span>
      </div>

      <div className="recognition-card flex-1 rounded-md border border-zinc-300 bg-[#FCFCFD] p-5 flex flex-col justify-between">
        {isAnalyzing && !prediction ? (
          <div className="flex-1 flex flex-col items-center justify-center gap-3 text-zinc-500 text-center">
            <div className="w-10 h-10 rounded-full border border-zinc-200 bg-zinc-100 flex items-center justify-center">
              <Waves className="w-5 h-5 text-zinc-700" />
            </div>
            <div className="space-y-1">
              <p className="text-sm font-medium text-zinc-800">
                Waiting for a stable prediction
              </p>
              <p className="text-xs font-mono">{connectionMessage}</p>
              {latestConfidence > 0 && (
                <p className="text-xs font-mono text-zinc-600">
                  Current model confidence: {displayedPercent}%
                </p>
              )}
              {latestConfidence > 0 && isLowConfidence && (
                <p className="text-xs font-medium text-amber-700">
                  Low confidence; waiting for a stable prediction.
                </p>
              )}
            </div>
          </div>
        ) : prediction ? (
          <div className="flex flex-col justify-between h-full gap-6">
            <div className="space-y-4">
              <div className="space-y-1">
                <span className="text-xs font-medium text-zinc-500 uppercase tracking-wide">
                  Smoothed Prediction
                </span>
                <div className="flex items-baseline justify-between gap-2 border-b border-zinc-200 pb-3">
                  <h2 className="text-3xl sm:text-4xl font-semibold tracking-tight text-zinc-900 font-mono">
                    {prediction.label}
                  </h2>
                  <div className="text-right">
                    <span className="text-xl font-bold font-mono text-zinc-800">
                      {percent}%
                    </span>
                    <span className="block text-[10px] text-zinc-500 uppercase tracking-wider">
                      Confidence
                    </span>
                  </div>
                </div>
              </div>

              <div className="space-y-1.5">
                <div className="flex justify-between text-xs text-zinc-600">
                  <span>Model confidence</span>
                  <span className="font-mono text-zinc-700">
                    {prediction.confidence.toFixed(3)}
                  </span>
                </div>
                <div className="w-full bg-zinc-200 h-2 rounded-full overflow-hidden">
                  <div
                    className="bg-zinc-800 h-full rounded-full"
                    style={{ width: `${percent}%` }}
                  />
                </div>
              </div>

              {isLowConfidence && (
                <p className="rounded border border-amber-200 bg-amber-50 p-3 text-xs font-medium text-amber-800">
                  Low confidence: treat this prediction cautiously.
                </p>
              )}

              <div className="rounded border border-zinc-200 bg-zinc-50 p-3 text-xs text-zinc-600">
                <div className="flex items-center gap-2 text-zinc-800 font-medium">
                  <Activity className="w-4 h-4" />
                  <span>Backend prediction stream</span>
                </div>
                <p className="mt-2 leading-relaxed">
                  Predictions only appear after the backend has a full landmark
                  window and the same label survives the smoothing gate.
                </p>
              </div>
            </div>

            <div className="pt-4 border-t border-zinc-200">
              <button
                type="button"
                onClick={onAddToSentence}
                className="w-full py-2.5 px-4 bg-zinc-900 hover:bg-zinc-800 text-zinc-100 font-medium text-xs rounded transition-colors flex items-center justify-center gap-2 cursor-pointer shadow-none"
              >
                <Plus className="w-4 h-4" />
                <span>Add to Sentence</span>
              </button>
            </div>
          </div>
        ) : (
          <div className="flex-1 flex flex-col items-center justify-center text-center p-6 gap-3">
            <div className="w-10 h-10 rounded-full bg-zinc-100 border border-zinc-200 flex items-center justify-center text-zinc-400">
              <Info className="w-5 h-5 text-zinc-500" />
            </div>
            <div className="space-y-1 max-w-xs">
              <h3 className="text-sm font-semibold text-zinc-800">
                No Prediction Yet
              </h3>
              <p className="text-xs text-zinc-500 leading-relaxed">
                Start the camera stream to send frames to the backend and wait
                for a stable prediction. The returned confidence describes the
                model output; it does not establish prediction accuracy.
              </p>
            </div>
          </div>
        )}
      </div>

    </section>
  );
}
