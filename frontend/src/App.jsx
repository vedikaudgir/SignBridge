import React, { useEffect, useRef, useState } from 'react';
import Navbar from './components/Navbar';
import InputSection from './components/InputSection';
import RecognitionSection from './components/RecognitionSection';
import TranslationSection from './components/TranslationSection';
import StatusBar from './components/StatusBar';
import AccessibilityPanel from './components/AccessibilityPanel';
import { SignStreamClient } from './services/signStreamClient';
import { useAccessibility } from './context/AccessibilityContext';

const WS_URL =
  import.meta.env.VITE_SIGNBRIDGE_WS_URL ?? 'ws://127.0.0.1:8000/ws/sign-stream';

export default function App() {
  const streamClientRef = useRef(null);
  const [currentInput, setCurrentInput] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [sentenceWords, setSentenceWords] = useState([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [toastMessage, setToastMessage] = useState(null);
  const [streamStatus, setStreamStatus] = useState('idle');
  const [connectionMessage, setConnectionMessage] = useState('Camera not started');
  const [frameCount, setFrameCount] = useState(0);

  const { theme, speakText, language, translateText } = useAccessibility();

  const showToast = (message) => {
    setToastMessage(message);
    window.clearTimeout(showToast.timeoutId);
    showToast.timeoutId = window.setTimeout(() => setToastMessage(null), 3000);
  };

  useEffect(() => {
    return () => {
      if (streamClientRef.current) {
        streamClientRef.current.stop();
        streamClientRef.current = null;
      }
    };
  }, []);

  const stopStreaming = async ({ silent = false } = {}) => {
    if (streamClientRef.current) {
      await streamClientRef.current.stop();
      streamClientRef.current = null;
    }

    setIsAnalyzing(false);
    setStreamStatus('idle');
    setConnectionMessage('Camera stopped');
    setFrameCount(0);

    if (!silent) {
      showToast('Webcam stream stopped');
    }
  };

  const startStreaming = async (videoElement) => {
    if (!videoElement) {
      showToast('Camera preview is not ready yet');
      return;
    }

    if (streamClientRef.current) {
      await stopStreaming({ silent: true });
    }

    setPrediction(null);
    setCurrentInput({
      id: `camera_${Date.now()}`,
      name: 'Live webcam stream',
      type: 'camera',
    });
    setStreamStatus('connecting');
    setConnectionMessage('Requesting camera and backend connection');
    setIsAnalyzing(true);
    setFrameCount(0);

    const client = new SignStreamClient({
      onStatusChange: (nextStatus, message) => {
        setStreamStatus(nextStatus);
        setConnectionMessage(message);
        setIsAnalyzing(
          nextStatus === 'connecting' || nextStatus === 'streaming'
        );
      },
      onPrediction: (payload) => {
        setFrameCount((count) => count + 1);

        if (payload.prediction) {
          setPrediction({
            label: String(payload.prediction).toUpperCase(),
            confidence: payload.confidence ?? 0,
            timestamp: new Date().toISOString(),
          });
        } else {
          setPrediction(null);
        }
      },
      onError: (error) => {
        console.error('Streaming error:', error);
        setPrediction(null);
        showToast(error.message || 'Streaming failed');
      },
    });

    streamClientRef.current = client;

    try {
      await client.start(videoElement);
      showToast('Webcam stream connected to backend');
    } catch (error) {
      console.error('Unable to start stream:', error);
      setIsAnalyzing(false);
      setStreamStatus('error');
      setCurrentInput(null);
      streamClientRef.current = null;

      let friendlyMsg = error.message || 'Unable to start stream';
      if (error.name === 'NotAllowedError' || error.message?.toLowerCase().includes('permission denied')) {
        friendlyMsg = 'Camera permission blocked. Click the lock/camera icon in your browser address bar and set Camera to "Allow", then try again.';
      }
      setConnectionMessage(friendlyMsg);
      showToast(friendlyMsg);
    }
  };

  const handleClearInput = async () => {
    await stopStreaming({ silent: true });
    setCurrentInput(null);
    setPrediction(null);
    setConnectionMessage('Camera not started');
  };

  const handleAddToSentence = () => {
    if (!prediction?.label) {
      return;
    }

    setSentenceWords((previous) => [...previous, prediction.label]);
    showToast(`Added "${prediction.label}" to sentence`);
  };

  const handleClearSentence = () => {
    setSentenceWords([]);
  };

  const handleUndoSentence = () => {
    setSentenceWords((previous) => previous.slice(0, -1));
  };

  const handleSpeakSentence = async (text) => {
    if (!text || text.trim() === '') {
      return;
    }

    if (!('speechSynthesis' in window)) {
      showToast('Speech synthesis is not supported in this browser');
      return;
    }

    // If language is not English, translate first then speak
    if (language !== 'en') {
      const translated = await translateText(text, language);
      speakText(translated, language);
      showToast(`Speaking translated text in ${language}`);
    } else {
      speakText(text, 'en');
      showToast(`Speaking "${text}"`);
    }
  };

  const handleResetAll = async () => {
    await stopStreaming({ silent: true });
    setCurrentInput(null);
    setPrediction(null);
    setSentenceWords([]);
    setConnectionMessage('Camera not started');
    showToast('Reset application workspace');
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#F9F9FA] text-zinc-900 selection:bg-zinc-200">
      <Navbar onReset={handleResetAll} />

      <main className="flex-1 max-w-7xl w-full mx-auto p-4 md:p-6 flex flex-col gap-6">
        {toastMessage && (
          <div className="fixed bottom-20 right-24 z-[9998] px-4 py-2.5 bg-zinc-900 text-zinc-100 text-xs font-medium rounded-lg shadow-lg border border-zinc-700 transition-all animate-[panel-enter_0.3s_ease]">
            {toastMessage}
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-stretch">
          <InputSection
            currentInput={currentInput}
            onClearInput={handleClearInput}
            onStartStreaming={startStreaming}
            onStopStreaming={() => stopStreaming()}
            isAnalyzing={isAnalyzing}
            streamStatus={streamStatus}
            connectionMessage={connectionMessage}
            websocketUrl={WS_URL}
          />

          <RecognitionSection
            prediction={prediction}
            onAddToSentence={handleAddToSentence}
            isAnalyzing={isAnalyzing}
            streamStatus={streamStatus}
            frameCount={frameCount}
            connectionMessage={connectionMessage}
          />
        </div>

        <TranslationSection
          sentenceWords={sentenceWords}
          onClear={handleClearSentence}
          onUndo={handleUndoSentence}
          onSpeak={handleSpeakSentence}
        />
      </main>

      <StatusBar
        modelStatus={streamStatus}
        totalClasses="Predict stub active"
        transportLabel="WebSocket"
        endpointLabel={WS_URL}
      />

      <AccessibilityPanel />
    </div>
  );
}
