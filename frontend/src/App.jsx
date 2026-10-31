import React, { useEffect, useRef, useState } from 'react';
import Navbar from './components/Navbar';
import InputSection, { CameraControlToolbar } from './components/InputSection';
import RecognitionSection from './components/RecognitionSection';
import TranslationSection from './components/TranslationSection';
import StatusBar from './components/StatusBar';
import AccessibilityPanel from './components/AccessibilityPanel';
import LegalPage from './components/LegalPage';
import { SignStreamClient } from './services/signStreamClient';
import { useAccessibility } from './context/AccessibilityContext';

const WS_URL =
  import.meta.env.VITE_SIGNBRIDGE_WS_URL ?? 'ws://127.0.0.1:8000/ws/sign-stream';

export default function App() {
  const streamClientRef = useRef(null);
  const cameraVideoRef = useRef(null);
  const [currentInput, setCurrentInput] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [latestConfidence, setLatestConfidence] = useState(0);
  const [sentenceWords, setSentenceWords] = useState([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [toastMessage, setToastMessage] = useState(null);
  const [streamStatus, setStreamStatus] = useState('idle');
  const [connectionMessage, setConnectionMessage] = useState('Camera not started');
  const [frameCount, setFrameCount] = useState(0);
  const [pathname, setPathname] = useState(window.location.pathname);

  const { speakText, language, translateText } = useAccessibility();

  const showToast = (message) => {
    setToastMessage(message);
    window.clearTimeout(showToast.timeoutId);
    showToast.timeoutId = window.setTimeout(() => setToastMessage(null), 3000);
  };

  useEffect(() => {
    const handlePopState = () => setPathname(window.location.pathname);
    window.addEventListener('popstate', handlePopState);
    return () => {
      window.removeEventListener('popstate', handlePopState);
      if (streamClientRef.current) {
        streamClientRef.current.stop();
        streamClientRef.current = null;
      }
    };
  }, []);

  if (pathname === '/privacy-policy' || pathname === '/terms') {
    return <LegalPage page={pathname === '/privacy-policy' ? 'privacy' : 'terms'} />;
  }

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
    setLatestConfidence(0);
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
        setLatestConfidence(payload.confidence ?? 0);

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
        setLatestConfidence(0);
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
      setLatestConfidence(0);

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
    setLatestConfidence(0);
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
    setLatestConfidence(0);
    setSentenceWords([]);
    setConnectionMessage('Camera not started');
    showToast('Reset application workspace');
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#F9F9FA] text-zinc-900 selection:bg-zinc-200">
      <Navbar onReset={handleResetAll} />

      <main className="app-main">
        <section className="app-intro" aria-labelledby="page-title">
          <p className="eyebrow">SIGNBRIDGE · LIVE CAMERA MODE</p>
          <h1 id="page-title">Read a live sign prediction</h1>
          <p>
            Send live camera frames to SignBridge and review predictions before
            adding them to your text.
          </p>
        </section>
        <CameraControlToolbar
          videoRef={cameraVideoRef}
          onStartStreaming={startStreaming}
          onStopStreaming={() => stopStreaming()}
          streamStatus={streamStatus}
          connectionMessage={connectionMessage}
          websocketUrl={WS_URL}
        />
        {toastMessage && (
          <div role="status" className="toast-message">
            {toastMessage}
          </div>
        )}

        <div className="recognition-layout" id="live-recognition">
          <InputSection
            videoRef={cameraVideoRef}
            currentInput={currentInput}
            onClearInput={handleClearInput}
            streamStatus={streamStatus}
            connectionMessage={connectionMessage}
          />

          <RecognitionSection
            prediction={prediction}
            latestConfidence={latestConfidence}
            onAddToSentence={handleAddToSentence}
            isAnalyzing={isAnalyzing}
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
      />

      <AccessibilityPanel />
    </div>
  );
}
