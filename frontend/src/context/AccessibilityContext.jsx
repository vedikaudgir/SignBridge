import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';

const AccessibilityContext = createContext(null);

const SUPPORTED_LANGUAGES = [
  { code: 'en', name: 'English' },
  { code: 'es', name: 'Spanish' },
  { code: 'fr', name: 'French' },
  { code: 'de', name: 'German' },
  { code: 'hi', name: 'Hindi' },
  { code: 'te', name: 'Telugu' },
  { code: 'ja', name: 'Japanese' },
  { code: 'ko', name: 'Korean' },
  { code: 'zh', name: 'Chinese' },
  { code: 'ar', name: 'Arabic' },
  { code: 'pt', name: 'Portuguese' },
  { code: 'ru', name: 'Russian' },
];

export function AccessibilityProvider({ children }) {
  const [theme, setTheme] = useState(() => {
    const saved = localStorage.getItem('signbridge-theme');
    if (saved) return saved;
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  });

  const [language, setLanguage] = useState(() => {
    return localStorage.getItem('signbridge-lang') || 'en';
  });

  const [isSpeaking, setIsSpeaking] = useState(false);
  const [speechRate, setSpeechRate] = useState(0.9);
  const [fontSize, setFontSize] = useState(() => {
    return parseInt(localStorage.getItem('signbridge-fontsize') || '100', 10);
  });

  // Apply theme to DOM
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('signbridge-theme', theme);
  }, [theme]);

  // Persist language
  useEffect(() => {
    localStorage.setItem('signbridge-lang', language);
  }, [language]);

  // Persist font size
  useEffect(() => {
    document.documentElement.style.fontSize = `${fontSize}%`;
    localStorage.setItem('signbridge-fontsize', String(fontSize));
  }, [fontSize]);

  const toggleTheme = useCallback(() => {
    setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'));
  }, []);

  // Translation function using MyMemory free API
  const translateText = useCallback(async (text, targetLang) => {
    if (!text || !text.trim() || targetLang === 'en') {
      return text;
    }

    try {
      const res = await fetch(
        `https://api.mymemory.translated.net/get?q=${encodeURIComponent(text)}&langpair=en|${targetLang}`
      );
      const data = await res.json();
      if (data.responseStatus === 200 && data.responseData?.translatedText) {
        return data.responseData.translatedText;
      }
      return text;
    } catch {
      console.warn('Translation failed, returning original text');
      return text;
    }
  }, []);

  // Enhanced speak function with language support
  const speakText = useCallback((text, lang = 'en') => {
    if (!text || !text.trim()) return;
    if (!('speechSynthesis' in window)) return;

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = speechRate;
    utterance.pitch = 1.0;
    utterance.lang = lang;

    // Try to find a voice for the chosen language
    const voices = window.speechSynthesis.getVoices();
    const match = voices.find((v) => v.lang.startsWith(lang));
    if (match) utterance.voice = match;

    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);

    window.speechSynthesis.speak(utterance);
  }, [speechRate]);

  const stopSpeaking = useCallback(() => {
    window.speechSynthesis.cancel();
    setIsSpeaking(false);
  }, []);

  return (
    <AccessibilityContext.Provider
      value={{
        theme,
        setTheme,
        toggleTheme,
        language,
        setLanguage,
        translateText,
        speakText,
        stopSpeaking,
        isSpeaking,
        speechRate,
        setSpeechRate,
        fontSize,
        setFontSize,
        supportedLanguages: SUPPORTED_LANGUAGES,
      }}
    >
      {children}
    </AccessibilityContext.Provider>
  );
}

export function useAccessibility() {
  const ctx = useContext(AccessibilityContext);
  if (!ctx) throw new Error('useAccessibility must be used within AccessibilityProvider');
  return ctx;
}
