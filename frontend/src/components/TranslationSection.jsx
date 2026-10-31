import React, { useState, useEffect } from 'react';
import { Volume2, VolumeX, RotateCcw, Trash2, Copy, Check, Languages } from 'lucide-react';
import { useAccessibility } from '../context/AccessibilityContext';

export default function TranslationSection({
  sentenceWords,
  onClear,
  onUndo,
  onSpeak
}) {
  const [copied, setCopied] = useState(false);
  const [translatedText, setTranslatedText] = useState('');
  const [isTranslating, setIsTranslating] = useState(false);

  const {
    language,
    translateText,
    speakText,
    stopSpeaking,
    isSpeaking,
    supportedLanguages,
  } = useAccessibility();

  const fullSentence = sentenceWords.join(' ');
  const currentLang = supportedLanguages.find((l) => l.code === language);

  // Auto-translate when sentence or language changes
  useEffect(() => {
    let cancelled = false;

    async function doTranslate() {
      if (!fullSentence.trim() || language === 'en') {
        setTranslatedText('');
        return;
      }

      setIsTranslating(true);
      try {
        const result = await translateText(fullSentence, language);
        if (!cancelled) {
          setTranslatedText(result);
        }
      } catch {
        if (!cancelled) setTranslatedText('');
      } finally {
        if (!cancelled) setIsTranslating(false);
      }
    }

    doTranslate();
    return () => { cancelled = true; };
  }, [fullSentence, language, translateText]);

  const handleCopy = () => {
    if (!fullSentence) return;
    const textToCopy = translatedText || fullSentence;
    navigator.clipboard.writeText(textToCopy);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleReadAloud = () => {
    if (isSpeaking) {
      stopSpeaking();
      return;
    }

    const textToSpeak = translatedText || fullSentence;
    if (!textToSpeak.trim()) return;

    speakText(textToSpeak, language);
  };

  return (
    <div className="flex flex-col gap-3 rounded-md border border-zinc-300 bg-[#FCFCFD] p-5">
      <div className="flex items-center justify-between">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-600">
          Translated Text
        </h2>
        <div className="flex items-center gap-2">
          {language !== 'en' && (
              <span className="translation-badge">
              <Languages className="w-3 h-3" />
              {currentLang?.name}
            </span>
          )}
          <span className="text-xs text-zinc-500 font-mono">
            {sentenceWords.length} {sentenceWords.length === 1 ? 'word' : 'words'}
          </span>
        </div>
      </div>

      {/* Original Output Container */}
      <div className="min-h-[72px] p-4 bg-zinc-50 border border-zinc-200 rounded flex items-center justify-between gap-4">
        {sentenceWords.length > 0 ? (
          <div className="flex flex-wrap items-center gap-2">
            {sentenceWords.map((word, index) => (
              <span
                key={index}
                className="px-2.5 py-1 bg-zinc-200/80 border border-zinc-300 rounded text-xs font-mono font-semibold text-zinc-900 tracking-wide"
              >
                {word}
              </span>
            ))}
          </div>
        ) : (
          <p className="text-xs text-zinc-400 italic">
            Accumulated sentence words will appear here...
          </p>
        )}
      </div>

      {/* Translated Output (shown when language is not English) */}
      {language !== 'en' && sentenceWords.length > 0 && (
        <div className="min-h-[48px] p-4 border border-zinc-200 rounded relative overflow-hidden"
          style={{
            background: 'var(--surface-muted)',
          }}
        >
          <div className="flex items-start justify-between gap-3">
            <div className="flex-1">
              <div className="flex items-center gap-1.5 mb-1.5">
                <Languages className="w-3.5 h-3.5 text-zinc-500" style={{ color: 'var(--accent-solid)' }} />
                <span className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: 'var(--accent-solid)' }}>
                  {currentLang?.name} Translation
                </span>
              </div>
              {isTranslating ? (
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 border-2 rounded-full animate-spin" style={{ borderColor: 'var(--border)', borderTopColor: 'var(--accent)' }} />
                  <span className="text-xs text-zinc-500">Translating...</span>
                </div>
              ) : (
                <p className="text-sm font-medium text-zinc-800 leading-relaxed">
                  {translatedText}
                </p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Sentence Action Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-zinc-200">
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={handleReadAloud}
            disabled={sentenceWords.length === 0}
            className="read-aloud-btn"
            title={isSpeaking ? 'Stop reading aloud' : 'Read sentence aloud via Text-to-Speech'}
          >
            {isSpeaking ? (
              <>
                <VolumeX className="read-aloud-btn-icon" />
                <span>Stop</span>
              </>
            ) : (
              <>
                <Volume2 className="read-aloud-btn-icon" />
                <span>Read Aloud</span>
              </>
            )}
          </button>

          {/* Speak Original (only when translated) */}
          {language !== 'en' && translatedText && (
            <button
              type="button"
              onClick={() => onSpeak(fullSentence)}
              disabled={sentenceWords.length === 0}
              className="px-4 py-2 bg-zinc-900 hover:bg-zinc-800 text-zinc-100 text-xs font-medium rounded flex items-center gap-2 transition-colors cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed shadow-none"
              title="Speak original English text"
            >
              <Volume2 className="w-4 h-4" />
              <span>Speak English</span>
            </button>
          )}

          {/* Copy Button */}
          <button
            type="button"
            onClick={handleCopy}
            disabled={sentenceWords.length === 0}
            className="px-3 py-2 bg-zinc-100 hover:bg-zinc-200/80 text-zinc-700 border border-zinc-300 rounded text-xs font-medium flex items-center gap-1.5 transition-colors disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
          >
            {copied ? (
              <>
                <Check className="w-3.5 h-3.5 text-zinc-800" />
                <span>Copied</span>
              </>
            ) : (
              <>
                <Copy className="w-3.5 h-3.5" />
                <span>Copy</span>
              </>
            )}
          </button>
        </div>

        {/* Editing Controls */}
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={onUndo}
            disabled={sentenceWords.length === 0}
            className="px-3 py-2 bg-zinc-100 hover:bg-zinc-200/80 text-zinc-700 border border-zinc-300 rounded text-xs font-medium flex items-center gap-1.5 transition-colors disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
            title="Remove last word"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Undo</span>
          </button>

          <button
            type="button"
            onClick={onClear}
            disabled={sentenceWords.length === 0}
            className="px-3 py-2 bg-zinc-100 hover:bg-zinc-200/80 text-zinc-700 border border-zinc-300 rounded text-xs font-medium flex items-center gap-1.5 transition-colors disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
            title="Clear all accumulated text"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Clear</span>
          </button>
        </div>
      </div>
    </div>
  );
}
