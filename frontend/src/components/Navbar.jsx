import React from 'react';
import { RotateCcw, Sun, Moon } from 'lucide-react';
import { useAccessibility } from '../context/AccessibilityContext';

export default function Navbar({ onReset }) {
  const { theme, toggleTheme } = useAccessibility();

  return (
    <header className="app-navbar bg-[#FCFCFD] border-b border-zinc-200 px-4 md:px-6 flex items-center justify-between shrink-0">
      <a href="/" className="font-semibold text-zinc-900 tracking-tight text-base no-underline">
        SignBridge
      </a>

      <div className="flex items-center gap-2 text-xs text-zinc-600">
        <a className="legal-nav-link" href="/privacy-policy">Privacy</a>
        <a className="legal-nav-link" href="/terms">Terms</a>
        <button
          type="button"
          onClick={toggleTheme}
          className="flex items-center gap-1.5 px-2.5 py-1 text-zinc-700 hover:text-zinc-900 hover:bg-zinc-100 border border-zinc-200 rounded transition-colors"
          title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
          aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
        >
          {theme === 'dark' ? (
            <Sun className="w-3.5 h-3.5" />
          ) : (
            <Moon className="w-3.5 h-3.5" />
          )}
          <span className="hidden xs:inline font-medium">
            {theme === 'dark' ? 'Light' : 'Dark'}
          </span>
        </button>

        {onReset && (
          <button
            type="button"
            onClick={onReset}
            className="flex items-center gap-1.5 px-2.5 py-1 text-zinc-700 hover:text-zinc-900 hover:bg-zinc-100 border border-zinc-200 rounded transition-colors"
            title="Reset application state"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span className="hidden xs:inline font-medium">Reset</span>
          </button>
        )}
      </div>
    </header>
  );
}
