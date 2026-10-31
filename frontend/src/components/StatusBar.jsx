import React from 'react';

export default function StatusBar({
  modelStatus = 'idle',
}) {
  return (
    <footer className="app-footer">
      <span>
        Connection: <strong aria-live="polite">{modelStatus}</strong>
      </span>
      <nav aria-label="Legal information">
        <a href="/privacy-policy">Privacy Policy</a>
        <a href="/terms">Terms and Conditions</a>
      </nav>
    </footer>
  );
}
