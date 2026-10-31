import React, { useEffect } from 'react';
import { ArrowLeft } from 'lucide-react';
import Navbar from './Navbar';

const privacySections = [
  {
    title: 'Camera frames and backend processing',
    paragraphs: [
      'Camera access starts only after you choose Start Camera and grant permission in your browser. The browser captures JPEG frames and sends them over a WebSocket to the configured SignBridge backend. The default endpoint is ws://127.0.0.1:8000/ws/sign-stream; a deployment can use a different endpoint.',
      'The backend decodes each received frame, extracts pose and hand landmarks, and runs its prediction path. In the repository implementation, frames are processed in memory and are not written to persistent storage. A rolling landmark window and prediction-smoothing state are held in memory for the active connection and cleared when it disconnects.',
      'The backend logs generated session identifiers and prediction labels/confidence values. The inspected application code does not log frame images. Hosting providers, proxies, or operators may have separate logging, monitoring, backup, or retention practices that this project cannot verify. Ask the operator of the backend you connect to about those practices.',
      'The connection endpoint is configurable. The default local ws:// endpoint is not encrypted; use an appropriately secured deployment for sensitive environments. Stopping the camera closes the WebSocket and stops the browser media tracks.',
    ],
  },
  {
    title: 'Text, translation, and browser preferences',
    paragraphs: [
      'The current session’s accumulated words remain in browser application memory and are cleared when you reset or leave the page. This implementation does not save those words to an application account or backend storage.',
      'If you select a language other than English, accumulated text is sent to the MyMemory translation API by the current frontend integration. That service processes the submitted text under its own terms and privacy practices. Do not send sensitive text unless you are comfortable with this transfer. Translation results may be incomplete or incorrect.',
      'The selected theme, translation language, and text-size preference are saved in this browser’s localStorage. Read-aloud uses the browser’s speech synthesis API; available voices and any platform-specific processing depend on your browser and operating system.',
    ],
  },
  {
    title: 'Your controls',
    paragraphs: [
      'You can deny camera access, stop the camera stream, clear the current input, reset the session text, or change browser permissions using your browser settings. The application does not request microphone access.',
    ],
  },
];

const termsSections = [
  {
    title: 'Prototype purpose',
    paragraphs: [
      'SignBridge is a software prototype for exploring live sign-language recognition and text output. It is provided for evaluation and experimentation, not as a certified interpreting, accessibility, medical, emergency, or safety service.',
      'Recognition depends on camera framing, lighting, signing variation, the configured model or heuristic, and backend configuration. The system may miss signs or return incorrect labels. Model confidence is a score produced by the prediction pipeline; it is not verified accuracy or a guarantee that a label is correct.',
      'The current interface adds backend-returned predictions to text only when you choose to add them. Review the output before relying on, sharing, or translating it. Do not use it as a substitute for a qualified interpreter or another communication method when accuracy is important.',
    ],
  },
  {
    title: 'Camera and external services',
    paragraphs: [
      'Starting the camera sends frames to the configured backend for processing. Optional translation sends accumulated text to the MyMemory API. See the Privacy Policy for details about the current data flow. Only use the prototype with data you are permitted to process and share with those services.',
      'You are responsible for obtaining any permissions required to use the camera and for ensuring that people captured by it understand the use of the prototype.',
    ],
  },
  {
    title: 'Availability and changes',
    paragraphs: [
      'The prototype and its prediction results are provided without a promise of continuous availability, a particular level of performance, or suitability for a specific purpose. Backend availability and data handling may depend on the operator and deployment. These terms describe the project’s intended use and are not a statement of legal guarantees.',
    ],
  },
];

export default function LegalPage({ page }) {
  const isPrivacy = page === 'privacy';
  const sections = isPrivacy ? privacySections : termsSections;
  const title = isPrivacy ? 'Privacy Policy' : 'Terms and Conditions';

  useEffect(() => {
    const previousTitle = document.title;
    document.title = `${title} — SignBridge`;
    return () => {
      document.title = previousTitle;
    };
  }, [title]);

  return (
    <div className="min-h-screen flex flex-col bg-[#F9F9FA] text-zinc-900">
      <Navbar />
      <main className="legal-page">
        <a className="back-link" href="/">
          <ArrowLeft aria-hidden="true" size={16} />
          Back to live recognition
        </a>
        <p className="eyebrow">SIGNBRIDGE · PROTOTYPE</p>
        <h1>{title}</h1>
        <p className="legal-lead">
          {isPrivacy
            ? 'This page describes the data flows implemented in this repository. A deployed instance may have additional operator or hosting practices.'
            : 'These terms describe the prototype’s intended use and important limitations.'}
        </p>
        {sections.map((section) => (
          <section className="legal-section" key={section.title}>
            <h2>{section.title}</h2>
            {section.paragraphs.map((paragraph) => (
              <p key={paragraph}>{paragraph}</p>
            ))}
          </section>
        ))}
      </main>
      <footer className="app-footer legal-footer">
        <a href="/privacy-policy">Privacy Policy</a>
        <a href="/terms">Terms and Conditions</a>
      </footer>
    </div>
  );
}
