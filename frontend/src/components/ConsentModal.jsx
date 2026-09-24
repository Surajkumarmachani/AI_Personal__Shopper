import { createPortal } from 'react-dom'
import './ConsentModal.css'

/**
 * Informed-consent gate shown BEFORE the browser's own permission prompt.
 * The browser popup only says "this site wants your camera" — this modal
 * explains what Aura actually DOES with it, which is the part that matters
 * for genuine informed consent (biometric inference is sensitive data).
 *
 * Props:
 *   open      — whether to show
 *   device    — "camera" | "mic" (picks the right copy)
 *   onAccept  — user explicitly agreed → caller proceeds to getUserMedia
 *   onDecline — user said no thanks → caller stays in text mode
 */
export default function ConsentModal({ open, device, onAccept, onDecline }) {
  if (!open) return null

  const isCamera = device === 'camera'

  // Portal to <body>: this component gets rendered inside the header / input
  // bar, whose `position: relative; z-index: 1` creates a stacking context.
  // Later siblings (.chat-window, .chat-input) share z-index 1 and paint above
  // that whole context — so without the portal, the transparent chat area sits
  // on top of the modal and silently swallows every click on its buttons.
  return createPortal(
    <div className="consent-overlay" role="dialog" aria-modal="true">
      <div className="consent-panel">
        <h2 className="consent-title">
          {isCamera ? '📷 Before turning on your camera' : '🎤 Before turning on your microphone'}
        </h2>

        {isCamera ? (
          <p className="consent-body">
            If you enable your camera, Aura analyzes your facial expression
            every second to adapt its tone — for example, gentler replies if
            you seem sad. Frames are processed by this app's own local backend
            and are <strong>not stored and not sent to any third party</strong>.
            You can turn the camera off at any time, and the chat works fully
            without it.
          </p>
        ) : (
          <p className="consent-body">
            If you enable your microphone, your recording is transcribed to
            text by this app's own local backend, and — when your camera is
            off — its tone (pitch, energy) is analyzed to estimate your mood.
            Audio is <strong>not stored and not sent to any third party</strong>.
            You can always type instead.
          </p>
        )}

        <div className="consent-actions">
          <button className="consent-accept" onClick={onAccept}>
            {isCamera ? 'Enable camera' : 'Enable microphone'}
          </button>
          <button className="consent-decline" onClick={onDecline}>
            No thanks, {isCamera ? 'text only' : "I'll type"}
          </button>
        </div>
      </div>
    </div>,
    document.body
  )
}
