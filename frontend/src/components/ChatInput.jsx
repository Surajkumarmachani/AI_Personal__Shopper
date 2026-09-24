import { useState, useRef } from 'react'
import VoiceInput from './VoiceInput'

export default function ChatInput({ onSend, disabled, apiBase, webcamActive, onMicTap }) {
  const [text, setText] = useState('')
  // Voice-tone emotion captured when the mic transcript arrives, held until the
  // user actually presses send — so it still rides along even though the text
  // now waits in the box for review/editing instead of auto-sending.
  const pendingVoiceEmotion = useRef(null)

  function submit() {
    if (!text.trim() || disabled) return
    onSend(text, pendingVoiceEmotion.current)
    pendingVoiceEmotion.current = null
    setText('')
  }

  // Voice transcript arrived: drop it in the box for review instead of sending
  // immediately, but keep its emotion ready for when send is pressed.
  function handleVoiceTranscript(transcribedText, voiceEmotion) {
    setText(transcribedText)
    pendingVoiceEmotion.current = voiceEmotion
  }

  function onKeyDown(e) {
    // Enter sends, Shift+Enter makes a new line
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  return (
    <div className="chat-input">
      <input
        className="input-field"
        type="text"
        value={text}
        placeholder="Ask Aura for anything…"
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKeyDown}
        disabled={disabled}
        aria-label="Type your message"
        autoFocus
      />

      {/* Tap to record → transcribes → fills the box for review (not auto-send).
          When the webcam is off, the transcription response also carries a
          voice-tone emotion, held until the user presses send. */}
      <VoiceInput
        apiBase={apiBase}
        disabled={disabled}
        webcamActive={webcamActive}
        onTranscript={handleVoiceTranscript}
        onMicTap={onMicTap}
      />

      <button
        className="send-btn"
        onClick={submit}
        disabled={disabled || !text.trim()}
        aria-label="Send message"
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
          <path
            d="M3 12L21 3L13 21L11 13L3 12Z"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinejoin="round"
          />
        </svg>
      </button>
    </div>
  )
}
