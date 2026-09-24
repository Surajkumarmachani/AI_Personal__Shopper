import { useState, useRef, useEffect } from 'react'
import ChatWindow from './components/ChatWindow'
import ChatInput from './components/ChatInput'
import Webcam from './components/Webcam'
import { API_BASE, authHeaders } from './api'
import './App.css'

// A stable per-browser session id so the backend remembers the conversation.
function getSessionId() {
  let id = localStorage.getItem('aura_session_id')
  if (!id) {
    id = (window.crypto && crypto.randomUUID) ? crypto.randomUUID() : String(Date.now())
    localStorage.setItem('aura_session_id', id)
  }
  return id
}

export default function App() {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: "Hi, I'm Aura ✨ I am your personal online shopping assistant. Tell me what you're looking for!",
      products: [],
    },
  ])
  const [loading, setLoading] = useState(false)
  // Live emotion — neutral until webcam (tier 1) or voice tone (tier 2) updates it.
  const [emotion, setEmotion] = useState('neutral')
  // Whether the webcam is actively detecting — drives the voice-emotion fallback.
  const [webcamActive, setWebcamActive] = useState(false)
  // When ON, Aura speaks her replies aloud (gTTS). Off by default.
  const [ttsEnabled, setTtsEnabled] = useState(false)
  // True while Aura is actively speaking — drives the header Stop button.
  const [speaking, setSpeaking] = useState(false)
  const sessionId = useRef(getSessionId())
  const audioRef = useRef(null)
  const speakAbortRef = useRef(null)

  function stopSpeech() {
    if (speakAbortRef.current) {
      speakAbortRef.current.abort()
      speakAbortRef.current = null
    }

    const audio = audioRef.current
    if (audio) {
      // Detach first so the natural-end handler can't fire on a stopped clip.
      audio.onended = null
      audio.pause()
      if (audio.src) {
        URL.revokeObjectURL(audio.src)
      }
      audio.removeAttribute('src')
      audio.load()
      audioRef.current = null
    }
    setSpeaking(false)
  }

  useEffect(() => stopSpeech, [])

  // Play Aura's reply as speech via the /speak endpoint
  async function speakText(text) {
    stopSpeech()
    const controller = new AbortController()
    speakAbortRef.current = controller
    try {
      const res = await fetch(`${API_BASE}/speak`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...authHeaders() },
        body: JSON.stringify({ text }),
        signal: controller.signal,
      })
      if (!res.ok) return
      const blob = await res.blob()
      if (controller.signal.aborted) return
      const audio = new Audio(URL.createObjectURL(blob))
      audioRef.current = audio
      audio.onended = () => {
        if (audioRef.current === audio) {
          URL.revokeObjectURL(audio.src)
          audioRef.current = null
          setSpeaking(false)
        }
      }
      setSpeaking(true)
      // play() rejects if playback is interrupted by pause() — swallow it so
      // it doesn't surface as an unhandled promise rejection.
      audio.play().catch(() => {})
    } catch {
      // TTS is a nice-to-have — never let it break the conversation
    } finally {
      if (speakAbortRef.current === controller) {
        speakAbortRef.current = null
      }
    }
  }

  // emotionOverride: set for voice messages when the webcam was off —
  // the voice-tone emotion arrives WITH the transcript, so we use it directly
  // instead of reading possibly-stale state.
  async function sendMessage(text, emotionOverride = null) {
    const trimmed = text.trim()
    if (!trimmed || loading) return

    // A new query should interrupt any reply currently being spoken.
    stopSpeech()

    const effectiveEmotion = emotionOverride || emotion
    if (emotionOverride) setEmotion(emotionOverride)  // keep UI state in sync

    // Optimistically show the user's message
    setMessages((m) => [...m, { role: 'user', content: trimmed }])
    setLoading(true)

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...authHeaders() },
        body: JSON.stringify({
          session_id: sessionId.current,
          message: trimmed,
          emotion: effectiveEmotion,
        }),
      })
      if (!res.ok) throw new Error(`Server responded ${res.status}`)
      const data = await res.json()

      setMessages((m) => [
        ...m,
        {
          role: 'assistant',
          content: data.response,
          products: data.products || [],
        },
      ])

      if (ttsEnabled && data.response) speakText(data.response)
    } catch (err) {
      setMessages((m) => [
        ...m,
        {
          role: 'assistant',
          content:
            "I couldn't reach the store. Make sure the backend is running on port 8000, then try again.",
          products: [],
          error: true,
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <div className="ambient" aria-hidden="true" />

      <header className="header">
        <div className="brand">
          <span className="orb" />
          <h1 className="brand-name">Aura</h1>
        </div>

        <div className="header-controls">
          <button
            className="tts-toggle"
            onClick={() => setTtsEnabled((v) => !v)}
            aria-pressed={ttsEnabled}
            title="Toggle spoken replies"
          >
            {ttsEnabled ? '🔊' : '🔇'} Voice reply
          </button>
          {speaking && (
            <button
              className="tts-toggle"
              onClick={stopSpeech}
              title="Stop Aura from speaking"
              aria-label="Stop Aura from speaking"
            >
              ⏹ Stop
            </button>
          )}
          <Webcam
            apiBase={API_BASE}
            sessionId={sessionId.current}
            onEmotion={setEmotion}
            onActiveChange={setWebcamActive}
          />
        </div>
      </header>

      <ChatWindow messages={messages} loading={loading} />
      <ChatInput
        onSend={sendMessage}
        disabled={loading}
        apiBase={API_BASE}
        webcamActive={webcamActive}
        onMicTap={stopSpeech}
      />
    </div>
  )
}
