import { useRef, useState } from 'react'
import { authHeaders } from '../api'
import ConsentModal from './ConsentModal'
import './VoiceInput.css'

const CONSENT_KEY = 'aura_consent_mic'

// Pick a recording format the browser actually supports
function pickMime() {
  const types = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/mp4',
    'audio/ogg;codecs=opus',
  ]
  if (typeof MediaRecorder === 'undefined') return ''
  for (const t of types) {
    if (MediaRecorder.isTypeSupported(t)) return t
  }
  return ''
}

export default function VoiceInput({ apiBase, disabled, onTranscript, webcamActive, onMicTap }) {
  const [recording, setRecording] = useState(false)
  const [busy, setBusy] = useState(false)   // transcribing
  const [error, setError] = useState('')
  const [showConsent, setShowConsent] = useState(false)

  const recorderRef = useRef(null)
  const chunksRef = useRef([])
  const streamRef = useRef(null)
  // webcamActive can change between record start and stop — capture at stop time
  const webcamActiveRef = useRef(webcamActive)
  webcamActiveRef.current = webcamActive

  async function startRecording() {
    setError('')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream
      const mimeType = pickMime()
      const recorder = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream)
      recorderRef.current = recorder
      chunksRef.current = []
      recorder.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data) }
      recorder.onstop = handleStop
      recorder.start()
      setRecording(true)
    } catch {
      // mic permission denied / no mic → stay in text mode
      setError('Mic blocked')
      setRecording(false)
    }
  }

  function stopRecording() {
    const recorder = recorderRef.current
    if (recorder && recorder.state !== 'inactive') recorder.stop()
    setRecording(false)
  }

  async function handleStop() {
    // release the mic
    if (streamRef.current) streamRef.current.getTracks().forEach((t) => t.stop())
    streamRef.current = null

    const type = chunksRef.current[0]?.type || 'audio/webm'
    const blob = new Blob(chunksRef.current, { type })
    chunksRef.current = []
    if (blob.size === 0) return

    setBusy(true)
    try {
      const form = new FormData()
      form.append('audio', blob, 'recording.webm')
      // Tell the backend whether the webcam is currently covering emotion.
      // If not, it analyzes this audio's tone (librosa) as the fallback.
      form.append('webcam_active', webcamActiveRef.current ? 'true' : 'false')

      const res = await fetch(`${apiBase}/transcribe`, {
        method: 'POST',
        headers: authHeaders(),
        body: form,
      })
      if (!res.ok) throw new Error('transcribe failed')
      const data = await res.json()
      const text = (data.text || '').trim()
      // Pass the voice-tone emotion along (null when webcam had it covered)
      const voiceEmotion = data.emotion_source === 'voice' ? data.emotion : null
      if (text) onTranscript?.(text, voiceEmotion)
    } catch {
      setError('Could not transcribe')
    } finally {
      setBusy(false)
    }
  }

  // Consent gate: only ask the browser for the mic AFTER informed opt-in.
  function requestToggle() {
    // Any mic tap silences Aura — the user is about to speak (or is managing
    // the recording) and shouldn't have to talk over her.
    onMicTap?.()
    if (busy || disabled) return
    if (recording) { stopRecording(); return }
    if (localStorage.getItem(CONSENT_KEY) === 'granted') {
      startRecording()
    } else {
      setShowConsent(true)
    }
  }

  function acceptConsent() {
    localStorage.setItem(CONSENT_KEY, 'granted')
    setShowConsent(false)
    startRecording()
  }

  function declineConsent() {
    setShowConsent(false)
    setError('Mic off')
  }

  const label = recording ? 'Stop' : busy ? 'Transcribing…' : 'Speak'

  return (
    <>
      <ConsentModal
        open={showConsent}
        device="mic"
        onAccept={acceptConsent}
        onDecline={declineConsent}
      />
      <button
        type="button"
        className={`mic-btn ${recording ? 'recording' : ''}`}
        onClick={requestToggle}
        disabled={disabled || busy}
        aria-label={label}
        title={error || label}
      >
        {busy ? <span className="mic-spinner" /> : recording ? '■' : '🎤'}
      </button>
    </>
  )
}
