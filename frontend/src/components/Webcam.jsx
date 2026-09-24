import { useRef, useState, useEffect } from 'react'
import ConsentModal from './ConsentModal'
import './Webcam.css'

const CONSENT_KEY = 'aura_consent_camera'

const EMOJI = {
  happy: '😊', sad: '😢', angry: '😠', surprised: '😲',
  fearful: '😨', disgusted: '😖', calm: '😌', neutral: '😐',
}

export default function Webcam({ apiBase, sessionId, onEmotion, onActiveChange }) {
  const [enabled, setEnabled] = useState(false)
  const [emotion, setEmotion] = useState('neutral')
  const [error, setError] = useState('')
  const [showConsent, setShowConsent] = useState(false)

  const videoRef    = useRef(null)
  const canvasRef   = useRef(null)   // hidden — only used to grab frames
  const streamRef   = useRef(null)
  const intervalRef = useRef(null)
  const inFlightRef = useRef(false)  // prevents overlapping requests
  const onEmotionRef = useRef(onEmotion)
  const onActiveRef  = useRef(onActiveChange)

  // Keep the latest callbacks without restarting the capture loop
  useEffect(() => { onEmotionRef.current = onEmotion }, [onEmotion])
  useEffect(() => { onActiveRef.current = onActiveChange }, [onActiveChange])

  // Attach the camera stream once the <video> element is actually mounted
  useEffect(() => {
    if (enabled && videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current
    }
  }, [enabled])

  // Stop everything on unmount
  useEffect(() => () => stopCamera(), [])

  async function captureAndDetect() {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas || video.readyState < 2) return  // video not ready yet
    if (inFlightRef.current) return                         // still awaiting last one
    inFlightRef.current = true
    try {
      canvas.width = 320
      canvas.height = 240
      canvas.getContext('2d').drawImage(video, 0, 0, 320, 240)
      const image = canvas.toDataURL('image/jpeg', 0.6)

      const res = await fetch(`${apiBase}/emotion`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, image }),
      })
      if (!res.ok) throw new Error('emotion request failed')
      const data = await res.json()

      const latest = data.emotion || 'neutral'
      setEmotion(latest)
      onEmotionRef.current?.(latest)
    } catch {
      // backend/network hiccup — keep the last known emotion, don't break the loop
    } finally {
      inFlightRef.current = false
    }
  }

  // Consent gate: only ask the browser for the camera AFTER informed opt-in.
  function requestEnable() {
    if (localStorage.getItem(CONSENT_KEY) === 'granted') {
      startCamera()
    } else {
      setShowConsent(true)
    }
  }

  function acceptConsent() {
    localStorage.setItem(CONSENT_KEY, 'granted')
    setShowConsent(false)
    startCamera()
  }

  function declineConsent() {
    setShowConsent(false)
    setError('Camera off — chatting in text mode.')
  }

  async function startCamera() {
    setError('')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 320, height: 240 },
        audio: false,
      })
      streamRef.current = stream
      setEnabled(true)                                   // mounts <video>
      onActiveRef.current?.(true)
      intervalRef.current = setInterval(captureAndDetect, 1000)
    } catch {
      // Permission denied / no camera → clean fallback to neutral (text mode)
      setError('Camera off — chatting in text mode.')
      setEnabled(false)
      onActiveRef.current?.(false)
      onEmotionRef.current?.('neutral')
    }
  }

  function stopCamera() {
    if (intervalRef.current) clearInterval(intervalRef.current)
    intervalRef.current = null
    if (streamRef.current) streamRef.current.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    setEnabled(false)
    setEmotion('neutral')
    onActiveRef.current?.(false)
    onEmotionRef.current?.('neutral')
  }

  return (
    <div className="webcam">
      <ConsentModal
        open={showConsent}
        device="camera"
        onAccept={acceptConsent}
        onDecline={declineConsent}
      />

      {/* hidden canvas used only to grab frames */}
      <canvas ref={canvasRef} style={{ display: 'none' }} />

      {enabled ? (
        <div className="webcam-live">
          <video ref={videoRef} className="webcam-video" autoPlay playsInline muted />
          <div className="webcam-chip">
            <span className="webcam-emoji">{EMOJI[emotion] || '😐'}</span>
            <span className="webcam-label">{emotion}</span>
          </div>
          <button className="webcam-stop" onClick={stopCamera} aria-label="Turn camera off">
            ✕
          </button>
        </div>
      ) : (
        <button className="webcam-enable" onClick={requestEnable}>
          📷 Enable camera
        </button>
      )}

      {error && !enabled && <span className="webcam-error">{error}</span>}
    </div>
  )
}
