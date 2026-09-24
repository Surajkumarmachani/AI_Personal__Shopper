// Backend location and API key, read from frontend/.env.local at build time:
//   VITE_API_BASE=http://127.0.0.1:8000
//   VITE_API_KEY=aura_live_...
// The key ends up inside the built app, so only ship builds to trusted users.
export const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'

export const authHeaders = () => ({ 'X-API-Key': import.meta.env.VITE_API_KEY || '' })
