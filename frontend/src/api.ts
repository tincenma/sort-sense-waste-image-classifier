import type { AppInfo, PredictionResponse } from './types'

export const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')

async function checkResponse(response: Response): Promise<void> {
  if (response.ok) return
  const body = await response.json().catch(() => null)
  throw new Error(typeof body?.detail === 'string' ? body.detail : 'Something went wrong. Please try again.')
}

export async function fetchInfo(signal: AbortSignal): Promise<AppInfo> {
  const response = await fetch(`${API_BASE}/api/info`, { signal })
  await checkResponse(response)
  return response.json()
}

export async function fetchExample(path: string, signal: AbortSignal): Promise<Blob> {
  const response = await fetch(`${API_BASE}${path}`, { signal })
  await checkResponse(response)
  return response.blob()
}

export async function classify(file: File, signal: AbortSignal): Promise<PredictionResponse> {
  const form = new FormData()
  form.append('file', file)
  form.append('model', 'both')
  const response = await fetch(`${API_BASE}/api/predict`, { method: 'POST', body: form, signal })
  await checkResponse(response)
  return response.json()
}
