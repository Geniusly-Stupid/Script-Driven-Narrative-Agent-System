import { ChatResponse, StatusResponse } from '../types'

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init)
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail ?? 'Request failed')
  }
  return res.json() as Promise<T>
}

export const api = {
  getStatus: () => request<StatusResponse>('/api/workflow/status'),
  getScenes: () => request<{ scenes: StatusResponse['scenes'] }>('/api/workflow/scenes'),
  uploadScript: async (file: File) => {
    const form = new FormData()
    form.append('file', file)
    return request('/api/workflow/upload-script', { method: 'POST', body: form })
  },
  confirmStructure: () => request('/api/workflow/confirm-structure', { method: 'POST' }),
  createCharacter: (payload: {
    name: string
    background: string
    traits: string[]
    stats: Record<string, number>
    special_skills: string[]
  }) => request('/api/workflow/character', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }),
  sendMessage: (message: string) =>
    request<ChatResponse>('/api/workflow/session/message', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    }),
}
