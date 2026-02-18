import { useEffect, useState } from 'react'

import { api } from '../api/client'
import { StatusResponse } from '../types'

export function useStatus() {
  const [status, setStatus] = useState<StatusResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string>('')

  const refresh = async () => {
    setLoading(true)
    try {
      const next = await api.getStatus()
      setStatus(next)
      setError('')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load status')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void refresh()
  }, [])

  return { status, loading, error, refresh }
}
