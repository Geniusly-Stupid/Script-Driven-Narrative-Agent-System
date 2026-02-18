import { useState } from 'react'

import { api } from '../api/client'
import { ChatResponse } from '../types'

type ChatTurn = { user: string; agent: string; dice?: string | null }

export function ChatBox({ onStateUpdate }: { onStateUpdate: (payload: ChatResponse) => void }) {
  const [turns, setTurns] = useState<ChatTurn[]>([])
  const [message, setMessage] = useState('')
  const [loading, setLoading] = useState(false)

  const send = async () => {
    if (!message.trim() || loading) return
    const user = message.trim()
    setMessage('')
    setLoading(true)
    try {
      const res = await api.sendMessage(user)
      setTurns((prev) => [...prev, { user, agent: res.response, dice: res.dice_result }])
      onStateUpdate(res)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex h-[560px] flex-col rounded-2xl border border-slate-200 bg-white/80 shadow-soft backdrop-blur dark:border-slate-700 dark:bg-slate-900/70">
      <div className="flex-1 space-y-4 overflow-y-auto p-4">
        {turns.length === 0 && <p className="text-sm text-slate-500">Narrative session is live. Ask what your character does next.</p>}
        {turns.map((turn, idx) => (
          <div key={idx} className="space-y-2">
            <div className="ml-auto max-w-[80%] rounded-xl bg-sky-600 px-3 py-2 text-sm text-white">{turn.user}</div>
            <div className="max-w-[90%] rounded-xl bg-slate-100 px-3 py-2 text-sm text-slate-700 dark:bg-slate-800 dark:text-slate-100">{turn.agent}</div>
            {turn.dice && <div className="animate-pulseUp text-xs font-bold text-amber-600 dark:text-amber-400">Dice: {turn.dice}</div>}
          </div>
        ))}
      </div>
      <div className="border-t border-slate-200 p-3 dark:border-slate-700">
        <div className="flex items-center gap-2">
          <input
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') void send()
            }}
            className="flex-1 rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm outline-none ring-sky-400 transition focus:ring-2 dark:border-slate-600 dark:bg-slate-950"
            placeholder="Type your action (example: roll d20 to sneak past guard)"
          />
          <button
            onClick={() => void send()}
            disabled={loading}
            className="rounded-xl bg-emerald-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-emerald-500 disabled:opacity-60"
          >
            {loading ? 'Thinking...' : 'Send'}
          </button>
        </div>
      </div>
    </div>
  )
}
