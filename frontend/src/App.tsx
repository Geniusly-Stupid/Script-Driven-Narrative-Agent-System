import { useMemo, useState } from 'react'
import { motion } from 'framer-motion'

import { api } from './api/client'
import { ChatBox } from './components/ChatBox'
import { SidePanel } from './components/SidePanel'
import { Stepper } from './components/Stepper'
import { useStatus } from './hooks/useStatus'
import { ChatResponse, Stage } from './types'

function App() {
  const { status, loading, error, refresh } = useStatus()
  const [busy, setBusy] = useState(false)
  const [actionError, setActionError] = useState('')
  const [uploadFile, setUploadFile] = useState<File | null>(null)
  const [name, setName] = useState('')
  const [background, setBackground] = useState('')
  const [traits, setTraits] = useState('')
  const [stats, setStats] = useState('Strength:8,Intelligence:10,Agility:9')
  const [skills, setSkills] = useState('Stealth,Persuasion')

  const stage: Stage = status?.system_state.stage ?? 'upload'

  const activeState = useMemo(
    () => ({
      scene: status?.system_state.current_scene_id ?? '',
      plot: status?.system_state.current_plot_id ?? '',
      sceneProgress: status?.system_state.scene_progress ?? 0,
      plotProgress: status?.system_state.plot_progress ?? 0,
    }),
    [status],
  )

  const submitUpload = async () => {
    if (!uploadFile) return
    setBusy(true)
    setActionError('')
    try {
      await api.uploadScript(uploadFile)
      await refresh()
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Upload failed')
    } finally {
      setBusy(false)
    }
  }

  const confirmStructure = async () => {
    setBusy(true)
    setActionError('')
    try {
      await api.confirmStructure()
      await refresh()
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Confirm failed')
    } finally {
      setBusy(false)
    }
  }

  const submitCharacter = async () => {
    setBusy(true)
    setActionError('')
    try {
      const statObj = Object.fromEntries(
        stats
          .split(',')
          .map((s) => s.trim())
          .filter(Boolean)
          .map((pair) => {
            const [k, v] = pair.split(':')
            return [k.trim(), Number(v)]
          }),
      )

      await api.createCharacter({
        name,
        background,
        traits: traits.split(',').map((x) => x.trim()).filter(Boolean),
        stats: statObj,
        special_skills: skills.split(',').map((x) => x.trim()).filter(Boolean),
      })
      await refresh()
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Character creation failed')
    } finally {
      setBusy(false)
    }
  }

  const onChatStateUpdate = (payload: ChatResponse) => {
    void refresh()
    if (payload.stage !== 'session') return
  }

  return (
    <div className="mx-auto max-w-7xl p-4 md:p-8">
      <header className="mb-6 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="font-display text-3xl font-extrabold tracking-tight">Script-Driven Narrative Agent System</h1>
          <p className="text-sm text-slate-600 dark:text-slate-300">LangGraph-powered sequential storytelling workflow with MongoDB + Chroma.</p>
        </div>
      </header>

      <Stepper stage={stage} />

      {error && <div className="mt-4 rounded-xl border border-red-200 bg-red-50 p-3 text-sm text-red-700">{error}</div>}
      {actionError && <div className="mt-4 rounded-xl border border-red-200 bg-red-50 p-3 text-sm text-red-700">{actionError}</div>}

      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} className="mt-6 grid gap-6 lg:grid-cols-[1.6fr_0.8fr]">
        <section className="space-y-4 rounded-2xl border border-slate-200 bg-white/85 p-5 shadow-soft dark:border-slate-700 dark:bg-slate-900/80">
          {loading ? (
            <p className="text-sm text-slate-500">Loading workflow state...</p>
          ) : (
            <>
              {stage === 'upload' && (
                <div className="space-y-3">
                  <h2 className="font-display text-xl font-bold">Step 1: Upload Script PDF</h2>
                  <input type="file" accept="application/pdf" onChange={(e) => setUploadFile(e.target.files?.[0] ?? null)} />
                  <button onClick={() => void submitUpload()} disabled={!uploadFile || busy} className="rounded-xl bg-sky-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-sky-500 disabled:opacity-60">
                    {busy ? 'Parsing...' : 'Upload & Parse'}
                  </button>
                </div>
              )}

              {stage === 'parse' && (
                <div className="space-y-4">
                  <h2 className="font-display text-xl font-bold">Step 2: Script Parsing & Structure Review</h2>
                  <div className="grid gap-3 md:grid-cols-2">
                    {status?.scenes.map((scene) => (
                      <article key={scene.scene_id} className="rounded-xl border border-slate-200 bg-slate-50 p-3 dark:border-slate-700 dark:bg-slate-800">
                        <h3 className="font-display text-base font-bold">{scene.scene_id}</h3>
                        <p className="text-sm text-slate-600 dark:text-slate-200">{scene.scene_goal}</p>
                        <p className="mt-2 text-xs uppercase tracking-wide text-slate-500">Plots: {scene.plots.length}</p>
                      </article>
                    ))}
                  </div>
                  <button onClick={() => void confirmStructure()} disabled={busy} className="rounded-xl bg-emerald-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-emerald-500 disabled:opacity-60">
                    {busy ? 'Saving...' : 'Confirm Structure'}
                  </button>
                </div>
              )}

              {stage === 'character' && (
                <div className="space-y-3">
                  <h2 className="font-display text-xl font-bold">Step 3: Character Creation</h2>
                  <input className="w-full rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm dark:border-slate-600 dark:bg-slate-950" placeholder="Name" value={name} onChange={(e) => setName(e.target.value)} />
                  <textarea className="w-full rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm dark:border-slate-600 dark:bg-slate-950" placeholder="Background" value={background} onChange={(e) => setBackground(e.target.value)} rows={3} />
                  <input className="w-full rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm dark:border-slate-600 dark:bg-slate-950" placeholder="Traits comma separated" value={traits} onChange={(e) => setTraits(e.target.value)} />
                  <input className="w-full rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm dark:border-slate-600 dark:bg-slate-950" placeholder="Stats (Strength:8,Intelligence:10)" value={stats} onChange={(e) => setStats(e.target.value)} />
                  <input className="w-full rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm dark:border-slate-600 dark:bg-slate-950" placeholder="Special skills comma separated" value={skills} onChange={(e) => setSkills(e.target.value)} />
                  <button onClick={() => void submitCharacter()} disabled={busy || !name || !background} className="rounded-xl bg-ember px-4 py-2 text-sm font-semibold text-white transition hover:opacity-90 disabled:opacity-60">
                    {busy ? 'Creating...' : 'Create Character'}
                  </button>
                </div>
              )}

              {stage === 'session' && (
                <div className="space-y-3">
                  <h2 className="font-display text-xl font-bold">Step 4: Narrative Session</h2>
                  <ChatBox onStateUpdate={onChatStateUpdate} />
                </div>
              )}
            </>
          )}
        </section>

        <SidePanel scene={activeState.scene} plot={activeState.plot} sceneProgress={activeState.sceneProgress} plotProgress={activeState.plotProgress} />
      </motion.div>
    </div>
  )
}

export default App
