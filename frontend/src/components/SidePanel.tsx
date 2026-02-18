export function SidePanel({
  scene,
  plot,
  sceneProgress,
  plotProgress,
}: {
  scene: string
  plot: string
  sceneProgress: number
  plotProgress: number
}) {
  return (
    <aside className="space-y-4 rounded-2xl border border-slate-200 bg-white/80 p-4 shadow-soft backdrop-blur dark:border-slate-700 dark:bg-slate-900/70">
      <div>
        <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Current Scene</p>
        <p className="font-display text-lg text-slate-800 dark:text-slate-100">{scene || 'Not started'}</p>
      </div>
      <div>
        <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Current Plot</p>
        <p className="font-display text-lg text-slate-800 dark:text-slate-100">{plot || 'Not started'}</p>
      </div>
      <div>
        <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">Plot Progress</p>
        <div className="h-2 rounded-full bg-slate-200 dark:bg-slate-700">
          <div className="h-2 rounded-full bg-sky-500 transition-all" style={{ width: `${Math.round(plotProgress * 100)}%` }} />
        </div>
      </div>
      <div>
        <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">Scene Progress</p>
        <div className="h-2 rounded-full bg-slate-200 dark:bg-slate-700">
          <div className="h-2 rounded-full bg-emerald-500 transition-all" style={{ width: `${Math.round(sceneProgress * 100)}%` }} />
        </div>
      </div>
    </aside>
  )
}
