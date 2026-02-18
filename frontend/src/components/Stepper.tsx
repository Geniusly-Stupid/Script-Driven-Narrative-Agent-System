import { motion } from 'framer-motion'

import { Stage } from '../types'

const labels: Stage[] = ['upload', 'parse', 'character', 'session']

export function Stepper({ stage }: { stage: Stage }) {
  const currentIndex = labels.indexOf(stage)

  return (
    <div className="rounded-2xl border border-sky-200/50 bg-white/70 p-4 shadow-soft backdrop-blur dark:border-slate-700 dark:bg-slate-900/70">
      <div className="flex flex-wrap items-center gap-3">
        {labels.map((name, index) => {
          const done = index < currentIndex
          const active = index === currentIndex
          return (
            <div key={name} className="flex items-center gap-2">
              <motion.div
                initial={{ scale: 0.92, opacity: 0.5 }}
                animate={{ scale: active ? 1.06 : 1, opacity: 1 }}
                className={`h-8 w-8 rounded-full text-center text-sm font-bold leading-8 ${done ? 'bg-emerald-500 text-white' : active ? 'bg-sky-500 text-white' : 'bg-slate-200 text-slate-500 dark:bg-slate-700 dark:text-slate-300'}`}
              >
                {index + 1}
              </motion.div>
              <span className={`font-display text-xs uppercase tracking-wide ${active ? 'text-sky-700 dark:text-sky-300' : 'text-slate-500 dark:text-slate-400'}`}>
                {name}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
