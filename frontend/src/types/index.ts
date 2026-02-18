export type Stage = 'upload' | 'parse' | 'character' | 'session'

export type Plot = {
  plot_id: string
  plot_goal: string
  mandatory_events: string[]
  npc: string[]
  locations: string[]
  status: 'pending' | 'in_progress' | 'completed'
  progress: number
}

export type Scene = {
  scene_id: string
  scene_goal: string
  plots: Plot[]
  status: 'pending' | 'in_progress' | 'completed'
  scene_summary: string
}

export type SystemState = {
  current_scene_id: string
  current_plot_id: string
  plot_progress: number
  scene_progress: number
  stage: Stage
}

export type StatusResponse = {
  system_state: SystemState
  scenes: Scene[]
}

export type ChatResponse = {
  response: string
  dice_result?: string | null
  stage: Stage
  current_scene_id: string
  current_plot_id: string
  plot_progress: number
  scene_progress: number
}
