import React from 'react'
import { LEVEL_COLORS } from '../../lib/colors'

/** @param {{ level: 'already' | 'quiet' | 'watch' | 'warning' | 'unavailable', config: import('../../services/contract').Config }} props */
export default function LevelTag({ level, config }) {
  let label
  if (level === 'already') label = `Already ≥ ${config.spikeThreshold}`
  else if (level === 'unavailable') label = 'Unavailable'
  else label = config.levels.find((l) => l.name.toLowerCase() === level)?.name ?? level

  const color = LEVEL_COLORS[label]
  return (
    <span className="inline-flex items-center gap-1.5">
      {color && <span className="inline-block w-2 h-2" style={{ background: color }} />}
      {label}
    </span>
  )
}
