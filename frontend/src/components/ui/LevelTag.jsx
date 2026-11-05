import React from 'react'
import { AMBER, RED } from '../../lib/colors'

const SQUARE = { quiet: '#9ca3af', watch: AMBER, warning: RED }

/** @param {{ level: 'already' | 'quiet' | 'watch' | 'warning' | 'unavailable', config: import('../../services/contract').Config }} props */
export default function LevelTag({ level, config }) {
  let label
  if (level === 'already') label = `Already ≥ ${config.spikeThreshold}`
  else if (level === 'unavailable') label = 'Unavailable'
  else label = config.levels.find((l) => l.name.toLowerCase() === level)?.name ?? level

  const color = SQUARE[level]
  return (
    <span className="inline-flex items-center gap-1.5">
      {color && <span className="inline-block w-2 h-2" style={{ background: color }} />}
      {label}
    </span>
  )
}
