import React from 'react'
import { categoryScaleX } from '../../lib/scale'
import { BLUE, RED } from '../../lib/colors'

/**
 * Four gray category segments with the forecast tick, the range (when there is one)
 * and the closure-level line.
 * @param {{ value: number, band: { lo: number, hi: number } | null,
 *   config: import('../../services/contract').Config }} props
 */
export default function CategoryScaleBar({ value, band, config }) {
  const { categories, spikeThreshold } = config
  const width = 100 / categories.length
  const lo = band ? categoryScaleX(band.lo, categories) : 0
  const hi = band ? categoryScaleX(band.hi, categories) : 0
  const p = categoryScaleX(value, categories)
  const line = categoryScaleX(spikeThreshold, categories)
  const lastIdx = categories.length - 1

  return (
    <div className="mt-4">
      <div className="relative h-6">
        <div className="absolute inset-0 flex gap-0.5">
          {categories.map((c) => (
            <div key={c.label} className="flex-1 bg-gray-100" />
          ))}
        </div>
        {band && (
          <div
            className="absolute top-1.5 bottom-1.5 border"
            style={{
              left: `${lo}%`,
              width: `${Math.max(hi - lo, 0.8)}%`,
              background: 'rgba(37,99,235,.18)',
              borderColor: 'rgba(37,99,235,.6)',
            }}
          />
        )}
        <div className="absolute top-0 bottom-0 w-0.5 -ml-px" style={{ left: `${p}%`, background: BLUE }} />
        <div className="absolute -top-1 -bottom-1 w-px" style={{ left: `${line}%`, background: RED }} />
      </div>
      <div className="relative h-4 mt-0.5 text-[10px] text-gray-500 tabular-nums">
        {categories.map((c, i) => (
          <span
            key={c.label}
            className="absolute -translate-x-1/2"
            style={{ left: `${i * width}%`, ...(c.lo === spikeThreshold ? { color: RED } : {}) }}
          >
            {c.lo}
          </span>
        ))}
        <span className="absolute right-0">{categories[lastIdx].lo * 2}+</span>
      </div>
      <div className="grid text-[10px] text-gray-500 text-center" style={{ gridTemplateColumns: `repeat(${categories.length}, 1fr)` }}>
        {categories.map((c) => (
          <span key={c.label}>{c.label}</span>
        ))}
      </div>
    </div>
  )
}
