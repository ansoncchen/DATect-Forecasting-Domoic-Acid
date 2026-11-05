import React from 'react'

const ROWS = [
  ['ensemble', 'Ensemble'],
  ['naive', 'Last value carried forward'],
  ['linear', 'Ridge'],
]

/**
 * @param {{ models: import('../../services/contract').Reliability['models'] }} props
 */
export default function ModelComparison({ models }) {
  return (
    <table className="text-sm tabular-nums mt-2">
      <thead className="text-left text-xs text-gray-500 border-b border-gray-200">
        <tr>
          <th className="py-2 pr-6 font-normal">Model</th>
          <th className="pr-6 font-normal">R²</th>
          <th className="font-normal">MAE (µg/g)</th>
        </tr>
      </thead>
      <tbody>
        {ROWS.map(([key, label]) => {
          const m = models?.[key]
          return (
            <tr key={key} className="border-b border-gray-100 last:border-0">
              <td className="py-1.5 pr-6 whitespace-nowrap">{label}</td>
              {m ? (
                <>
                  <td className="pr-6">{m.r2.toFixed(2)}</td>
                  <td>{m.mae.toFixed(1)}</td>
                </>
              ) : (
                <td colSpan={2} className="text-gray-500">too few</td>
              )}
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}
