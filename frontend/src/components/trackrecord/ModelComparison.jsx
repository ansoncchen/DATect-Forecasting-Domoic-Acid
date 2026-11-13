import React from 'react'

const ROWS = [
  ['ensemble', 'DATect model'],
  ['naive', 'Repeat the last measurement'],
  ['linear', 'Simple linear model (ridge)'],
]

/**
 * @param {{ models: import('../../services/contract').Reliability['models'] }} props
 */
export default function ModelComparison({ models }) {
  return (
    <table className="text-sm tabular-nums mt-2">
      <thead className="text-left text-xs text-gray-500 border-b border-gray-200">
        <tr>
          <th className="py-2 pr-6 font-normal">Method</th>
          <th className="pr-6 font-normal">Typical error (µg/g)</th>
          <th className="font-normal">R² (higher is better)</th>
        </tr>
      </thead>
      <tbody>
        {ROWS.map(([key, label]) => {
          const m = models?.[key]
          return (
            <tr key={key} className="border-b border-gray-100 last:border-0">
              <td className="py-1.5 pr-6 whitespace-nowrap">{label}</td>
              {m && m.r2 != null && m.mae != null ? (
                <>
                  <td className="pr-6">{m.mae.toFixed(1)}</td>
                  <td>{m.r2.toFixed(2)}</td>
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
