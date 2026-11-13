import React, { useState } from 'react'
import Panel from '../ui/Panel'
import { formatPct, formatStat } from '../../lib/format'
import { siteReliability } from '../../lib/risk'

const muted = 'text-gray-500'

/** The share of measurements inside the range, as 'inside of banded (pct)'; 'too few' when the API's sample is small. */
export function RangeHeld({ reliability }) {
  if (siteReliability(reliability) === 'too-few') return <span className={muted}>too few</span>
  const { inside, nBanded, coverage } = reliability.band
  return <>{inside} of {nBanded} ({formatPct(coverage)})</>
}

function Stat({ stat, digits }) {
  return stat ? <>{formatStat(stat, digits)}</> : <span className={muted}>too few</span>
}

/**
 * One row per site. The plain columns show by default; R², log R² and intervals are one click away.
 * @param {{ config: import('../../services/contract').Config,
 *   perSite: Object<string, import('../../services/contract').Reliability>,
 *   site: string, label: string, onSelectSite: (s: string) => void }} props
 */
export default function PerSiteTable({ config, perSite, site, label, onSelectSite }) {
  const [full, setFull] = useState(false)
  const columns = full ? 8 : 5
  return (
    <Panel className="p-5">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <h2 className="text-base font-semibold">By site, {label}</h2>
          <p className="text-xs text-gray-500 mt-0.5">North to south. Click a site to see only its results.</p>
        </div>
        <label className="inline-flex items-center gap-1.5 text-xs text-gray-700 cursor-pointer">
          <input type="checkbox" checked={full} onChange={(e) => setFull(e.target.checked)} />
          Show all statistics
        </label>
      </div>
      <div className="overflow-x-auto mt-3">
        <table className="w-full text-sm tabular-nums">
          <thead className="text-left text-xs text-gray-500 border-b border-gray-200">
            <tr>
              <th className="py-2 pr-3 font-normal">Site</th>
              <th className="pr-3 font-normal">Measurements</th>
              <th className="pr-3 font-normal">Typical error (µg/g)</th>
              <th className="pr-3 font-normal">Inside likely range</th>
              <th className="pr-3 font-normal">Sudden rises caught</th>
              {full && (
                <>
                  <th className="pr-3 font-normal">Alarms followed by a rise</th>
                  <th className="pr-3 font-normal">R²</th>
                  <th className="font-normal">Log R²</th>
                </>
              )}
            </tr>
          </thead>
          <tbody>
            {config.sites.map((name) => {
              const r = perSite[name]
              const selected = name === site
              return (
                <tr
                  key={name}
                  onClick={() => onSelectSite(name)}
                  className={`border-b border-gray-100 last:border-0 cursor-pointer hover:bg-gray-50 ${selected ? 'bg-gray-50 font-medium' : ''}`}
                >
                  <td className="py-1.5 pr-3 whitespace-nowrap">
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation()
                        onSelectSite(name)
                      }}
                      aria-current={selected ? 'true' : undefined}
                      className="text-left"
                    >
                      {name}
                    </button>
                  </td>
                  {r ? (
                    <>
                      <td className="pr-3">{r.n}</td>
                      <td className="pr-3 whitespace-nowrap">
                        {full ? <Stat stat={r.mae} digits={1} /> : r.mae ? r.mae.value.toFixed(1) : <span className={muted}>too few</span>}
                      </td>
                      <td className="pr-3 whitespace-nowrap"><RangeHeld reliability={r} /></td>
                      <td className="pr-3 whitespace-nowrap">{r.onsets.n ? `${r.onsets.caught} of ${r.onsets.n}` : <span className={muted}>no rises</span>}</td>
                      {full && (
                        <>
                          <td className="pr-3 whitespace-nowrap">{r.alarms.followed} of {r.alarms.n}</td>
                          <td className="pr-3 whitespace-nowrap"><Stat stat={r.r2} digits={2} /></td>
                          <td className="whitespace-nowrap"><Stat stat={r.logR2} digits={2} /></td>
                        </>
                      )}
                    </>
                  ) : (
                    <td colSpan={columns - 1} className={muted}>No measurements in this window</td>
                  )}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      {full && <p className="mt-2 text-xs text-gray-500">Brackets are 95% bootstrap intervals.</p>}
    </Panel>
  )
}
