import React from 'react'
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
 * @param {{ config: import('../../services/contract').Config,
 *   perSite: Object<string, import('../../services/contract').Reliability>,
 *   site: string, label: string, onSelectSite: (s: string) => void }} props
 */
export default function PerSiteTable({ config, perSite, site, label, onSelectSite }) {
  return (
    <Panel className="p-5">
      <h2 className="text-base font-semibold">By site, {label}</h2>
      <p className="text-xs text-gray-500 mt-0.5">Select a row to filter to that site.</p>
      <div className="overflow-x-auto mt-3">
        <table className="w-full text-sm tabular-nums">
          <thead className="text-left text-xs text-gray-500 border-b border-gray-200">
            <tr>
              <th className="py-2 pr-3 font-normal">Site</th>
              <th className="pr-3 font-normal">N</th>
              <th className="pr-3 font-normal">R²</th>
              <th className="pr-3 font-normal">Log R²</th>
              <th className="pr-3 font-normal">MAE (µg/g)</th>
              <th className="pr-3 font-normal">Range held</th>
              <th className="pr-3 font-normal">Jumps caught</th>
              <th className="font-normal">Alarms followed</th>
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
                      <td className="pr-3 whitespace-nowrap"><Stat stat={r.r2} digits={2} /></td>
                      <td className="pr-3 whitespace-nowrap"><Stat stat={r.logR2} digits={2} /></td>
                      <td className="pr-3 whitespace-nowrap"><Stat stat={r.mae} digits={1} /></td>
                      <td className="pr-3 whitespace-nowrap"><RangeHeld reliability={r} /></td>
                      <td className="pr-3 whitespace-nowrap">{r.onsets.caught} of {r.onsets.n}</td>
                      <td className="whitespace-nowrap">{r.alarms.followed} of {r.alarms.n}</td>
                    </>
                  ) : (
                    <td colSpan={7} className={muted}>No measurements in this window</td>
                  )}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </Panel>
  )
}
