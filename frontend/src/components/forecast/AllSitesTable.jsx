import React, { useEffect, useRef, useState } from 'react'
import Panel from '../ui/Panel'
import Caution from '../ui/Caution'
import LevelTag from '../ui/LevelTag'
import { getAllSites } from '../../services/datect'
import { daysBetween, formatAge, formatDate, formatPct, formatScore, formatValue } from '../../lib/format'
import { MIN_ROWS, STALE_DAYS, WEAK_COVERAGE, isStale, levelKey, siteReliability } from '../../lib/risk'

const muted = 'text-gray-500'

function LastCell({ forecast }) {
  const last = forecast?.lastMeasurement
  if (!last) return <span className={muted}>none</span>
  if (!forecast.anchorDate) return formatValue(last.value)
  const age = daysBetween(last.date, forecast.anchorDate)
  return (
    <>
      {formatValue(last.value)}{' '}
      <span className={muted}>
        ({formatAge(age)} before){isStale(last.date, forecast.anchorDate) ? ' *' : ''}
      </span>
    </>
  )
}

function ForecastCells({ forecast, config }) {
  if (!forecast) return <td colSpan={2} className={`pr-3 ${muted}`}>No forecast</td>
  if (forecast.error) return <td colSpan={2} className={`pr-3 ${muted}`}>{forecast.error}</td>
  const { band, alert } = forecast
  const level = levelKey(alert)
  const score = level === 'already' ? null : alert?.probability
  return (
    <>
      <td className="pr-3">
        {forecast.value == null ? '—' : formatValue(forecast.value)}{' '}
        {band && <span className={muted}>[{formatValue(band.lo)}–{formatValue(band.hi)}]</span>}
      </td>
      <td className="pr-3">
        <LevelTag level={level} config={config} />{' '}
        {score != null && <span className={muted}>{formatScore(score)}</span>}
      </td>
    </>
  )
}

/** `summary` is null until it arrives, or for good if it failed: that says nothing about the site. */
function HeldCell({ summary, site }) {
  if (!summary) return <span className={muted}>—</span>
  const reliability = summary.reliability.holdout.perSite[site]
  const status = siteReliability(reliability)
  if (status === 'too-few') return <span className={muted}>too few</span>
  return <>{formatPct(reliability.band.coverage)}{status === 'weak' ? ' †' : ''}</>
}

/**
 * Every site north to south for one week; selecting a row switches the page's site.
 * Fetches on its own, once `ready`, so it never competes with the page's forecast.
 * @param {{ week: string, site: string, ready: boolean, config: import('../../services/contract').Config,
 *   summary: import('../../services/contract').Summary | null, onSelectSite: (s: string) => void }} props
 */
export default function AllSitesTable({ week, site, ready, config, summary, onSelectSite }) {
  const [shown, setShown] = useState(null) // { week, rows }
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const latest = useRef(0)

  useEffect(() => {
    if (!ready) return undefined
    const id = ++latest.current
    setLoading(true)
    setError(null)
    getAllSites(week)
      .then((r) => latest.current === id && (setShown({ week, rows: r }), setLoading(false)))
      .catch((e) => latest.current === id && (setShown(null), setError(e?.message || String(e)), setLoading(false)))
    return () => {
      latest.current++
    }
  }, [week, ready])

  const bySite = new Map((shown?.rows ?? []).map((f) => [f.site, f]))
  const heldLabel = config.windows.holdout.label

  return (
    <Panel className="p-5">
      <h2 className="text-base font-semibold">All sites{shown && `, week of ${formatDate(shown.week)}`}</h2>
      <p className="text-xs text-gray-500 mt-0.5">North to south. Select a row to switch site.</p>
      {(loading || (!ready && !shown)) && (
        <p className="text-xs text-gray-500 mt-2">
          Loading all sites for the week of {formatDate(week)}; this can take a minute when forecasts are not precomputed
        </p>
      )}
      {error && <Caution>Could not load all sites: {error}</Caution>}
      {shown && (
        <div className={`overflow-x-auto mt-3 ${loading ? 'opacity-60' : ''}`}>
          <table className="w-full text-sm tabular-nums">
            <thead className="text-left text-xs text-gray-500 border-b border-gray-200">
              <tr>
                <th className="py-2 pr-3 font-normal">Site</th>
                <th className="pr-3 font-normal">Last measured (µg/g)</th>
                <th className="pr-3 font-normal">Forecast [{formatPct(config.bandLevel)} range]</th>
                <th className="pr-3 font-normal">Spike risk, classifier score</th>
                <th className="font-normal">Range held, {heldLabel}</th>
              </tr>
            </thead>
            <tbody>
              {config.sites.map((name) => {
                const f = bySite.get(name)
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
                    <td className="pr-3 whitespace-nowrap"><LastCell forecast={f} /></td>
                    <ForecastCells forecast={f} config={config} />
                    <td className="whitespace-nowrap"><HeldCell summary={summary} site={name} /></td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
      {shown && (
        <p className="text-xs text-gray-500 mt-2">
          * Last sample more than {STALE_DAYS / 7} weeks before the forecast. † Range held below{' '}
          {formatPct(WEAK_COVERAGE)} of at least {MIN_ROWS} measurements.
        </p>
      )}
    </Panel>
  )
}
