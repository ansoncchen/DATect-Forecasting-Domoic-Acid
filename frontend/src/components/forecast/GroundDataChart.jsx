import React, { useEffect, useMemo, useRef, useState } from 'react'
import Plot from '../../utils/Plot'
import Panel from '../ui/Panel'
import Caution from '../ui/Caution'
import { HISTORY_SINCE, getSiteHistory } from '../../services/datect'
import { buildGroundTraces, noRowsNote } from '../../lib/groundTraces'

const RANGES = [
  ['6m', '6 mo'],
  ['1y', '1 yr'],
  ['2y', '2 yr'],
  ['all', 'Since 2019'],
]
const PLOT_CONFIG = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ['lasso2d', 'select2d'] }

/**
 * Real lab measurements with the forecast each was given a week earlier and its range.
 * Fetches its own history per site; a failure stays inside this panel.
 * @param {{ site: string, week: string, forecast: import('../../services/contract').Forecast | null,
 *   config: import('../../services/contract').Config, loading?: boolean }} props
 */
export default function GroundDataChart({ site, week, forecast, config, loading = false }) {
  const [history, setHistory] = useState(null) // { site, rows }
  const [error, setError] = useState(null)
  const [range, setRange] = useState('2y')
  const [showAfter, setShowAfter] = useState(false)
  const latest = useRef(0)
  const lastChart = useRef(null)

  useEffect(() => {
    const id = ++latest.current
    setError(null)
    getSiteHistory(site)
      .then((r) => latest.current === id && setHistory({ site, rows: r }))
      .catch((e) => latest.current === id && (setHistory(null), setError(e?.message || String(e))))
  }, [site])

  useEffect(() => {
    setShowAfter(false)
  }, [site, week])

  // Plot history only against the forecast of the same site; while they differ (a request
  // is in flight) keep the last matched chart, dimmed, under its own site's heading.
  const matched = history && forecast && history.site === forecast.site
  const built = useMemo(
    () => (matched ? buildGroundTraces(history.rows, forecast, config, { range, showAfter }) : null),
    [matched, history, forecast, config, range, showAfter],
  )
  if (built) lastChart.current = { site: forecast.site, ...built }
  if (error) lastChart.current = null
  const shown = lastChart.current
  const stale = !built && shown !== null

  if (!forecast) return null
  const { holdout, validation } = config.windows
  const past = forecast.date < config.latestWeek

  return (
    <Panel className="p-5">
      <div className="flex flex-wrap items-start gap-3">
        <div>
          <h2 className="text-base font-semibold">Past measurements and forecasts, {shown ? shown.site : site}</h2>
          <p className="text-sm text-gray-600 max-w-3xl mt-0.5">
            Dots are lab measurements. The blue line is what the model forecast a week before each one, with its likely
            range shaded; hollow dots fell outside that range. The red line marks {config.spikeThreshold} µg/g.
          </p>
          <p className="text-xs text-gray-500 max-w-3xl mt-1">
            Ticks under the chart mark Watch and Warning weeks. The model was tuned on {validation.label}; {holdout.label} was
            never used for tuning.
          </p>
        </div>
        <div className="ml-auto flex text-xs border border-gray-300 rounded overflow-hidden divide-x divide-gray-300">
          {RANGES.map(([k, label]) => (
            <button
              key={k}
              type="button"
              aria-pressed={k === range}
              className={`px-2 py-1 ${k === range ? 'bg-gray-100 text-gray-900' : 'text-gray-600 hover:bg-gray-50'}`}
              onClick={() => setRange(k)}
            >
              {label}
            </button>
          ))}
        </div>
      </div>
      {past && (
        <label className="mt-2 inline-flex items-center gap-1.5 text-xs text-gray-700 cursor-pointer">
          <input type="checkbox" className="accent-gray-700" checked={showAfter} onChange={(e) => setShowAfter(e.target.checked)} />
          Show what happened
        </label>
      )}
      {error && <Caution>{error}</Caution>}
      {built?.empty && <p className="mt-2 text-sm text-gray-500">{noRowsNote(history.rows, forecast, HISTORY_SINCE)}</p>}
      {shown && (
        <div className={`mt-2 ${loading || stale ? 'opacity-60' : ''}`}>
          <Plot
            data={shown.traces}
            layout={shown.layout}
            config={PLOT_CONFIG}
            useResizeHandler
            style={{ width: '100%', height: 400 }}
          />
        </div>
      )}
      {!shown && !error && <p className="mt-2 text-sm text-gray-500">Loading</p>}
    </Panel>
  )
}
