import React, { useEffect, useMemo, useRef, useState } from 'react'
import Plot from '../../utils/Plot'
import Panel from '../ui/Panel'
import Caution from '../ui/Caution'
import { getSiteHistory } from '../../services/datect'
import { buildGroundTraces } from '../../lib/groundTraces'

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
  const [rows, setRows] = useState(null)
  const [historyLoading, setHistoryLoading] = useState(true)
  const [error, setError] = useState(null)
  const [range, setRange] = useState('2y')
  const [showAfter, setShowAfter] = useState(false)
  const latest = useRef(0)

  useEffect(() => {
    const id = ++latest.current
    setHistoryLoading(true)
    setError(null)
    getSiteHistory(site)
      .then((r) => latest.current === id && (setRows(r), setHistoryLoading(false)))
      .catch((e) => latest.current === id && (setRows(null), setError(e?.message || String(e)), setHistoryLoading(false)))
  }, [site])

  useEffect(() => {
    setShowAfter(false)
  }, [site, week])

  const chart = useMemo(
    () => (rows && forecast ? buildGroundTraces(rows, forecast, config, { range, showAfter }) : null),
    [rows, forecast, config, range, showAfter],
  )

  if (!forecast) return null
  const { holdout, validation } = config.windows
  const past = forecast.date < config.latestWeek

  return (
    <Panel className="p-5">
      <div className="flex flex-wrap items-start gap-3">
        <div>
          <h2 className="text-base font-semibold">Measurements and forecasts, {forecast.site}</h2>
          <p className="text-sm text-gray-600 max-w-3xl mt-0.5">
            Dots are lab measurements; hollow dots fell outside the {Math.round(config.bandLevel * 100)}% range. The line is
            the forecast made a week before each measurement, shaded by its range. Ticks below mark Watch and Warning weeks.
            The models were tuned on {validation.label}; {holdout.label} is the untouched holdout.
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
      {chart && (
        <div className={`mt-2 ${loading || historyLoading ? 'opacity-60' : ''}`}>
          <Plot
            data={chart.traces}
            layout={chart.layout}
            config={PLOT_CONFIG}
            useResizeHandler
            style={{ width: '100%', height: 400 }}
          />
        </div>
      )}
      {!chart && !error && <p className="mt-2 text-sm text-gray-500">Loading</p>}
    </Panel>
  )
}
