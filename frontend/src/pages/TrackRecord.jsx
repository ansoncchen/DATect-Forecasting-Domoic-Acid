import React, { useEffect, useMemo, useRef, useState } from 'react'
import { useConfig } from '../context/ConfigContext'
import { getSiteHistory, getTrackRecord } from '../services/datect'
import { buildGroundTraces } from '../lib/groundTraces'
import { exportCsv } from '../lib/csv'
import { formatPct, formatStat } from '../lib/format'
import Panel from '../components/ui/Panel'
import Caution from '../components/ui/Caution'
import Plot from '../utils/Plot'
import WindowSelector from '../components/trackrecord/WindowSelector'
import PerSiteTable, { RangeHeld } from '../components/trackrecord/PerSiteTable'
import PredictedVsActualChart from '../components/trackrecord/PredictedVsActualChart'
import ModelComparison from '../components/trackrecord/ModelComparison'

const PLOT_CONFIG = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ['lasso2d', 'select2d'] }
const muted = 'text-gray-500'

export default function TrackRecord() {
  const { config, error } = useConfig()
  if (error) return <Caution>{error}</Caution>
  if (!config) return <p className="text-sm text-gray-500">Loading</p>
  return <TrackRecordPage config={config} />
}

function Item({ label, children }) {
  return (
    <div>
      <dt className="text-xs text-gray-500">{label}</dt>
      <dd className="tabular-nums">{children}</dd>
    </div>
  )
}

function SiteHistoryChart({ site, rows, window, config }) {
  const [history, setHistory] = useState(null) // { site, rows }
  const [error, setError] = useState(null)
  const latest = useRef(0)

  useEffect(() => {
    const id = ++latest.current
    setError(null)
    getSiteHistory(site)
      .then((r) => latest.current === id && setHistory({ site, rows: r }))
      .catch((e) => latest.current === id && (setHistory(null), setError(e?.message || String(e))))
  }, [site])

  const built = useMemo(() => {
    if (!history || history.site !== site) return null
    const inWindow = history.rows.filter((r) => r.date >= window.start && r.date < window.end)
    return buildGroundTraces(inWindow, null, config, { range: 'all' })
  }, [history, site, window, config])

  return (
    <>
      <h2 className="text-base font-semibold">Measurements and forecasts, {site}</h2>
      <p className="text-sm text-gray-600 max-w-3xl mt-0.5">
        Dots are lab measurements; hollow dots fell outside the {Math.round(config.bandLevel * 100)}% range. The line is
        the forecast made a week before each measurement, shaded by its range.
      </p>
      {error && <Caution>{error}</Caution>}
      {built && rows.length > 0 && (
        <Plot data={built.traces} layout={built.layout} config={PLOT_CONFIG} useResizeHandler style={{ width: '100%', height: 400 }} />
      )}
      {!built && !error && <p className="mt-2 text-sm text-gray-500">Loading</p>}
    </>
  )
}

function TrackRecordPage({ config }) {
  const [windowName, setWindowName] = useState('holdout')
  const [site, setSite] = useState('')
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const latest = useRef(0)

  useEffect(() => {
    const id = ++latest.current
    setLoading(true)
    setError(null)
    getTrackRecord(windowName)
      .then((r) => latest.current === id && (setData(r), setLoading(false)))
      .catch((e) => latest.current === id && (setData(null), setError(e?.message || String(e)), setLoading(false)))
    return () => {
      latest.current++
    }
  }, [windowName])

  const shownName = data?.window.name
  const label = data ? config.windows[shownName]?.label ?? data.window.label : null
  const scope = data ? (site ? data.perSite[site] : data.overall) : null
  const rows = useMemo(() => (data ? data.rows.filter((r) => !site || r.site === site) : []), [data, site])

  return (
    <div className="space-y-5">
      <h1 className="text-xl font-semibold text-gray-900">Track record</h1>
      <WindowSelector config={config} windowName={windowName} site={site} onWindow={setWindowName} onSite={setSite} />
      {error && <Caution>{error}</Caution>}
      {loading && !data && !error && <p className="text-sm text-gray-500">Loading</p>}
      {data && (
        <div className={`space-y-5 ${loading ? 'opacity-60' : ''}`}>
          <div>
            <p className="text-sm text-gray-700">
              Every real measurement in {label} (N = {scope ? scope.n : 0}). Each forecast was made a week ahead from data up to its anchor.
            </p>
            <p className="text-xs text-gray-500 mt-1">
              {shownName === 'validation' ? `${label} was used for tuning.` : `${label} was never used for tuning.`}
            </p>
          </div>

          <Panel className="p-5">
            <h2 className="text-base font-semibold">{site || 'All sites'}, {label}</h2>
            {scope ? (
              <dl className="mt-3 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-7 gap-x-6 gap-y-3 text-sm">
                <Item label="N">{scope.n}</Item>
                <Item label="R²">{scope.r2 ? formatStat(scope.r2, 2) : <span className={muted}>too few</span>}</Item>
                <Item label="Log R²">{scope.logR2 ? formatStat(scope.logR2, 2) : <span className={muted}>too few</span>}</Item>
                <Item label="MAE (µg/g)">{scope.mae ? formatStat(scope.mae, 1) : <span className={muted}>too few</span>}</Item>
                <Item label={`Range held (${formatPct(config.bandLevel)})`}><RangeHeld reliability={scope} /></Item>
                <Item label="Onsets caught">{scope.onsets.caught} of {scope.onsets.n}</Item>
                <Item label="Alarms followed by a jump">{scope.alarms.followed} of {scope.alarms.n}</Item>
              </dl>
            ) : (
              <p className="mt-2 text-sm text-gray-500">No measurements at this site in this window.</p>
            )}
          </Panel>

          <Panel className="p-5">
            {site ? (
              <SiteHistoryChart site={site} rows={rows} window={data.window} config={config} />
            ) : (
              <>
                <h2 className="text-base font-semibold">Forecast against measurement, all sites</h2>
                <p className="text-sm text-gray-600 max-w-3xl mt-0.5">
                  Each point is one measurement; the blue bar is its {formatPct(config.bandLevel)} range. Points on the gray
                  line were forecast exactly; dotted red lines mark {config.spikeThreshold} µg/g.
                </p>
                <PredictedVsActualChart rows={rows} config={config} />
              </>
            )}
          </Panel>

          <Panel className="p-5">
            <h2 className="text-base font-semibold">Against simpler models, {site || 'all sites'}</h2>
            <p className="text-xs text-gray-500 mt-0.5">Same measurements, same week-ahead anchors.</p>
            <ModelComparison models={scope?.models ?? null} />
          </Panel>

          <PerSiteTable config={config} perSite={data.perSite} site={site} label={label} onSelectSite={setSite} />

          <div>
            <button
              type="button"
              disabled={rows.length === 0}
              onClick={() => exportCsv(rows, `datect_track_record_${shownName}_${site ? site.replace(/\s+/g, '_') : 'all'}.csv`)}
              className="px-3 py-1.5 border border-gray-300 rounded text-sm bg-white hover:bg-gray-50 disabled:opacity-50"
            >
              Download CSV
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
