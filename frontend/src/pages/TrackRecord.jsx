import React, { useEffect, useMemo, useRef, useState } from 'react'
import { useConfig } from '../context/ConfigContext'
import { getSiteHistory, getTrackRecord } from '../services/datect'
import { buildGroundTraces } from '../lib/groundTraces'
import { exportCsv } from '../lib/csv'
import { jumpDefinition } from '../lib/copy'
import { formatPct, formatStat, formatValue } from '../lib/format'
import Panel from '../components/ui/Panel'
import More from '../components/ui/More'
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

function BigStat({ label, value, note }) {
  return (
    <div>
      <dt className="text-sm text-gray-700">{label}</dt>
      <dd className="mt-0.5 text-2xl font-semibold tabular-nums">{value}</dd>
      {note && <dd className="text-xs text-gray-500 mt-0.5">{note}</dd>}
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
        Dots are lab measurements. The blue line is what the model forecast a week before each one, with its likely range
        shaded; hollow dots fell outside that range.
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
      <div>
        <h1 className="text-xl font-semibold text-gray-900">Track record</h1>
        <p className="text-sm text-gray-700 mt-1 max-w-3xl">
          How good were past forecasts? Each lab measurement here was forecast one week ahead, using only data
          available at the time, and then compared with what the lab found.
        </p>
      </div>
      <WindowSelector config={config} windowName={windowName} site={site} onWindow={setWindowName} onSite={setSite} />
      {error && <Caution>{error}</Caution>}
      {loading && !data && !error && <p className="text-sm text-gray-500">Loading</p>}
      {data && (
        <div className={`space-y-5 ${loading ? 'opacity-60' : ''}`}>
          <Panel className="p-5">
            <h2 className="text-base font-semibold">{site || 'All sites'}, {label}</h2>
            <p className="text-xs text-gray-500 mt-0.5">
              {scope ? `${scope.n} lab measurements. ` : ''}
              {shownName === 'validation'
                ? `${label} was used to tune the model, so these numbers flatter it.`
                : `${label} was never used to tune the model.`}
            </p>
            {scope ? (
              <>
                <dl className="mt-4 grid grid-cols-2 lg:grid-cols-4 gap-x-6 gap-y-4">
                  <BigStat
                    label="Typical error"
                    value={scope.mae ? `${formatValue(scope.mae.value)} µg/g` : 'too few'}
                    note="how far off a forecast was, on average"
                  />
                  <BigStat
                    label="Inside the likely range"
                    value={scope.band.nBanded ? formatPct(scope.band.coverage) : 'too few'}
                    note={scope.band.nBanded ? `${scope.band.inside} of ${scope.band.nBanded} measurements (target ${formatPct(config.bandLevel)})` : null}
                  />
                  <BigStat
                    label="Sudden rises caught"
                    value={scope.onsets.n ? `${scope.onsets.caught} of ${scope.onsets.n}` : 'no rises'}
                    note={scope.onsets.n ? `the alert was raised a week before a rise to ${config.spikeThreshold}+` : null}
                  />
                  <BigStat
                    label="Alarms followed by a rise"
                    value={`${scope.alarms.followed} of ${scope.alarms.n}`}
                    note={scope.alarms.n > scope.alarms.followed ? `${scope.alarms.falseNearMiss} of the false alarms came close (10–20 µg/g)` : null}
                  />
                </dl>
                <p className="mt-4 text-xs text-gray-500">{jumpDefinition(config.spikeThreshold)}</p>
                <More summary="Show statistics" className="mt-3">
                  <dl className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-x-6 gap-y-3 text-sm">
                    <Item label="N">{scope.n}</Item>
                    <Item label="R²">{scope.r2 ? formatStat(scope.r2, 2) : <span className={muted}>too few</span>}</Item>
                    <Item label="Log R²">{scope.logR2 ? formatStat(scope.logR2, 2) : <span className={muted}>too few</span>}</Item>
                    <Item label="MAE (µg/g)">{scope.mae ? formatStat(scope.mae, 1) : <span className={muted}>too few</span>}</Item>
                    <Item label={`Range held (${formatPct(config.bandLevel)})`}><RangeHeld reliability={scope} /></Item>
                  </dl>
                  <p className="mt-2 text-xs text-gray-500">
                    Brackets are 95% bootstrap intervals. R² is the share of the ups and downs the forecast explains (1 is
                    perfect); log R² is the same on a log scale, so low values count as much as spikes.
                  </p>
                </More>
              </>
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
                  Each point is one measurement. A point on the gray line was forecast exactly; points above it were
                  over-forecast, points below it under-forecast. The dotted red lines mark {config.spikeThreshold} µg/g.
                </p>
                <PredictedVsActualChart rows={rows} config={config} />
              </>
            )}
          </Panel>

          <Panel className="p-5">
            <h2 className="text-base font-semibold">Compared with simpler methods, {site || 'all sites'}</h2>
            <p className="text-sm text-gray-600 mt-0.5 max-w-3xl">
              The simplest forecast is to repeat the last measurement. A useful model should beat it on the same
              measurements.
            </p>
            <ModelComparison models={scope?.r2 ? scope.models : null} />
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
