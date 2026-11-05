import React from 'react'
import LevelTag from '../ui/LevelTag'
import { formatDate, formatPct, formatValue } from '../../lib/format'
import { bandVsAlertText, siteRecordText, trackRecordText } from '../../lib/copy'
import { siteReliability } from '../../lib/risk'
import { featureLabel } from '../../lib/featureNames'

const GRAY_RAMP = ['#e5e7eb', '#9ca3af', '#4b5563', '#111827']
const TOP_DRIVERS = 5

/** All tiles take { site, forecast, summary, config }; a tile returns null when its inputs have not arrived. */

function Tile({ title, help, foot, children }) {
  return (
    <div className="border border-gray-200 rounded p-4">
      <div className="text-xs text-gray-500" title={help}>
        {title}
        {help && <span className="ml-1 cursor-help underline decoration-dotted">?</span>}
      </div>
      <div className="mt-1">{children}</div>
      {foot && <div className="mt-2 text-[11px] leading-snug text-gray-500">{foot}</div>}
    </div>
  )
}

const Big = ({ children, unit }) => (
  <span className="tabular-nums">
    <span className="text-2xl font-semibold">{children}</span>
    {unit && <span className="text-sm text-gray-500 ml-1">{unit}</span>}
  </span>
)

const Few = ({ label, n }) => <span className="text-sm text-gray-500">Too few {label} measurements ({n})</span>

const dash = (v, fmt) => (v == null ? '—' : fmt(v))
const two = (v) => v.toFixed(2)

function ciText(stat, fmt, sep) {
  return stat.ci ? `95% CI ${fmt(stat.ci[0])}${sep}${fmt(stat.ci[1])}` : null
}

function holdoutContext(site, summary, config) {
  const label = config.windows.holdout.label
  const reliability = summary.reliability.holdout.perSite[site]
  return {
    label,
    headline: summary.headline.perSite[site],
    all: summary.headline.overall,
    reliability,
    n: reliability?.n ?? 0,
  }
}

export function AlertTile({ site, summary, config }) {
  if (!summary) return null
  const holdout = summary.reliability.holdout
  const validation = summary.reliability.validation
  const row = holdout.perSite[site]
  const onsets = row?.onsets
  const record = (r, name) => {
    const l = r?.overall.levels[name]
    return l ? `${l.followed} / ${l.weeks}` : '—'
  }
  return (
    <Tile
      title="Alert track record"
      help={`A jump: the last value known when the forecast was made was below ${config.spikeThreshold} and the measurement reached ${config.spikeThreshold}. Weeks already at or above ${config.spikeThreshold} are left out.`}
      foot={
        <>
          {trackRecordText(holdout.overall, config.windows.holdout.label)} Weeks at each level followed by a jump, all
          sites. Per-site counts are small; read them as counts, not rates.
        </>
      }
    >
      {row ? (
        <>
          {onsets.n > 0 && <Big>{`${onsets.caught} of ${onsets.n}`}</Big>}
          <div className="text-sm text-gray-600">
            {config.windows.holdout.label}. {siteRecordText(site, onsets)}
          </div>
        </>
      ) : (
        <Few label={config.windows.holdout.label} n={0} />
      )}
      <table className="mt-2 text-xs tabular-nums">
        <thead className="text-gray-500">
          <tr>
            <th className="text-left pr-3 font-normal">Level</th>
            <th className="text-left pr-3 font-normal">{config.windows.holdout.label}</th>
            <th className="text-left font-normal">{config.windows.validation.label}</th>
          </tr>
        </thead>
        <tbody>
          {config.levels.map((l) => (
            <tr key={l.name}>
              <td className="pr-3">
                <LevelTag level={l.name.toLowerCase()} config={config} />
              </td>
              <td className="pr-3">{record(holdout, l.name)}</td>
              <td className="text-gray-500">{record(validation, l.name)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Tile>
  )
}

export function CoverageTile({ site, summary, config }) {
  if (!summary) return null
  const label = config.windows.holdout.label
  const reliability = summary.reliability.holdout.perSite[site]
  const overall = summary.reliability.holdout.overall
  const validation = summary.reliability.validation.perSite[site]
  const few = siteReliability(reliability) === 'too-few'
  const band = reliability?.band
  return (
    <Tile
      title="Range reliability"
      help="The range is built from the model's past errors pooled across sites, so it is too wide at some sites and too narrow at others."
      foot={
        <>
          Target {formatPct(config.bandLevel)}. All sites {dash(overall.band.coverage, formatPct)}; this site in{' '}
          {config.windows.validation.label}: {dash(validation?.band.coverage, formatPct)}. At jumps to ≥{' '}
          {config.spikeThreshold} the range held only {dash(overall.band.onsetCoverage, formatPct)}: it describes
          typical weeks; the alert covers jumps.
        </>
      }
    >
      {few ? (
        <Few label={label} n={reliability?.n ?? 0} />
      ) : (
        <>
          <Big>{formatPct(band.coverage)}</Big>
          <div className="text-sm text-gray-600">
            of {band.nBanded} measurements inside the {formatPct(config.bandLevel)} range, {label}
          </div>
        </>
      )}
    </Tile>
  )
}

export function MaeTile({ site, summary, config }) {
  if (!summary) return null
  const { label, headline, all, n } = holdoutContext(site, summary, config)
  const mae = headline?.mae
  return (
    <Tile
      title="Typical error"
      foot={
        headline && `Mean absolute error, ${label}, ${headline.n} measurements. All sites ${dash(all.mae?.value, formatValue)}.`
      }
    >
      {mae ? (
        <>
          <Big unit="µg/g">{formatValue(mae.value)}</Big>
          <div className="text-sm text-gray-600 tabular-nums">{ciText(mae, formatValue, '–')}</div>
        </>
      ) : (
        <Few label={label} n={n} />
      )}
    </Tile>
  )
}

export function CatsTile({ forecast, config }) {
  if (!forecast) return null
  const { band, alert, categoryChances } = forecast
  const ready = band && alert && categoryChances
  return (
    <Tile
      title="Category chances (from the range)"
      help="Share of past forecast errors that would put next week in each category."
      foot={ready && bandVsAlertText(forecast)}
    >
      {ready ? (
        <>
          <div className="flex h-2.5 gap-0.5">
            {categoryChances.map((p, i) => (
              <div key={config.categories[i].label} style={{ flex: `${p} 1 0`, background: GRAY_RAMP[i] }} />
            ))}
          </div>
          <div className="grid grid-cols-2 gap-x-3 mt-2 text-xs tabular-nums">
            {categoryChances.map((p, i) => (
              <span key={config.categories[i].label}>
                {config.categories[i].label} {formatPct(p)}
              </span>
            ))}
          </div>
        </>
      ) : (
        <span className="text-sm text-gray-500">No range for this forecast</span>
      )}
    </Tile>
  )
}

export function R2Tile({ site, summary, config }) {
  if (!summary) return null
  const { label, headline, all, n } = holdoutContext(site, summary, config)
  const r2 = headline?.r2
  const logR2 = headline?.logR2
  return (
    <Tile
      title="Fit (R²)"
      foot={
        headline && (
          <>
            {label}, all real measurements, {headline.n} at this site. All sites {dash(all.r2?.value, two)}
            {logR2 && `; on a log scale ${two(logR2.value)} here`}. A few large values carry the pooled number.
          </>
        )
      }
    >
      {r2 ? (
        <>
          <Big>{two(r2.value)}</Big>
          <div className="text-sm text-gray-600 tabular-nums">{ciText(r2, two, ' to ')}</div>
        </>
      ) : (
        <Few label={label} n={n} />
      )}
    </Tile>
  )
}

export function LineTile({ site, summary, config }) {
  if (!summary) return null
  const { label, headline, all, n } = holdoutContext(site, summary, config)
  const spike = headline?.spike
  const allSpike = all.spike
  return (
    <Tile
      title="Spike detection by the forecast"
      foot={
        spike && (
          <>
            {label}. Precision {dash(spike.precision, formatPct)}, F2 {dash(spike.f2, two)}. All sites: recall{' '}
            {dash(allSpike.recall, formatPct)}, precision {dash(allSpike.precision, formatPct)}. Counts every reading ≥{' '}
            {config.spikeThreshold}, including ones already high the week before.
          </>
        )
      }
    >
      {headline ? (
        <>
          <Big>{dash(spike.recall, formatPct)}</Big>
          <div className="text-sm text-gray-600">
            of {headline.nSpikes} readings ≥ {config.spikeThreshold} had a forecast ≥ {config.regressionAlert}
          </div>
        </>
      ) : (
        <Few label={label} n={n} />
      )}
    </Tile>
  )
}

const NO_FORECAST = <span className="text-sm text-gray-500">No forecast for this week</span>

export function PartsTile({ forecast }) {
  if (!forecast) return null
  const { parts, error, value } = forecast
  let body = NO_FORECAST
  if (!error && parts) {
    const [wx, wr, wn] = parts.weights
    const rows = [
      ['Ensemble', value, null],
      ['XGBoost', parts.xgb, wx],
      ['Random Forest', parts.rf, wr],
      ['Last value carried forward', parts.naive, wn],
    ]
    body = (
      <div className="text-sm space-y-0.5 tabular-nums">
        {rows.map(([name, v, w]) => (
          <div key={name} className={w == null ? 'font-medium' : ''}>
            {name} {dash(v, formatValue)} µg/g
            {w != null && <span className="text-gray-500"> (weight {formatPct(w)})</span>}
          </div>
        ))}
      </div>
    )
  } else if (!error) {
    body = <span className="text-sm text-gray-500">Not available for this forecast</span>
  }
  return (
    <Tile title="Model parts" foot={body !== NO_FORECAST && 'The forecast blends these three; large gaps between them mean the models disagree.'}>
      {body}
    </Tile>
  )
}

export function DriversTile({ forecast }) {
  if (!forecast) return null
  const { drivers, error } = forecast
  const top = [...drivers].sort((a, b) => b.importance - a.importance).slice(0, TOP_DRIVERS)
  let body = NO_FORECAST
  if (!error) {
    body = top.length ? (
      <ol className="text-xs space-y-0.5 list-decimal list-inside text-gray-700">
        {top.map((d) => (
          <li key={d.feature}>{featureLabel(d.feature)}</li>
        ))}
      </ol>
    ) : (
      <span className="text-sm text-gray-500">Not available for this forecast</span>
    )
  }
  return (
    <Tile title="Top drivers" foot={body !== NO_FORECAST && 'Features the model leaned on most for this forecast. Importance is association, not cause.'}>
      {body}
    </Tile>
  )
}

export function WidthTile({ forecast, summary, config }) {
  if (!forecast || !summary) return null
  const { band } = forecast
  const median = summary.reliability.holdout.overall.band.medianWidth
  return (
    <Tile
      title="Range width"
      foot={
        band &&
        `Median ${formatPct(config.bandLevel)} width in ${config.windows.holdout.label}: ${dash(median, formatValue)} µg/g. The range widens with the forecast (it is built on a log scale).`
      }
    >
      {band ? <Big unit="µg/g">{formatValue(band.hi - band.lo)}</Big> : <span className="text-sm text-gray-500">No range for this forecast</span>}
    </Tile>
  )
}

const SOURCE_WORDS = { precomputed: 'Served from the precomputed store', computed: 'Computed for this request' }

export function ProvTile({ forecast, summary }) {
  if (!forecast) return null
  const { source, modelVersion, trainingSamples } = forecast.provenance
  return (
    <Tile title="Data and model" foot="From the forecast response and the results files behind the track-record numbers.">
      <div className="text-xs space-y-0.5 text-gray-700 tabular-nums">
        <div>Data through {formatDate(forecast.anchorDate)}</div>
        <div>{source ? (SOURCE_WORDS[source] ?? source) : 'Source not reported'}</div>
        <div>Model version {modelVersion ?? 'not reported'}</div>
        <div>Training samples {trainingSamples ?? 'not reported'}</div>
        {summary?.source && (
          <ul className="pt-1 text-gray-500 break-all">
            {Object.entries(summary.source).map(([name, path]) => (
              <li key={name}>
                {name}: {String(path)}
              </li>
            ))}
          </ul>
        )}
      </div>
    </Tile>
  )
}
