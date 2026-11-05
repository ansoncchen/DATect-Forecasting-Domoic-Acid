import React from 'react'
import Panel from '../ui/Panel'
import LevelTag from '../ui/LevelTag'
import { INK } from '../../lib/colors'
import { formatDate, formatPct, formatValue } from '../../lib/format'
import { levelKey } from '../../lib/risk'
import { levelRecordText, siteRecordText, trackRecordText } from '../../lib/copy'

function Condition({ holds, children }) {
  return (
    <li className="flex gap-2">
      <span className={holds ? 'text-gray-900' : 'text-gray-400'}>{holds ? '●' : '○'}</span>
      <span className={holds ? '' : 'text-gray-500'}>{children}</span>
    </li>
  )
}

function Tick({ cutoff, name }) {
  const left = `${100 * cutoff}%`
  return (
    <>
      <div className="absolute top-0 h-2.5 w-px bg-gray-500" style={{ left }} />
      <span className="absolute top-3 -translate-x-1/2 text-[10px] text-gray-500 whitespace-nowrap" style={{ left }}>
        {formatPct(cutoff)} {name}
      </span>
    </>
  )
}

/**
 * Graded spike-risk level beside the alert's track record.
 * @param {{ forecast: import('../../services/contract').Forecast,
 *   summary: import('../../services/contract').Summary | null,
 *   config: import('../../services/contract').Config }} props
 */
export default function SpikeRiskCard({ forecast, summary, config }) {
  const { site, alert, lastMeasurement, recentMeasurements } = forecast
  const level = levelKey(alert)
  const available = level !== 'unavailable'
  const already = level === 'already'
  const probability = alert?.probability
  const marker = Math.min(1, Math.max(0, probability ?? 0))
  const holdout = summary?.reliability?.holdout
  const holdoutLabel = config.windows.holdout.label
  const levelName = config.levels.find((l) => l.name.toLowerCase() === level)?.name
  const record = levelName && holdout?.overall.levels[levelName]
  const recent = (recentMeasurements ?? []).slice(-5)
  const placeholders = config.levels.filter((l) => l.placeholder)

  return (
    <Panel className="p-5">
      <div className="flex items-center justify-between">
        <div className="text-xs text-gray-500">Spike risk</div>
        <div className="text-sm font-medium">
          <LevelTag level={level} config={config} />
        </div>
      </div>
      <div className="mt-1 flex items-baseline gap-2 tabular-nums">
        <span className="text-4xl font-semibold">{available ? formatPct(probability) : '—'}</span>
        <span className="text-sm text-gray-600 leading-tight">
          {available
            ? `chance of a jump to ≥ ${config.spikeThreshold} µg/g next week`
            : 'Spike classifier unavailable for this week'}
        </span>
      </div>
      {available && (
        <div className="relative mt-3 mb-6 h-4">
          <div className="absolute top-1 left-0 right-0 h-0.5 bg-gray-200" />
          {config.levels
            .filter((l) => l.cutoff != null)
            .map((l) => (
              <Tick key={l.name} cutoff={l.cutoff} name={l.name} />
            ))}
          <div className="absolute -top-0.5 w-1 h-3.5 -ml-0.5" style={{ left: `${100 * marker}%`, background: INK }} />
        </div>
      )}
      {already && lastMeasurement && (
        <p className="mt-2 text-sm text-gray-700">
          The last measurement ({formatValue(lastMeasurement.value)} on {formatDate(lastMeasurement.date)}) is already at
          or above {config.spikeThreshold}. This alert is for jumps from below {config.spikeThreshold}; follow the
          forecast and the measurements.
        </p>
      )}
      {available && !already && (
        <>
          <ul className="mt-1 text-sm space-y-0.5 tabular-nums">
            <Condition holds={alert.byClassifier}>
              Classifier {formatPct(probability)} {alert.byClassifier ? '≥' : '<'} {formatPct(config.classifierCutoff)}
            </Condition>
            <Condition holds={alert.byRegression}>
              Forecast {formatValue(forecast.value)} {alert.byRegression ? '≥' : '<'} {config.regressionAlert} µg/g
            </Condition>
          </ul>
          {record && (
            <p className="mt-2 text-sm text-gray-700">{levelRecordText(levelName, record, holdoutLabel)}</p>
          )}
        </>
      )}
      {recent.length > 0 && (
        <p className="mt-1 text-xs text-gray-500 tabular-nums">
          Measured lately: {recent.map((r) => formatValue(r.value)).join(', ')} µg/g
        </p>
      )}
      {(holdout || placeholders.length > 0) && (
        <div className="mt-3 pt-3 border-t border-gray-200 text-xs text-gray-600 space-y-1">
          {holdout && <p>{trackRecordText(holdout.overall, holdoutLabel)}</p>}
          {holdout?.perSite[site] && <p>{siteRecordText(site, holdout.perSite[site].onsets)}</p>}
          {placeholders.map((l) => (
            <p key={l.name} className="text-gray-400">
              The {l.name} cutoff is a placeholder until the alert experiments set the levels.
            </p>
          ))}
        </div>
      )}
    </Panel>
  )
}
