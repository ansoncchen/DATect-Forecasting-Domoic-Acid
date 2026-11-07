import React from 'react'
import Panel from '../ui/Panel'
import LevelTag from '../ui/LevelTag'
import { INK } from '../../lib/colors'
import { formatDate, formatScore, formatValue } from '../../lib/format'
import { levelKey } from '../../lib/risk'
import {
  classifierConditionText, levelCutoffsText, levelRecordText, recentValuesText, scoreCaption, siteRecordText, trackRecordText,
} from '../../lib/copy'

function Condition({ holds, children }) {
  return (
    <li className="flex gap-2">
      <span className={holds ? 'text-gray-900' : 'text-gray-400'}>{holds ? '●' : '○'}</span>
      <span className={holds ? '' : 'text-gray-500'}>{children}</span>
    </li>
  )
}

function Tick({ cutoff }) {
  const left = `${100 * cutoff}%`
  return (
    <>
      <div className="absolute top-0 h-2.5 w-px bg-gray-500" style={{ left }} />
      <span className="absolute top-3 -translate-x-1/2 text-[10px] text-gray-500 whitespace-nowrap" style={{ left }}>
        {formatScore(cutoff)}
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
  const score = available && !already ? alert.probability : null
  const holdout = summary?.reliability?.holdout
  const holdoutLabel = config.windows.holdout.label
  const levelName = config.levels.find((l) => l.name.toLowerCase() === level)?.name
  // A level's record describes weeks the classifier scored, so it does not stand behind a week it did not.
  const record = levelName && score != null ? holdout?.overall.levels[levelName] : null
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
      {!already && (
        <div className="mt-1 tabular-nums">
          <div className="text-4xl font-semibold">{formatScore(score)}</div>
          <p className="mt-1 text-sm text-gray-600 leading-snug">
            {score != null ? scoreCaption(config.spikeThreshold, Boolean(record)) : 'Spike classifier unavailable for this week'}
          </p>
        </div>
      )}
      {score != null && (
        <>
          <div className="relative mt-3 mb-5 h-4">
            <div className="absolute top-1 left-0 right-0 h-0.5 bg-gray-200" />
            {config.levels
              .filter((l) => l.cutoff != null)
              .map((l) => (
                <Tick key={l.name} cutoff={l.cutoff} />
              ))}
            <div
              className="absolute -top-0.5 w-1 h-3.5 -ml-0.5"
              style={{ left: `${100 * Math.min(1, Math.max(0, score))}%`, background: INK }}
            />
          </div>
          <p className="mb-2 text-[11px] text-gray-500 tabular-nums">{levelCutoffsText(config.levels)}</p>
        </>
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
              {classifierConditionText(score, config.classifierCutoff, alert.byClassifier)}
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
          {recentValuesText(recent.map((r) => r.value))}
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
