import React from 'react'
import Panel from '../ui/Panel'
import LevelTag from '../ui/LevelTag'
import More from '../ui/More'
import { INK } from '../../lib/colors'
import { formatScore, formatValue } from '../../lib/format'
import { levelKey } from '../../lib/risk'
import {
  alertPlain, classifierConditionText, jumpDefinition, levelCutoffsText, levelRecordPlain, recentValuesText,
  scoreCaption, siteRecordText, trackRecordText,
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
 * The rise alert's level in words, with how it works one click away.
 * @param {{ forecast: import('../../services/contract').Forecast,
 *   summary: import('../../services/contract').Summary | null,
 *   config: import('../../services/contract').Config }} props
 */
export default function SpikeRiskCard({ forecast, summary, config }) {
  const { site, alert, recentMeasurements } = forecast
  const threshold = config.spikeThreshold
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
      <h2 className="text-sm font-semibold text-gray-900">Any sign of a sudden rise?</h2>
      <div className="mt-2 text-3xl font-semibold">
        <LevelTag level={level} config={config} />
      </div>
      <p className="mt-1 text-sm text-gray-700">{alertPlain(level, threshold)}</p>
      {record && (
        <p className="mt-2 text-sm text-gray-600">{levelRecordPlain(levelName, record, holdoutLabel, threshold)}</p>
      )}
      {(level === 'watch' || level === 'warning') && forecast.value != null && forecast.value < threshold && (
        <p className="mt-2 text-xs text-gray-500">
          The forecast is the most likely level. This alert looks separately for a sudden jump, which the forecast tends
          to miss, so the two can disagree.
        </p>
      )}

      {available && !already && (
        <More summary="How this alert works" className="mt-4 pt-3 border-t border-gray-200">
          <div className="space-y-3 text-sm">
            <p className="text-gray-700">
              The alert has three levels. It moves up from Quiet when a separate model, trained only to spot sudden
              rises, gives a high score, or when the forecast itself reaches {config.regressionAlert} µg/g.
            </p>
            {score != null && (
              <div>
                <div className="tabular-nums">
                  <span className="text-xl font-semibold">{formatScore(score)}</span>{' '}
                  <span className="text-gray-600">{scoreCaption(threshold)}</span>
                </div>
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
                <p className="text-[11px] text-gray-500 tabular-nums">{levelCutoffsText(config.levels)}</p>
              </div>
            )}
            <ul className="space-y-0.5 tabular-nums">
              <Condition holds={alert.byClassifier}>
                {classifierConditionText(score, config.classifierCutoff, alert.byClassifier)}
              </Condition>
              <Condition holds={alert.byRegression}>
                Forecast {formatValue(forecast.value)} {alert.byRegression ? '≥' : '<'} {config.regressionAlert} µg/g
              </Condition>
            </ul>
            {recent.length > 0 && (
              <p className="text-xs text-gray-500 tabular-nums">{recentValuesText(recent.map((r) => r.value))}</p>
            )}
            <div className="text-xs text-gray-600 space-y-1">
              <p>{jumpDefinition(threshold)}</p>
              {holdout && <p>{trackRecordText(holdout.overall, holdoutLabel)}</p>}
              {holdout?.perSite[site] && <p>{siteRecordText(site, holdout.perSite[site].onsets)}</p>}
              {placeholders.map((l) => (
                <p key={l.name} className="text-gray-400">
                  The {l.name} cutoff is a placeholder until the alert experiments set the levels.
                </p>
              ))}
            </div>
          </div>
        </More>
      )}
    </Panel>
  )
}
