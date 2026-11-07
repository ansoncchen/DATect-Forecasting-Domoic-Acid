import React from 'react'
import Panel from '../ui/Panel'
import Caution from '../ui/Caution'
import { RED } from '../../lib/colors'
import { daysBetween, formatAge, formatDate, formatValue } from '../../lib/format'
import { isStale, noMeasurementText } from '../../lib/risk'

/**
 * The last real lab measurements and how old they are.
 * @param {{ forecast: import('../../services/contract').Forecast,
 *   config: import('../../services/contract').Config }} props
 */
export default function LatestMeasurementCard({ forecast, config }) {
  const { lastMeasurement, recentMeasurements, anchorDate } = forecast
  const threshold = config.spikeThreshold

  if (!lastMeasurement) {
    return (
      <Panel className="p-5">
        <div className="text-xs text-gray-500">Latest measurement</div>
        <p className="mt-1 text-sm text-gray-700">{noMeasurementText(forecast)}</p>
      </Panel>
    )
  }

  const age = anchorDate ? daysBetween(lastMeasurement.date, anchorDate) : null
  const ageText = age === null ? null : age === 0 ? 'the day the forecast was made' : `${formatAge(age)} before the forecast was made`
  const rows = [...(recentMeasurements ?? [])].reverse()
  const max = Math.max(...rows.map((r) => r.value), threshold)

  return (
    <Panel className="p-5">
      <div className="text-xs text-gray-500">Latest measurement</div>
      <div className="mt-1 flex items-baseline gap-1.5 tabular-nums">
        <span className="text-4xl font-semibold">{formatValue(lastMeasurement.value)}</span>
        <span className="text-gray-500">µg/g</span>
      </div>
      <div className="text-sm text-gray-700">
        {formatDate(lastMeasurement.date)}
        {ageText && `, ${ageText}`}
      </div>
      {anchorDate && isStale(lastMeasurement.date, anchorDate) && (
        <Caution>
          <span className="font-medium">No recent sample.</span> Without one the forecast leans on ocean conditions and
          this site&apos;s history; give it less weight.
        </Caution>
      )}
      {rows.length > 0 && (
        <>
          <div className="mt-4 text-xs text-gray-500">Last {rows.length} measurements</div>
          <div className="mt-1 space-y-1">
            {rows.map((r) => (
              <div key={r.date} className="flex items-center gap-2 text-xs tabular-nums">
                <span className="w-20 text-gray-500">{formatDate(r.date)}</span>
                <div className="flex-1 h-1.5 bg-gray-100 relative">
                  <div
                    className={`absolute inset-y-0 left-0 ${r.value >= threshold ? 'bg-gray-800' : 'bg-gray-400'}`}
                    style={{ width: `${Math.max((100 * r.value) / max, 0.5)}%` }}
                  />
                  <div className="absolute -inset-y-0.5 w-px" style={{ left: `${(100 * threshold) / max}%`, background: RED }} />
                </div>
                <span className="w-9 text-right">{formatValue(r.value)}</span>
              </div>
            ))}
          </div>
          <div className="mt-1 text-[10px] text-gray-500">
            <span className="inline-block w-px h-2 align-middle mr-1" style={{ background: RED }} />
            {threshold} µg/g
          </div>
        </>
      )}
    </Panel>
  )
}
