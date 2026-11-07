import React from 'react'
import Panel from '../ui/Panel'
import Caution from '../ui/Caution'
import CategoryScaleBar from './CategoryScaleBar'
import { formatValue } from '../../lib/format'
import { categoryOf, reachesThreshold, siteReliability } from '../../lib/risk'
import { reliabilityText } from '../../lib/copy'

const catText = (c) => (c.hi === Infinity ? `${c.label} (≥ ${c.lo})` : `${c.label} (${c.lo}–${c.hi})`)

/**
 * @param {{ forecast: import('../../services/contract').Forecast,
 *   summary: import('../../services/contract').Summary | null,
 *   config: import('../../services/contract').Config }} props
 */
export default function ForecastCard({ forecast, summary, config }) {
  const { site, value, band, bandReason, error } = forecast
  const holdout = summary?.reliability?.holdout
  let reliability = null
  if (holdout && !error) {
    const rel = holdout.perSite[site]
    const text = reliabilityText(site, rel, holdout.overall, config.windows.holdout.label)
    reliability =
      siteReliability(rel) === 'weak' ? (
        <Caution>{text}</Caution>
      ) : (
        <p className="text-xs text-gray-500 mt-3">{text}</p>
      )
  }

  return (
    <Panel className="p-5">
      <div className="text-xs text-gray-500">Next week&apos;s forecast, {site}</div>
      {error ? (
        <Caution>{error}</Caution>
      ) : (
        <>
          <div className="mt-1 flex items-baseline gap-1.5 tabular-nums">
            <span className="text-4xl font-semibold">{formatValue(value)}</span>
            <span className="text-gray-500">µg/g</span>
          </div>
          <div className="text-sm text-gray-700 tabular-nums">
            {band
              ? `${Math.round(config.bandLevel * 100)}% range ${formatValue(band.lo)}–${formatValue(band.hi)} µg/g`
              : `No range: ${bandReason}`}
          </div>
          <div className="text-sm text-gray-600 mt-1">
            In the {catText(categoryOf(value, config.categories))} category
            {reachesThreshold(band, config.spikeThreshold)
              ? `; the range reaches the ${config.spikeThreshold} µg/g closure level`
              : ''}
            .
          </div>
          <CategoryScaleBar value={value} band={band} config={config} />
          {reliability}
        </>
      )}
    </Panel>
  )
}
