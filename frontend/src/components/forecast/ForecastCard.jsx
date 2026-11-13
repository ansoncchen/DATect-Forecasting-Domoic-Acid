import React from 'react'
import Panel from '../ui/Panel'
import Caution from '../ui/Caution'
import CategoryScaleBar from './CategoryScaleBar'
import { formatValue } from '../../lib/format'
import { categoryOf, reachesThreshold, siteReliability } from '../../lib/risk'
import { weakSiteText } from '../../lib/copy'

/**
 * @param {{ forecast: import('../../services/contract').Forecast,
 *   summary: import('../../services/contract').Summary | null,
 *   config: import('../../services/contract').Config }} props
 */
export default function ForecastCard({ forecast, summary, config }) {
  const { site, value, band, bandReason, error } = forecast
  const threshold = config.spikeThreshold
  const rel = summary?.reliability?.holdout?.perSite[site]
  const weak = !error && siteReliability(rel) === 'weak'
  const category = value == null ? null : categoryOf(value, config.categories)

  return (
    <Panel className="p-5">
      <h2 className="text-sm font-semibold text-gray-900">How much toxin next week?</h2>
      {error ? (
        <Caution>{error}</Caution>
      ) : (
        <>
          <div className="mt-2 flex items-baseline gap-2 tabular-nums">
            <span className="text-4xl font-semibold">{formatValue(value)}</span>
            <span className="text-gray-500">µg/g</span>
            {category && <span className="ml-1 text-base font-medium text-gray-700">{category.label}</span>}
          </div>
          <p className="mt-1 text-sm text-gray-700 tabular-nums">
            {band ? `Likely between ${formatValue(band.lo)} and ${formatValue(band.hi)} µg/g.` : `No likely range: ${bandReason}`}
          </p>
          <p className="mt-1 text-sm text-gray-600">
            {reachesThreshold(band, threshold)
              ? `The likely range reaches ${threshold} µg/g, the level at which harvests close.`
              : `Harvests close at ${threshold} µg/g.`}
          </p>
          <CategoryScaleBar value={value} band={band} config={config} />
          {weak && <Caution>{weakSiteText(site, rel, config.windows.holdout.label)}</Caution>}
        </>
      )}
    </Panel>
  )
}
