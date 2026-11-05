import React from 'react'
import { format, parseISO, startOfWeek } from 'date-fns'
import Panel from '../ui/Panel'
import { formatDate } from '../../lib/format'
import { siteReliability } from '../../lib/risk'

const SUFFIX = { weak: ' (less reliable)', 'too-few': ' (little recent data)' }

/**
 * @param {{ config: import('../../services/contract').Config,
 *   summary: import('../../services/contract').Summary | null,
 *   site: string, week: string, anchorDate: string | null,
 *   onSite: (site: string) => void, onWeek: (week: string) => void }} props
 */
export default function FilterBar({ config, summary, site, week, anchorDate, onSite, onWeek }) {
  const holdout = summary?.reliability?.holdout

  const pickWeek = (e) => {
    if (!e.target.value) return
    onWeek(format(startOfWeek(parseISO(e.target.value), { weekStartsOn: 1 }), 'yyyy-MM-dd'))
  }

  return (
    <Panel className="px-4 py-3">
      <div className="flex flex-wrap items-end gap-4">
        <div>
          <label htmlFor="site" className="block text-xs text-gray-500 mb-1">Site</label>
          <select
            id="site"
            value={site}
            onChange={(e) => onSite(e.target.value)}
            className="px-2 py-1.5 border border-gray-300 rounded text-sm min-w-[14rem] bg-white"
          >
            {config.sites.map((s) => (
              <option key={s} value={s}>
                {s}
                {holdout ? SUFFIX[siteReliability(holdout.perSite[s])] ?? '' : ''}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label htmlFor="week" className="block text-xs text-gray-500 mb-1">Forecast week</label>
          <input
            id="week"
            type="date"
            value={week}
            min={config.dateRange.min}
            max={config.latestWeek}
            onChange={pickWeek}
            className="px-2 py-1.5 border border-gray-300 rounded text-sm bg-white"
          />
        </div>
        {anchorDate && (
          <div className="text-sm text-gray-600 pb-1.5">Uses data through {formatDate(anchorDate)}</div>
        )}
        <div className="ml-auto text-xs text-gray-500 pb-1.5">Model: ensemble (XGBoost + Random Forest)</div>
      </div>
    </Panel>
  )
}
