import React, { useEffect, useRef, useState } from 'react'
import Panel from '../ui/Panel'
import { formatDate } from '../../lib/format'
import { siteReliability } from '../../lib/risk'
import { resolveWeek, weekRangeNote } from '../../lib/weekField'

const SUFFIX = { weak: ' (less reliable)', 'too-few': ' (little recent data)' }
const COMMIT_PAUSE_MS = 800

/**
 * @param {{ config: import('../../services/contract').Config,
 *   summary: import('../../services/contract').Summary | null,
 *   site: string, week: string, anchorDate: string | null,
 *   onSite: (site: string) => void, onWeek: (week: string) => void }} props
 */
export default function FilterBar({ config, summary, site, week, anchorDate, onSite, onWeek }) {
  const holdout = summary?.reliability?.holdout
  const weekRange = { min: config.dateRange.min, max: config.latestWeek }
  const [draft, setDraft] = useState(week)
  const [rejected, setRejected] = useState(false)
  const weekInput = useRef(null)
  const pause = useRef(null)

  useEffect(() => () => clearTimeout(pause.current), [])

  // Typing passes through years like 0002, so the week only changes at commit, and the field,
  // which someone may still be typing in, is only rewritten once it has lost focus.
  const commitWeek = (value, hasFocus) => {
    clearTimeout(pause.current)
    const next = resolveWeek(value, weekRange)
    setRejected(next === null)
    if (next && next !== week) onWeek(next)
    if (!hasFocus) setDraft(next ?? week)
  }

  const editWeek = (e) => {
    const { value } = e.target
    setDraft(value)
    setRejected(false)
    clearTimeout(pause.current)
    pause.current = setTimeout(() => commitWeek(value, document.activeElement === weekInput.current), COMMIT_PAUSE_MS)
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
            ref={weekInput}
            value={draft}
            min={weekRange.min}
            max={weekRange.max}
            onChange={editWeek}
            onBlur={() => commitWeek(draft, false)}
            onKeyDown={(e) => e.key === 'Enter' && commitWeek(draft, true)}
            className="px-2 py-1.5 border border-gray-300 rounded text-sm bg-white"
          />
        </div>
        {anchorDate && (
          <div className="text-sm text-gray-600 pb-1.5">Uses data through {formatDate(anchorDate)}</div>
        )}
        <div className="ml-auto text-xs text-gray-500 pb-1.5">Model: ensemble (XGBoost + Random Forest)</div>
      </div>
      {rejected && <p className="mt-2 text-xs text-gray-500">{weekRangeNote(weekRange)}</p>}
    </Panel>
  )
}
