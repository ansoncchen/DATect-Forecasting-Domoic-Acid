import React from 'react'
import Panel from '../ui/Panel'

/**
 * @param {{ config: import('../../services/contract').Config, windowName: string, site: string,
 *   onWindow: (name: string) => void, onSite: (site: string) => void }} props
 */
export default function WindowSelector({ config, windowName, site, onWindow, onSite }) {
  const select = 'px-2 py-1.5 border border-gray-300 rounded text-sm min-w-[14rem] bg-white'
  return (
    <Panel className="px-4 py-3">
      <div className="flex flex-wrap items-end gap-4">
        <div>
          <label htmlFor="window" className="block text-xs text-gray-500 mb-1">Window</label>
          <select id="window" value={windowName} onChange={(e) => onWindow(e.target.value)} className={select}>
            {['holdout', 'validation'].map((n) => (
              <option key={n} value={n}>
                {config.windows[n].label} ({n})
              </option>
            ))}
          </select>
        </div>
        <div>
          <label htmlFor="tr-site" className="block text-xs text-gray-500 mb-1">Site</label>
          <select id="tr-site" value={site} onChange={(e) => onSite(e.target.value)} className={select}>
            <option value="">All sites</option>
            {config.sites.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </div>
      </div>
    </Panel>
  )
}
