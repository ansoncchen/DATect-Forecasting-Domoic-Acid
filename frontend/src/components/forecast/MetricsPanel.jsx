import React, { useState } from 'react'
import Panel from '../ui/Panel'
import { METRICS, loadMetricSelection, saveMetricSelection } from '../../lib/metricSelection'
import {
  AlertTile, CoverageTile, MaeTile, CatsTile, R2Tile, LineTile, PartsTile, DriversTile, WidthTile, ProvTile,
} from './metricTiles'

const TILES = {
  alert: AlertTile,
  coverage: CoverageTile,
  mae: MaeTile,
  cats: CatsTile,
  r2: R2Tile,
  line: LineTile,
  parts: PartsTile,
  drivers: DriversTile,
  width: WidthTile,
  prov: ProvTile,
}

/**
 * Selectable metric tiles in ranked order; the selection is remembered in `storage`.
 * @param {{ site: string, forecast: import('../../services/contract').Forecast | null,
 *   summary: import('../../services/contract').Summary | null, summaryError: string | null,
 *   config: import('../../services/contract').Config, storage: Storage | null,
 *   loading: boolean }} props
 */
export default function MetricsPanel({ site, forecast, summary, summaryError, config, storage, loading }) {
  const [shown, setShown] = useState(() => loadMetricSelection(storage))
  const current = forecast && forecast.site === site ? forecast : null

  const toggle = (key) => {
    const next = METRICS.map(([k]) => k).filter((k) => (k === key ? !shown.includes(k) : shown.includes(k)))
    setShown(next)
    saveMetricSelection(storage, next)
  }

  return (
    <Panel className={`p-5 ${loading ? 'opacity-60' : ''}`}>
      <h2 className="text-base font-semibold">Metrics, {site}</h2>
      <p className="text-xs text-gray-500 mt-0.5">
        Listed by how much each should move a harvest decision. The selection is remembered on this device.
      </p>
      <div className="flex flex-wrap gap-x-4 gap-y-1 mt-3">
        {METRICS.map(([key, label]) => (
          <label key={key} className="inline-flex items-center gap-1.5 text-xs text-gray-700 cursor-pointer">
            <input type="checkbox" checked={shown.includes(key)} onChange={() => toggle(key)} />
            {label}
          </label>
        ))}
      </div>
      {!summary && (
        <p className="text-xs text-gray-500 mt-3">
          {summaryError ? 'Tiles that need the track record are hidden.' : 'Loading the track record.'}
        </p>
      )}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mt-4">
        {shown.length === 0 && <p className="text-sm text-gray-500">No metrics selected.</p>}
        {METRICS.filter(([k]) => shown.includes(k)).map(([key]) => {
          const Tile = TILES[key]
          return <Tile key={key} site={site} forecast={current} summary={summary} config={config} />
        })}
      </div>
    </Panel>
  )
}
