import React, { useEffect, useRef, useState } from 'react'
import Plot from '../utils/Plot'
import api from '../services/api'
import { plotConfig, getPlotFilename } from '../utils/plotConfig'
import { useConfig } from '../context/ConfigContext'
import Panel from '../components/ui/Panel'
import Caution from '../components/ui/Caution'

const VISUALIZATIONS = [
  ['correlation', 'Correlation heatmap'],
  ['sensitivity', 'Sensitivity analysis'],
  ['comparison', 'DA vs Pseudo-nitzschia'],
  ['waterfall', 'Waterfall plot'],
  ['spectral', 'Spectral analysis'],
  ['map', 'Site map'],
]

// Which visualizations take a site scope; the rest are fixed to one site (comparison) or all sites.
const SCOPED = ['correlation', 'sensitivity', 'spectral']
const SINGLE_ONLY = 'comparison'
const ALL_ONLY = ['waterfall', 'map']

/** @returns {string} the API path for this choice */
function endpointFor(type, scope, site) {
  if (ALL_ONLY.includes(type)) return `/api/visualizations/${type}`
  if (type === SINGLE_ONLY || scope === 'single') return `/api/visualizations/${type}/${site}`
  return `/api/visualizations/${type}/all`
}

const selectClass = 'px-2 py-1.5 border border-gray-300 rounded text-sm min-w-[14rem] max-w-full bg-white'

export default function DataExplorer() {
  const { config, error } = useConfig()
  if (error) return <Caution>{error}</Caution>
  if (!config) return <p className="text-sm text-gray-500">Loading</p>
  return <DataExplorerPage sites={config.sites} />
}

function DataExplorerPage({ sites }) {
  const [type, setType] = useState('correlation')
  const [scope, setScope] = useState('single')
  const [site, setSite] = useState(sites[0])
  const [figure, setFigure] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const latest = useRef(0)

  const forceSingle = type === SINGLE_ONLY
  const showScope = SCOPED.includes(type)
  const showSite = !ALL_ONLY.includes(type) && (forceSingle || scope === 'single')
  const effectiveScope = forceSingle ? 'single' : scope

  useEffect(() => {
    const endpoint = endpointFor(type, effectiveScope, site)
    const id = ++latest.current
    setLoading(true)
    setError(null)
    api
      .get(endpoint)
      .then((r) => latest.current === id && setFigure(r.data))
      .catch((e) => {
        if (latest.current !== id) return
        const detail = e.response?.data?.detail
        setFigure(null)
        setError(typeof detail === 'string' ? detail : e.message || 'Request failed')
      })
      .finally(() => latest.current === id && setLoading(false))
  }, [type, effectiveScope, site])

  const filenameBase = `${type}_${ALL_ONLY.includes(type) || effectiveScope === 'all' ? 'all-sites' : site}`
  const plotProps = (suffix) => ({
    config: {
      ...plotConfig,
      toImageButtonOptions: {
        ...plotConfig.toImageButtonOptions,
        filename: getPlotFilename(suffix ? `${type}_plot${suffix}` : filenameBase),
      },
    },
    useResizeHandler: true,
    style: { width: '100%' },
  })

  const renderFigure = () => {
    if (error) return <Caution>{error}</Caution>
    if (!figure) return <p className="text-sm text-gray-500">Loading</p>
    if (figure.plot) {
      return (
        <div className="overflow-x-auto">
          <Plot data={figure.plot.data} layout={figure.plot.layout} {...plotProps()} />
        </div>
      )
    }
    if (Array.isArray(figure.plots)) {
      return (
        <div className="space-y-4">
          {figure.plots.map((p, i) => (
            <div key={i} className="overflow-x-auto">
              <Plot data={p.data} layout={p.layout} {...plotProps(i + 1)} />
            </div>
          ))}
        </div>
      )
    }
    return <p className="text-sm text-gray-500">No visualization data available.</p>
  }

  return (
    <div className="space-y-5">
      <h1 className="text-xl font-semibold">Data explorer</h1>

      <Panel className="px-4 py-3 space-y-3">
        <div>
          <div className="text-xs text-gray-500 mb-1" id="viz-label">Visualization</div>
          <div
            role="group"
            aria-labelledby="viz-label"
            className="flex flex-wrap gap-1 text-xs"
          >
            {VISUALIZATIONS.map(([k, label]) => (
              <button
                key={k}
                type="button"
                aria-pressed={k === type}
                className={`px-2 py-1 border border-gray-300 rounded ${
                  k === type ? 'bg-gray-100 text-gray-900' : 'text-gray-600 hover:bg-gray-50'
                }`}
                onClick={() => setType(k)}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        {(showScope || showSite) && (
          <div className="flex flex-wrap items-end gap-4">
            {showScope && (
              <div>
                <label htmlFor="scope" className="block text-xs text-gray-500 mb-1">Scope</label>
                <select id="scope" value={scope} onChange={(e) => setScope(e.target.value)} className={selectClass}>
                  <option value="single">Single site</option>
                  <option value="all">All sites</option>
                </select>
              </div>
            )}
            {showSite && (
              <div>
                <label htmlFor="site" className="block text-xs text-gray-500 mb-1">Site</label>
                <select id="site" value={site} onChange={(e) => setSite(e.target.value)} className={selectClass}>
                  {sites.map((s) => (
                    <option key={s} value={s}>{s}</option>
                  ))}
                </select>
              </div>
            )}
          </div>
        )}
      </Panel>

      <Panel className={`p-4 ${loading && figure ? 'opacity-60' : ''}`}>{renderFigure()}</Panel>
    </div>
  )
}
