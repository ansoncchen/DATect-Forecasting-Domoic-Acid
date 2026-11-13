import React, { useEffect, useRef, useState } from 'react'
import Plot from '../utils/Plot'
import api from '../services/api'
import { plotConfig, getPlotFilename } from '../utils/plotConfig'
import { useConfig } from '../context/ConfigContext'
import { plainSiteMap, readableHeatmap, readableLabels } from '../lib/figures'
import Panel from '../components/ui/Panel'
import Caution from '../components/ui/Caution'

const VISUALIZATIONS = [
  ['correlation', 'What moves together'],
  ['sensitivity', 'What the toxin tracks'],
  ['comparison', 'Toxin and algae over time'],
  ['waterfall', 'Toxin at every site'],
  ['spectral', 'Repeating cycles'],
  ['map', 'Site map'],
]

const DESCRIPTIONS = {
  correlation:
    'Each square compares two measurements over the years. Blue means they tend to rise together, red means one rises as the other falls, white means no clear link. Hover a square for its value.',
  sensitivity:
    'Inputs ranked by how closely they track the toxin level. Tracking is not cause: two things can move together because of a third.',
  comparison:
    'Domoic acid and Pseudo-nitzschia, the algae that produce it, at one site. Both are scaled to 0–1 so you can compare their timing. Weeks between samples are filled in.',
  waterfall:
    'Toxin over time at every site, north at the top. Each line sits at its site\'s latitude, and peaks are toxin events. Weeks between samples are filled in.',
  spectral:
    'Looks for patterns that repeat, such as a yearly bloom season, in the toxin record and in the forecasts.',
  map: 'Where the ten monitoring beaches are. Hover a site for its name.',
}

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

// Server-rendered layouts carry a fixed width; drop it so the figure fills its panel and redraws on resize.
const fitted = (layout) => ({ ...layout, autosize: true, width: undefined })

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
  const [allVariables, setAllVariables] = useState(false)
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
      .then((r) => latest.current === id && setFigure({ ...r.data, type }))
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
      responsive: true,
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
      let plot = figure.plot
      if (figure.type === 'map') plot = plainSiteMap(plot)
      else if (figure.type === 'correlation') plot = readableHeatmap(plot, { all: allVariables })
      else if (figure.type === 'sensitivity') plot = readableLabels(plot)
      return (
        <Plot data={plot.data} layout={fitted(plot.layout)} {...plotProps()} />
      )
    }
    if (Array.isArray(figure.plots)) {
      const readable = figure.type === 'sensitivity' ? readableLabels : (p) => p
      return (
        <div className="space-y-4">
          {figure.plots.map(readable).map((p, i) => (
            <Plot key={i} data={p.data} layout={fitted(p.layout)} {...plotProps(i + 1)} />
          ))}
        </div>
      )
    }
    return <p className="text-sm text-gray-500">No visualization data available.</p>
  }

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl font-semibold">Data explorer</h1>
        <p className="text-sm text-gray-700 mt-1 max-w-3xl">
          Charts of the data behind the forecasts: toxin measurements, algae counts, satellite ocean readings, river flow
          and climate indices.
        </p>
      </div>

      <Panel className="px-4 py-3 space-y-3">
        <div>
          <div className="text-xs text-gray-500 mb-1" id="viz-label">Chart</div>
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

        <p className="text-sm text-gray-600 max-w-3xl">{DESCRIPTIONS[type]}</p>

        {(showScope || showSite || type === 'correlation') && (
          <div className="flex flex-wrap items-end gap-4">
            {showScope && (
              <div>
                <label htmlFor="scope" className="block text-xs text-gray-500 mb-1">Sites</label>
                <select id="scope" value={scope} onChange={(e) => setScope(e.target.value)} className={selectClass}>
                  <option value="single">One site</option>
                  <option value="all">All sites together</option>
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
            {type === 'correlation' && (
              <label className="inline-flex items-center gap-1.5 text-xs text-gray-700 cursor-pointer pb-2">
                <input type="checkbox" checked={allVariables} onChange={(e) => setAllVariables(e.target.checked)} />
                Show every variable (adds the other ocean-anomaly variants)
              </label>
            )}
          </div>
        )}
      </Panel>

      <Panel className={`p-4 ${loading && figure ? 'opacity-60' : ''}`}>{renderFigure()}</Panel>
    </div>
  )
}
