import React, { useState, useEffect } from 'react'
import { MapPin, BarChart3, Activity, Map } from 'lucide-react'
import Select from 'react-select'
import Plot from 'react-plotly.js'
import api from '../services/api'
import { plotConfig, plotConfigSquare, getPlotFilename } from '../utils/plotConfig'

const Historical = () => {
  const [sites, setSites] = useState([])
  const [selectedSite, setSelectedSite] = useState(null)
  const [visualizationType, setVisualizationType] = useState('correlation')
  const [siteScope, setSiteScope] = useState('single') // 'single' or 'all'
  const [visualizationData, setVisualizationData] = useState(null)
  const [loadingVisualization, setLoadingVisualization] = useState(false)

  const visualizationOptions = [
    { value: 'correlation', label: 'Correlation Heatmap', icon: BarChart3 },
    { value: 'sensitivity', label: 'Sensitivity Analysis', icon: BarChart3 },
    { value: 'comparison', label: 'DA vs Pseudo-nitzschia', icon: Activity },
    { value: 'waterfall', label: 'Waterfall Plot', icon: BarChart3 },
    { value: 'spectral', label: 'Spectral Analysis', icon: Activity },
    { value: 'map', label: 'Site Map', icon: Map },
  ]

  const siteScopeOptions = [
    { value: 'single', label: 'Single Site' },
    { value: 'all', label: 'All Sites' }
  ]

  useEffect(() => {
    loadSites()
  }, [])

  const loadSites = async () => {
    try {
      const response = await api.get('/api/sites')
      const sitesList = response.data.sites
      setSites(sitesList)
      
      if (sitesList.length > 0) {
        setSelectedSite({ value: sitesList[0], label: sitesList[0] })
      }
    } catch (err) {
      console.error('Failed to load sites:', err)
    }
  }


  const loadVisualizationData = async () => {
    setLoadingVisualization(true)
    try {
      const params = new URLSearchParams()

      let endpoint = ''
      if (visualizationType === 'correlation') {
        endpoint = siteScope === 'single' && selectedSite 
          ? `/api/visualizations/correlation/${selectedSite.value}` 
          : '/api/visualizations/correlation/all'
      } else if (visualizationType === 'sensitivity') {
        endpoint = siteScope === 'single' && selectedSite 
          ? `/api/visualizations/sensitivity/${selectedSite.value}` 
          : '/api/visualizations/sensitivity/all'
      } else if (visualizationType === 'comparison') {
        // DA vs Pseudo-nitzschia only supports single site
        endpoint = selectedSite
          ? `/api/visualizations/comparison/${selectedSite.value}`
          : null
      } else if (visualizationType === 'waterfall') {
        endpoint = '/api/visualizations/waterfall'
      } else if (visualizationType === 'spectral') {
        endpoint = siteScope === 'single' && selectedSite
          ? `/api/visualizations/spectral/${selectedSite.value}`
          : '/api/visualizations/spectral/all'
      } else if (visualizationType === 'map') {
        endpoint = '/api/visualizations/map'
      }

      if (endpoint) {
        const response = await api.get(`${endpoint}?${params}`)
        setVisualizationData(response.data)
      }
    } catch (err) {
      console.error('Failed to load visualization data:', err)
      setVisualizationData(null)
    } finally {
      setLoadingVisualization(false)
    }
  }


  useEffect(() => {
    if (forceSingleSite && siteScope === 'all') {
      setSiteScope('single')
    }
    loadVisualizationData()
  }, [visualizationType, selectedSite, siteScope])

  const siteOptions = sites.map(site => ({ value: site, label: site }))

  // Check if current visualization supports site scope selection
  const supportsSiteScope = ['correlation', 'spectral', 'sensitivity'].includes(visualizationType)
  // DA vs Pseudo-nitzschia only supports single site
  const forceSingleSite = visualizationType === 'comparison'
  // Waterfall plot and map are all-sites only; hide site controls
  const hideSiteControls = visualizationType === 'waterfall' || visualizationType === 'map'

  const renderVisualization = () => {
    if (!visualizationData) {
      return <p className="text-center text-gray-500">Loading visualization...</p>
    }

    if (visualizationData.plot) {
      const isHeatmap = visualizationType === 'correlation'
      const config = {
        ...(isHeatmap ? plotConfigSquare : plotConfig),
        toImageButtonOptions: {
          ...(isHeatmap ? plotConfigSquare : plotConfig).toImageButtonOptions,
          filename: getPlotFilename(`${visualizationType}_${siteScope === 'all' ? 'all-sites' : selectedSite?.value || 'plot'}`)
        }
      }
      
      return (
        <div className={isHeatmap ? "flex justify-center" : ""}>
          <Plot
            data={visualizationData.plot.data}
            layout={visualizationData.plot.layout}
            config={config}
            className={isHeatmap ? "" : "w-full"}
            style={isHeatmap ? { maxWidth: '800px' } : {}}
          />
        </div>
      )
    }

    if (visualizationData.plots && Array.isArray(visualizationData.plots)) {
      return (
        <div className="space-y-4">
          {visualizationData.plots.map((plot, index) => (
            <div key={index} className="flex justify-center">
              <Plot
                data={plot.data}
                layout={plot.layout}
                config={{
                  ...plotConfig,
                  toImageButtonOptions: {
                    ...plotConfig.toImageButtonOptions,
                    filename: getPlotFilename(`${visualizationType}_plot${index + 1}`)
                  }
                }}
                className="w-full"
                style={{ maxWidth: '1000px' }}
              />
            </div>
          ))}
        </div>
      )
    }

    return <p className="text-center text-gray-500">No visualization data available.</p>
  }

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      <div className="bg-white rounded-lg shadow-md p-6">
        <h1 className="text-3xl font-bold text-gray-800 mb-2">
          Historical Data Analysis
        </h1>
        <p className="text-gray-600">
          Explore historical domoic acid measurements and advanced visualizations
        </p>
      </div>

      {/* Controls */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold mb-4">Analysis Parameters</h2>
        
        {/* Visualization Type Selector */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Visualization Type
          </label>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-2">
            {visualizationOptions.map(option => (
              <button
                key={option.value}
                onClick={() => setVisualizationType(option.value)}
                className={`px-4 py-2 rounded-lg border transition-colors ${
                  visualizationType === option.value
                    ? 'bg-blue-500 text-white border-blue-500'
                    : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
                }`}
              >
                <div className="flex items-center justify-center space-x-2">
                  <option.icon className="w-4 h-4" />
                  <span className="text-sm">{option.label}</span>
                </div>
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Site Scope Selector */}
          {supportsSiteScope && !hideSiteControls && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                <BarChart3 className="w-4 h-4 inline mr-1" />
                Site Scope
              </label>
              <Select
                value={siteScopeOptions.find(opt => opt.value === siteScope)}
                onChange={(option) => setSiteScope(option.value)}
                options={siteScopeOptions}
                className="text-sm"
              />
            </div>
          )}

          {/* Site Selector - show for single site scope or when forced */}
          {(siteScope === 'single' || forceSingleSite) && !hideSiteControls && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                <MapPin className="w-4 h-4 inline mr-1" />
                Monitoring Site
              </label>
              <Select
                value={selectedSite}
                onChange={setSelectedSite}
                options={siteOptions}
                className="text-sm"
                placeholder="Select site..."
              />
            </div>
          )}
        </div>
      </div>

      {/* Visualization */}
      <div className="bg-white rounded-lg shadow-md p-6">
        {loadingVisualization ? (
          <div className="text-center py-8">
            <p className="text-gray-600">Loading visualization...</p>
          </div>
        ) : (
          renderVisualization()
        )}
      </div>
    </div>
  )
}

export default Historical