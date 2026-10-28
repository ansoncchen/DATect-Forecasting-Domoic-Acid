import React, { useState, useEffect } from 'react'
import { Calendar, MapPin, Cpu, AlertTriangle, Settings, Play, BarChart3, TrendingUp, Clock, Download, Info } from 'lucide-react'
import DatePicker from 'react-datepicker'
import Select from 'react-select'
import Plot from '../utils/Plot'
import { format, subDays } from 'date-fns'
import api from '../services/api'
import { plotConfig, getPlotFilename } from '../utils/plotConfig'
import {
  buildRetrospectiveScatter,
  buildRetrospectiveTimeSeries,
  calculateClassificationMetrics,
  calculateRegressionMetrics,
  exportResultsCSV,
  parsePlotlyJson,
} from '../utils/retrospective'
import 'react-datepicker/dist/react-datepicker.css'

const Dashboard = () => {
  const [currentStep, setCurrentStep] = useState('config')
  
  const [config, setConfig] = useState({
    forecast_mode: 'realtime',
    forecast_task: 'regression',
    forecast_model: 'ensemble',
    selected_sites: []
  })
  const [configLoading, setConfigLoading] = useState(false)
  
  const [sites, setSites] = useState([])
  const [models, setModels] = useState({ regression: [], classification: [] })
  const [dateRange, setDateRange] = useState({ min: null, max: null })
  // Headline holdout metrics from the latest committed evaluation (/api/metrics/summary)
  const [metricsSummary, setMetricsSummary] = useState(null)
  
  const [selectedDate, setSelectedDate] = useState(null)
  const [selectedSite, setSelectedSite] = useState(null)
  const [selectedModel, setSelectedModel] = useState('ensemble')
  const [task, setTask] = useState('regression')
  
  const [forecast, setForecast] = useState(null)
  const [retrospectiveResults, setRetrospectiveResults] = useState(null)
  const [filteredResults, setFilteredResults] = useState(null)
  const [selectedSiteFilter, setSelectedSiteFilter] = useState('all')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    loadInitialData()
    loadConfig()
  }, [])

  const loadInitialData = async () => {
    try {
      const [sitesRes, modelsRes] = await Promise.all([
        api.get('/api/sites'),
        api.get('/api/models')
      ])
      
      setSites(sitesRes.data.sites)
      setModels(modelsRes.data.available_models)
      setDateRange(sitesRes.data.date_range)
      
      if (sitesRes.data.sites.length > 0) {
        setSelectedSite({ value: sitesRes.data.sites[0], label: sitesRes.data.sites[0] })
        setConfig(prev => ({ ...prev, selected_sites: sitesRes.data.sites }))
      }
      
      if (sitesRes.data.date_range.max) {
        setSelectedDate(subDays(new Date(sitesRes.data.date_range.max), 30))
      }
      
    } catch (err) {
      setError('Failed to load initial data')
    }
    try {
      const summaryRes = await api.get('/api/metrics/summary')
      setMetricsSummary(summaryRes.data)
    } catch (err) {
      // Optional: the dashboard works without headline metrics
      console.warn('Metrics summary unavailable:', err.message)
    }
  }

  const loadConfig = async () => {
    try {
      const response = await api.get('/api/config')
      setConfig(response.data)
      setTask(response.data.forecast_task)
      setSelectedModel(response.data.forecast_model)
    } catch (err) {
      console.error('Failed to load config:', err)
    }
  }

  const applyConfig = async () => {
    if (!config.forecast_mode) {
      setError('Please select a forecast mode')
      return
    }
    
    if (config.forecast_mode === 'retrospective' && (!config.forecast_task || !config.forecast_model)) {
      setError('Please select forecast task and model for retrospective analysis')
      return
    }

    setConfigLoading(true)
    setError(null)

    // Settings travel with each request; the server holds no per-user state.
    setTask(config.forecast_task)
    setSelectedModel(config.forecast_model)
    try {
      if (config.forecast_mode === 'realtime') {
        setCurrentStep('realtime')
      } else {
        // Start retrospective analysis immediately
        setCurrentStep('retrospective')
        await runRetrospectiveAnalysis()
      }
    } finally {
      setConfigLoading(false)
    }
  }

  const filterResultsBySite = (siteFilter) => {
    if (!retrospectiveResults) return
    
    setSelectedSiteFilter(siteFilter)
    
    if (siteFilter === 'all') {
      setFilteredResults(retrospectiveResults)
      return
    }

    const filtered = {
      ...retrospectiveResults,
      results: retrospectiveResults.results.filter(r => r.site === siteFilter)
    }
    
    const isClassification = config.forecast_task === 'classification'
    const metrics = isClassification 
      ? calculateClassificationMetrics(filtered.results)
      : calculateRegressionMetrics(filtered.results)
    
    filtered.summary = {
      ...filtered.summary,
      total_forecasts: filtered.results.length,
      ...metrics
    }
    
    setFilteredResults(filtered)
  }

  const runRetrospectiveAnalysis = async () => {
    setLoading(true)
    setError(null)
    
    try {
      const response = await api.post('/api/retrospective', {
        selected_sites: config.selected_sites,
        task: config.forecast_task,
        model: config.forecast_model
      })
      
      setRetrospectiveResults(response.data)
      setFilteredResults(response.data)
      setSelectedSiteFilter('all')
      setCurrentStep('results')
    } catch (err) {
      setError('Failed to run retrospective analysis')
    } finally {
      setLoading(false)
    }
  }

  const handleRealtimeForecast = async () => {
    if (!selectedDate || !selectedSite) {
      setError('Please select both date and site')
      return
    }

    setLoading(true)
    setError(null)
    
    try {
      const response = await api.post('/api/forecast/enhanced', {
        date: format(selectedDate, 'yyyy-MM-dd'),
        site: selectedSite.value,
        task: task,
        model: selectedModel
      })
      
      setForecast(response.data)
      setCurrentStep('results')
    } catch (err) {
      setError(`Failed to generate forecast: ${err.response?.data?.detail || err.message}`)
    } finally {
      setLoading(false)
    }
  }

  const resetWorkflow = () => {
    setCurrentStep('config')
    setForecast(null)
    setRetrospectiveResults(null)
    setError(null)
  }

  // Create retrospective time series graph
  // Render different steps
  const renderConfigStep = () => (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="bg-white rounded-lg shadow-md p-8">
        <div className="text-center mb-6">
          <Settings className="w-12 h-12 text-blue-600 mx-auto mb-4" />
          <h1 className="text-3xl font-bold text-gray-800 mb-2">
            Configure DATect Forecasting System
          </h1>
          <p className="text-gray-600">
            Select your forecasting mode and parameters before proceeding
          </p>
        </div>

        <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4 text-blue-800">System Configuration</h2>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                <Clock className="w-4 h-4 inline mr-1" />
                Forecast Mode *
              </label>
              <select
                value={config.forecast_mode}
                onChange={(e) => setConfig({...config, forecast_mode: e.target.value})}
                className="w-full p-3 border border-gray-300 rounded-md text-lg"
              >
                <option value="realtime">Realtime - Interactive single forecasts</option>
                <option value="retrospective">Retrospective - Historical validation analysis</option>
              </select>
              <p className="text-xs text-gray-500 mt-1">
                {config.forecast_mode === 'realtime' 
                  ? 'Generate forecasts for specific dates and sites'
                  : 'Run comprehensive historical analysis with actual vs predicted comparisons'
                }
              </p>
            </div>


            {/* Task and Model selection only for retrospective mode */}
            {config.forecast_mode === 'retrospective' && (
              <>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    <BarChart3 className="w-4 h-4 inline mr-1" />
                    Forecast Task *
                  </label>
                  <select
                    value={config.forecast_task}
                    onChange={(e) => setConfig({...config, forecast_task: e.target.value})}
                    className="w-full p-3 border border-gray-300 rounded-md text-lg"
                  >
                    <option value="regression">Regression - Predict continuous DA levels (μg/g)</option>
                    <option value="classification">Classification - Predict risk categories (Low/Moderate/High/Extreme)</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    <Cpu className="w-4 h-4 inline mr-1" />
                    Machine Learning Model *
                  </label>
                  <select
                    value={config.forecast_model}
                    onChange={(e) => setConfig({...config, forecast_model: e.target.value})}
                    className="w-full p-3 border border-gray-300 rounded-md text-lg"
                  >
                    <option value="ensemble">Ensemble - XGBoost + RF (Recommended)</option>
                    <option value="naive">Naive Baseline - Most recent DA before anchor date</option>
                    <option value="linear">Ridge / Logistic - Interpretable linear models</option>
                  </select>
                </div>
              </>
            )}
          </div>

          <div className="mt-6 pt-4 border-t border-blue-200">
            <button
              onClick={applyConfig}
              disabled={configLoading}
              className="w-full bg-blue-600 text-white px-6 py-3 rounded-md hover:bg-blue-700 disabled:opacity-50 text-lg font-medium flex items-center justify-center"
            >
              {configLoading ? (
                <>
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                  Applying Configuration...
                </>
              ) : (
                <>
                  <Play className="w-5 h-5 mr-2" />
                  Apply Configuration & Continue
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  )

  const renderRealtimeStep = () => (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header with config summary */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-2xl font-bold text-gray-800 mb-2">
              Realtime Forecasting Interface
            </h1>
            <p className="text-gray-600">
              Mode: <span className="font-medium">{config.forecast_mode}</span> | 
              Task: <span className="font-medium">{config.forecast_task}</span> | 
              Model: <span className="font-medium">{config.forecast_model}</span>
            </p>
          </div>
          <button
            onClick={resetWorkflow}
            className="bg-gray-500 text-white px-4 py-2 rounded-md hover:bg-gray-600"
          >
            Change Config
          </button>
        </div>
      </div>

      {/* Forecast form */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold mb-4">Generate Forecast</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              <Calendar className="w-4 h-4 inline mr-1" />
              Forecast Date
            </label>
            <DatePicker
              selected={selectedDate}
              onChange={setSelectedDate}
              minDate={dateRange.min ? new Date(dateRange.min) : null}
              /* Allow forecasting beyond dataset end; no hard max */
              maxDate={null}
              className="w-full p-2 border border-gray-300 rounded-md"
              dateFormat="yyyy-MM-dd"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              <MapPin className="w-4 h-4 inline mr-1" />
              Monitoring Site
            </label>
            <Select
              value={selectedSite}
              onChange={setSelectedSite}
              options={sites.map(site => ({ value: site, label: site }))}
              className="text-sm"
              placeholder="Select site..."
            />
          </div>

          <div className="flex items-end">
            <button
              onClick={handleRealtimeForecast}
              disabled={loading || !selectedDate || !selectedSite}
              className="w-full bg-green-600 text-white px-6 py-2 rounded-md hover:bg-green-700 disabled:opacity-50 flex items-center justify-center"
            >
              {loading ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                  Generating...
                </>
              ) : (
                <>
                  <TrendingUp className="w-4 h-4 mr-2" />
                  Generate Forecast
                </>
              )}
            </button>
          </div>
        </div>
        
        {/* Helpful note */}
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mt-4">
          <p className="text-sm text-blue-700">
            💡 <strong>Tip:</strong> Change the date or site above and click "Generate Forecast" to get new predictions. 
            Each forecast is specific to the selected date and monitoring location.
          </p>
        </div>
      </div>
    </div>
  )

  const renderRetrospectiveStep = () => (
    <div className="max-w-4xl mx-auto">
      <div className="bg-white rounded-lg shadow-md p-8">
        <div className="mb-6">
          <h2 className="text-2xl font-bold text-gray-800 mb-2 text-center">
            Running Retrospective Analysis
          </h2>
          <p className="text-gray-600 mb-6 text-center">
            Processing historical data with {config.forecast_model} model for {config.forecast_task}...
          </p>
          
          {/* Loading spinner */}
          <div className="text-center mb-6">
            <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 mx-auto mb-4"></div>
          </div>
          
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <p className="text-sm text-blue-700">
              This analysis runs forecasts across historical time periods and compares 
              predicted values against actual measurements to validate model performance.
            </p>
          </div>
        </div>
      </div>
    </div>
  )

  const renderResults = () => {
    if (config.forecast_mode === 'realtime' && forecast) {
      return renderRealtimeResults()
    } else if (config.forecast_mode === 'retrospective' && retrospectiveResults) {
      return renderRetrospectiveResults()
    }
    return (
      <div className="max-w-4xl mx-auto">
        <div className="bg-white rounded-lg shadow-md p-6 text-center text-gray-600">
          No results available yet.
        </div>
      </div>
    )
  }

  const renderRealtimeResults = () => (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Results header */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex justify-between items-center">
          <h2 className="text-xl font-semibold">Realtime Forecast Results</h2>
          <button
            onClick={() => setCurrentStep('realtime')}
            className="bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600"
          >
            Generate Another
          </button>
        </div>
      </div>

      {forecast && !forecast.success && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center">
            <AlertTriangle className="w-5 h-5 text-red-600 mr-2" />
            <span className="text-red-800">
              {forecast.error || 'Realtime forecast failed. Please try another date or site.'}
            </span>
          </div>
        </div>
      )}

      {forecast && forecast.success && (
        <div className="space-y-6">
          {/* Summary */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {forecast.regression && (
                <div className="bg-blue-50 p-4 rounded-lg">
                  <h3 className="text-lg font-medium text-blue-800 mb-2">
                    🎯 DA Concentration Prediction
                  </h3>
                  <div className="text-2xl font-bold text-blue-600">
                    {forecast.regression.predicted_da?.toFixed(3)} μg/g
                  </div>
                  <p className="text-sm text-gray-600 mt-2">
                    Training samples: {forecast.regression.training_samples}
                  </p>
                </div>
              )}

              {forecast.classification && (
                <div className="bg-green-50 p-4 rounded-lg">
                  <h3 className="text-lg font-medium text-green-800 mb-2">
                    📊 Risk Category Prediction
                  </h3>
                  <div className="text-2xl font-bold text-green-600">
                    {['Low', 'Moderate', 'High', 'Extreme'][forecast.classification.predicted_category] || 'Unknown'}
                  </div>
                  <p className="text-sm text-gray-600 mt-2">
                    Training samples: {forecast.classification.training_samples}
                  </p>
                </div>
              )}

              {forecast.regression?.spike_probability != null && (
                <div className={`p-4 rounded-lg ${forecast.regression.spike_alert ? 'bg-red-50 border-2 border-red-300' : 'bg-emerald-50'}`}>
                  <h3 className={`text-lg font-medium mb-2 ${forecast.regression.spike_alert ? 'text-red-800' : 'text-emerald-800'}`}>
                    {forecast.regression.spike_alert ? '⚠️ Spike Alert' : '✅ No Spike Alert'}
                  </h3>
                  <div className={`text-2xl font-bold ${forecast.regression.spike_alert ? 'text-red-600' : 'text-emerald-600'}`}>
                    {(forecast.regression.spike_probability * 100).toFixed(1)}% spike probability
                  </div>
                  <p className="text-sm text-gray-600 mt-2">
                    Alert threshold: &gt;10% probability of DA &ge; 20 &mu;g/g
                  </p>
                </div>
              )}

            </div>
          </div>

          {/* Ensemble Breakdown */}
          {forecast.ensemble && (
            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-medium text-purple-800 mb-4">Ensemble Model Breakdown</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-purple-50 p-4 rounded-lg text-center">
                  <p className="text-sm text-gray-600 mb-1">XGBoost</p>
                  <p className="text-xl font-bold text-purple-700">
                    {forecast.ensemble.xgb_prediction?.toFixed(2)} μg/g
                  </p>
                  {forecast.ensemble.ensemble_weights && (
                    <p className="text-xs text-gray-500 mt-1">
                      weight: {(forecast.ensemble.ensemble_weights[0] * 100).toFixed(0)}%
                    </p>
                  )}
                </div>
                <div className="bg-purple-50 p-4 rounded-lg text-center">
                  <p className="text-sm text-gray-600 mb-1">Random Forest</p>
                  <p className="text-xl font-bold text-purple-700">
                    {forecast.ensemble.rf_prediction?.toFixed(2)} μg/g
                  </p>
                  {forecast.ensemble.ensemble_weights && (
                    <p className="text-xs text-gray-500 mt-1">
                      weight: {(forecast.ensemble.ensemble_weights[1] * 100).toFixed(0)}%
                    </p>
                  )}
                </div>
              </div>
              {forecast.ensemble.ensemble_prediction != null && (
                <div className="mt-4 bg-purple-100 p-3 rounded-lg text-center">
                  <p className="text-sm text-gray-600">Ensemble Prediction</p>
                  <p className="text-2xl font-bold text-purple-800">
                    {forecast.ensemble.ensemble_prediction?.toFixed(3)} μg/g
                  </p>
                </div>
              )}
              {forecast.ensemble.naive_prediction != null && (
                <div className="mt-4 border-t border-gray-200 pt-4">
                  <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-2">Baseline Comparison</p>
                  <div className="bg-gray-50 p-3 rounded-lg text-center">
                    <p className="text-sm text-gray-600 mb-1">Naive Persistence</p>
                    <p className="text-lg font-semibold text-gray-700">
                      {forecast.ensemble.naive_prediction?.toFixed(2)} μg/g
                    </p>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Level Range and Category Range Graphs - Match modular-forecast exactly */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Level Range Graph - Advanced Gradient Visualization */}
              {forecast.graphs && forecast.graphs.level_range && (
                <div>
                  {forecast.graphs.level_range.type === 'gradient_uncertainty' && forecast.graphs.level_range.gradient_plot && parsePlotlyJson(forecast.graphs.level_range.gradient_plot) ? (
                    // Use the advanced gradient plot from backend
                    <Plot
                      {...parsePlotlyJson(forecast.graphs.level_range.gradient_plot)}
                      config={{ responsive: true }}
                      style={{ width: '100%' }}
                    />
                  ) : (
                    // Fallback to simple visualization
                    <Plot
                      data={(() => {
                        const levelData = forecast.graphs.level_range;
                        const quantiles = levelData.gradient_quantiles || {};
                        // Use nullish coalescing so 0 is treated as valid
                        const q05 = (quantiles.q05 ?? levelData.q05);
                        const q50 = (quantiles.q50 ?? levelData.q50);
                        const q95 = (quantiles.q95 ?? levelData.q95);
                        const xgb_pred = (levelData.xgboost_prediction ?? levelData.predicted_da);

                        const traces = [];
                        const n_segments = 30;
                        const rawRange = (q95 ?? 0) - (q05 ?? 0);
                        const hasSpread = Number.isFinite(rawRange) && Math.abs(rawRange) > 1e-9;

                        // Median line (Bootstrap Q50)
                        traces.push({
                          x: [q50, q50],
                          y: [0.35, 0.65],
                          mode: 'lines',
                          line: { color: 'rgb(30, 60, 90)', width: 3 },
                          name: 'Bootstrap Median (Q50)'
                        });

                        // Range endpoints (Bootstrap quantiles)
                        traces.push({
                          x: [q05, q95],
                          y: [0.5, 0.5],
                          mode: 'markers',
                          marker: { size: 12, color: 'rgba(70, 130, 180, 0.4)', symbol: 'line-ns-open' },
                          name: 'Bootstrap Range (Q05-Q95)'
                        });

                        // XGBoost point prediction
                        if (xgb_pred !== undefined && xgb_pred !== null) {
                          traces.push({
                            x: [xgb_pred],
                            y: [0.5],
                            mode: 'markers',
                            marker: {
                              size: 14,
                              color: 'darkorange',
                              symbol: 'diamond-tall',
                              line: { width: 2, color: 'black' }
                            },
                            name: 'XGBoost Prediction'
                          });
                        }

                        return traces;
                      })()}
                      layout={(() => {
                        const levelData = forecast.graphs.level_range;
                        const quantiles = levelData.gradient_quantiles || {};
                        const q05 = (quantiles.q05 ?? levelData.q05);
                        const q50 = (quantiles.q50 ?? levelData.q50);
                        const q95 = (quantiles.q95 ?? levelData.q95);
                        const xgb_pred = (levelData.xgboost_prediction ?? levelData.predicted_da);

                        const n_segments = 30;
                        const rawRange = (q95 ?? 0) - (q05 ?? 0);
                        const hasSpread = Number.isFinite(rawRange) && Math.abs(rawRange) > 1e-9;

                        // Define a display range if quantiles collapse
                        const displayQ05 = hasSpread ? q05 : (Number.isFinite(q50) ? q50 - 1 : -1);
                        const displayQ95 = hasSpread ? q95 : (Number.isFinite(q50) ? q50 + 1 : 1);
                        const displayWidth = (displayQ95 ?? 0) - (displayQ05 ?? 0);
                        const maxDistance = Math.max((q50 ?? 0) - (displayQ05 ?? 0), (displayQ95 ?? 0) - (q50 ?? 0)) || 1;

                        // Compute x-axis range with padding, ensuring nonzero span
                        let xCandidates = [displayQ05, displayQ95];
                        if (q50 !== undefined && q50 !== null) xCandidates.push(q50);
                        if (xgb_pred !== undefined && xgb_pred !== null) xCandidates.push(xgb_pred);
                        let xMin = Math.min(...xCandidates);
                        let xMax = Math.max(...xCandidates);
                        if (!Number.isFinite(xMin) || !Number.isFinite(xMax) || Math.abs(xMax - xMin) < 1e-9) {
                          const base = Number.isFinite(q50) ? q50 : 0;
                          xMin = base - 1.5;
                          xMax = base + 1.5;
                        } else {
                          const pad = 0.05 * (xMax - xMin);
                          xMin -= pad;
                          xMax += pad;
                        }

                        // Build gradient shapes over display interval
                        const shapes = [];
                        for (let i = 0; i < n_segments; i++) {
                          const x0 = (displayQ05 ?? 0) + (i / n_segments) * displayWidth;
                          const x1 = (displayQ05 ?? 0) + ((i + 1) / n_segments) * displayWidth;
                          const midpoint = (x0 + x1) / 2;
                          const distance = Math.abs(midpoint - (q50 ?? 0));
                          const opacity = Math.max(0.1, Math.min(0.9, 1 - Math.pow(distance / maxDistance, 0.5)));
                          shapes.push({
                            type: 'rect',
                            x0,
                            x1,
                            y0: 0.35,
                            y1: 0.65,
                            fillcolor: `rgba(70, 130, 180, ${opacity})`,
                            line: { width: 0 },
                            layer: 'below'
                          });
                        }

                        return {
                          title: "DA Level Forecast with Bootstrap Confidence Intervals",
                          xaxis: { title: "DA Level (μg/L)", range: [xMin, xMax] },
                          yaxis: { visible: false, range: [0, 1] },
                          showlegend: true,
                          height: 350,
                          plot_bgcolor: 'white',
                          shapes
                        };
                      })()}
                      config={{ responsive: true }}
                      style={{ width: '100%' }}
                    />
                  )}
                </div>
              )}
              
              {/* Category Range Graph - for classification */}
              {forecast.graphs && forecast.graphs.category_range && (
                <div>
                  <Plot
                    data={[{
                      x: forecast.graphs.category_range.category_labels || ['Low (<5)', 'Moderate (5–<20)', 'High (20–<40)', 'Extreme (≥40)'],
                      y: forecast.graphs.category_range.class_probabilities || [0, 0, 0, 0],
                      type: 'bar',
                      marker: {
                        color: forecast.graphs.category_range.category_labels?.map((_, i) => 
                          i === forecast.graphs.category_range.predicted_category ? '#2ca02c' : '#1f77b4'
                        ) || ['#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4']
                      },
                      text: forecast.graphs.category_range.class_probabilities?.map(p => `${(p * 100).toFixed(1)}%`) || [],
                      textposition: 'auto'
                    }]}
                    layout={{
                      title: "Category Probability Distribution",
                      yaxis: { title: "Probability", range: [0, 1.1] },
                      xaxis: { title: "Category" },
                      showlegend: false,
                      height: 400
                    }}
                    config={{ responsive: true }}
                    style={{ width: '100%' }}
                  />
                </div>
              )}
            </div>
          </div>
          
          {/* Feature Importance Graph - if available */}
          {(forecast.regression?.feature_importance || forecast.classification?.feature_importance) && (
            <div className="bg-white rounded-lg shadow-md p-6 mt-6">
              <h3 className="text-lg font-semibold mb-4">Top Feature Importance</h3>
              <Plot
                data={[{
                  x: (() => {
                    const raw = forecast.regression?.feature_importance || forecast.classification?.feature_importance
                    const items = Array.isArray(raw)
                      ? raw
                      : Object.entries(raw || {}).map(([feature, importance]) => ({ feature, importance }))
                    return items
                      .sort((a, b) => b.importance - a.importance)
                      .slice(0, 15)
                      .map(f => f.importance)
                  })(),
                  y: (() => {
                    const raw = forecast.regression?.feature_importance || forecast.classification?.feature_importance
                    const items = Array.isArray(raw)
                      ? raw
                      : Object.entries(raw || {}).map(([feature, importance]) => ({ feature, importance }))
                    return items
                      .sort((a, b) => b.importance - a.importance)
                      .slice(0, 15)
                      .map(f => f.feature)
                  })(),
                  type: 'bar',
                  orientation: 'h',
                  marker: { color: 'steelblue' }
                }]}
                layout={{
                  title: "Top Feature Importance",
                  xaxis_title: "Importance Score",
                  yaxis_title: "Features",
                  height: 400,
                  yaxis: { categoryorder: 'total ascending' }
                }}
                config={{ responsive: true }}
                style={{ width: '100%' }}
              />
            </div>
          )}
        </div>
      )}
    </div>
  )

  const renderRetrospectiveResults = () => {
    const timeSeriesData = buildRetrospectiveTimeSeries(filteredResults, config)
    const scatterData = buildRetrospectiveScatter(filteredResults, config)
    
    return (
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Results header */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex justify-between items-center">
            <div>
              <h2 className="text-2xl font-semibold">Retrospective Analysis Results</h2>
              <p className="text-gray-600">
                Model: {config.forecast_model} | Task: {config.forecast_task}
              </p>
            </div>
            <button
              onClick={resetWorkflow}
              className="bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600"
            >
              New Analysis
            </button>
          </div>
        </div>

        {/* Site filtering controls + per-site quality badges + coverage warning */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center gap-4 flex-wrap">
            <label className="text-sm font-medium text-gray-700">
              <MapPin className="w-4 h-4 inline mr-1" />
              Filter by Site:
            </label>
            <select
              value={selectedSiteFilter}
              onChange={(e) => filterResultsBySite(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-md text-sm"
            >
              <option value="all">All Sites</option>
              {sites.map(site => {
                // Quality tier from each site's deterministic holdout R²
                // (all real DA in the window, no sampling; /api/metrics/summary)
                const siteMetrics = metricsSummary?.per_site?.[site]
                let qualityTier = ''
                if (metricsSummary) {
                  if (!siteMetrics) qualityTier = '⚪'  // too few holdout measurements to score
                  else if (siteMetrics.r2[0] >= 0.4) qualityTier = '🟢'
                  else if (siteMetrics.r2[0] >= 0) qualityTier = '🟡'
                  else qualityTier = '🔴'
                }
                return (
                  <option key={site} value={site}>
                    {qualityTier} {site}
                  </option>
                )
              })}
            </select>
            <span className="text-sm text-gray-600">
              Showing {filteredResults?.results?.length || 0} forecasts
            </span>
            <span className="text-xs text-gray-500 italic">
              🟢 R² &gt; 0.4 · 🟡 R² 0–0.4 · 🔴 R² &lt; 0 · ⚪ insufficient holdout data
            </span>
            <button
              onClick={() => exportResultsCSV(filteredResults, selectedSiteFilter, config.forecast_task)}
              disabled={!filteredResults?.results?.length}
              className="ml-auto inline-flex items-center gap-1.5 px-3 py-1.5 text-sm bg-gray-100 hover:bg-gray-200 disabled:opacity-40 rounded-md transition-colors"
              title="Download filtered results as CSV"
            >
              <Download className="w-4 h-4" /> Export CSV
            </button>
          </div>

          {/* Data-coverage warning banner — appears when filtered site has few recent samples */}
          {(() => {
            if (selectedSiteFilter === 'all' || !filteredResults?.results?.length) return null
            const recent = filteredResults.results.filter(r => new Date(r.date) >= new Date('2022-01-01'))
            const recentSpikes = recent.filter(r => r.actual_da >= 20).length
            if (recent.length < 10) {
              return (
                <div className="mt-3 bg-amber-50 border border-amber-200 rounded-lg p-3 text-sm text-amber-900">
                  <AlertTriangle className="w-4 h-4 inline mr-1.5 text-amber-600" />
                  <strong>Limited 2022-2023 holdout data for {selectedSiteFilter}:</strong>{' '}
                  only {recent.length} test points, {recentSpikes} spike events.
                  {recent.length === 0
                    ? ' This evaluation samples no recent data — metrics reflect historical periods only.'
                    : ' Small holdout makes single-seed metrics unreliable; treat per-site numbers as indicative, not authoritative.'}
                </div>
              )
            }
            if (recentSpikes === 0 && recent.length >= 10) {
              return (
                <div className="mt-3 bg-amber-50 border border-amber-200 rounded-lg p-3 text-sm text-amber-900">
                  <AlertTriangle className="w-4 h-4 inline mr-1.5 text-amber-600" />
                  <strong>{selectedSiteFilter}: no spike events in 2022-2023 holdout</strong> (across {recent.length} samples).
                  Spike-detection metrics cannot be evaluated here. Site may still have historical spikes
                  visible in the time series below.
                </div>
              )
            }
            return null
          })()}
        </div>

        {/* Summary statistics */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4">Performance Summary</h3>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
            <div className="bg-blue-50 p-4 rounded-lg text-center">
              <div className="text-2xl font-bold text-blue-600">{filteredResults?.summary?.total_forecasts || 0}</div>
              <div className="text-sm text-gray-600">Total Forecasts</div>
            </div>
            {config.forecast_task === 'regression' && filteredResults?.summary?.r2_score !== undefined && (
              <div className="bg-green-50 p-4 rounded-lg text-center relative group">
                <div className="text-2xl font-bold text-green-600">{filteredResults.summary.r2_score.toFixed(3)}</div>
                <div className="text-sm text-gray-600 inline-flex items-center justify-center gap-1">
                  R² Score
                  <Info className="w-3 h-3 text-gray-400 cursor-help" />
                </div>
                <div className="absolute z-10 hidden group-hover:block w-80 left-1/2 -translate-x-1/2 mt-2 p-3 bg-gray-900 text-white text-xs rounded-lg shadow-lg text-left">
                  <div className="font-semibold mb-1">3-window evaluation split</div>
                  <div className="space-y-1">
                    <div><span className="text-blue-300">Training:</span> pre-anchor data only (per-anchor)</div>
                    <div><span className="text-yellow-300">Validation:</span> 2019–2022 (used by tuning)</div>
                    <div><span className="text-green-300">Holdout:</span> 2022–2023 (never seen by tuning)</div>
                  </div>
                  {metricsSummary && (
                    <div className="mt-2 text-gray-300 italic">
                      Headline holdout R² = <span className="text-green-300 font-semibold">{metricsSummary.overall.r2[0].toFixed(3)}</span>{' '}
                      [{metricsSummary.overall.r2[1].toFixed(3)}, {metricsSummary.overall.r2[2].toFixed(3)}] (deterministic, all real DA, N={metricsSummary.overall.n})
                      {metricsSummary.multi_seed?.r2_mean != null && (
                        <>; {metricsSummary.multi_seed.n_seeds}-seed bootstrap confirmation {metricsSummary.multi_seed.r2_mean.toFixed(3)} ± {metricsSummary.multi_seed.r2_std.toFixed(3)}</>
                      )}. Pooled R² shown above mixes all windows. Filter by year/site for holdout-only view.
                    </div>
                  )}
                </div>
              </div>
            )}
            {config.forecast_task === 'regression' && filteredResults?.summary?.mae !== undefined && (
              <div className="bg-yellow-50 p-4 rounded-lg text-center">
                <div className="text-2xl font-bold text-yellow-600">{filteredResults.summary.mae.toFixed(2)}</div>
                <div className="text-sm text-gray-600">MAE (μg/g)</div>
              </div>
            )}
            {config.forecast_task === 'regression' && filteredResults?.summary?.spike_recall !== undefined && (
              <div className="bg-orange-50 p-4 rounded-lg text-center">
                <div className="text-2xl font-bold text-orange-600">
                  {(filteredResults.summary.spike_recall * 100).toFixed(0)}%
                </div>
                <div className="text-sm text-gray-600">Spike Recall</div>
                <div className="text-xs text-gray-500 mt-1">
                  {filteredResults.summary.n_spike_events ?? '?'} spike events,
                  {' '}precision {((filteredResults.summary.spike_precision ?? 0) * 100).toFixed(0)}%,
                  {' '}F2 {(filteredResults.summary.spike_f2 ?? 0).toFixed(2)}
                </div>
                <div className="text-xs text-gray-400 mt-1">
                  Regression-only: predicted ≥{filteredResults.summary.spike_threshold_predicted ?? 12} →
                  actual ≥{filteredResults.summary.spike_threshold_actual ?? 20} µg/g
                </div>
              </div>
            )}
            {config.forecast_task === 'regression' && (() => {
              const results = filteredResults?.results || []
              const withAlerts = results.filter(d => d.spike_alert !== undefined && d.spike_alert !== null)
              if (withAlerts.length === 0) return null
              const alertCount = withAlerts.filter(d => d.spike_alert === true).length
              const truePositives = withAlerts.filter(d => d.spike_alert === true && d.actual_da >= 20).length
              const actualSpikes = withAlerts.filter(d => d.actual_da >= 20).length
              const recall = actualSpikes > 0 ? (truePositives / actualSpikes * 100) : null
              return (
                <div className={`p-4 rounded-lg text-center ${alertCount > 0 ? 'bg-red-50' : 'bg-emerald-50'}`}>
                  <div className={`text-2xl font-bold ${alertCount > 0 ? 'text-red-600' : 'text-emerald-600'}`}>
                    {recall != null ? `${recall.toFixed(0)}%` : 'N/A'}
                  </div>
                  <div className="text-sm text-gray-600">Hybrid Alert Recall</div>
                  <div className="text-xs text-gray-500 mt-1">
                    {truePositives} of {actualSpikes} spikes caught · {alertCount} alerts raised
                  </div>
                  <div className="text-xs text-gray-400 mt-1">
                    Uses combined classifier probability + regression threshold
                  </div>
                </div>
              )
            })()}
            {config.forecast_task === 'classification' && filteredResults?.summary?.accuracy !== undefined && (
              <>
                <div className="bg-purple-50 p-4 rounded-lg text-center">
                  <div className="text-2xl font-bold text-purple-600">{(filteredResults.summary.accuracy * 100).toFixed(1)}%</div>
                  <div className="text-sm text-gray-600">Accuracy</div>
                </div>
                {filteredResults?.summary?.balanced_accuracy !== undefined && (
                  <div className="bg-indigo-50 p-4 rounded-lg text-center">
                    <div className="text-2xl font-bold text-indigo-600">{(filteredResults.summary.balanced_accuracy * 100).toFixed(1)}%</div>
                    <div className="text-sm text-gray-600">Balanced Accuracy</div>
                    <div className="text-xs text-gray-500 mt-1">Accounts for class imbalance</div>
                  </div>
                )}
              </>
            )}
          </div>
        </div>

        {/* Per-class recall metrics for classification */}
        {config.forecast_task === 'classification' && filteredResults?.summary?.per_class_metrics && (
          <div className="bg-white rounded-lg shadow-md p-6 mb-6">
            <h3 className="text-lg font-semibold mb-4">Per-Class Performance</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {Object.entries(filteredResults.summary.per_class_metrics).map(([className, metrics]) => (
                <div key={className} className="border rounded-lg p-3">
                  <div className="font-semibold text-sm mb-2">{className}</div>
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-600">Recall:</span>
                      <span className="font-medium">{(metrics.recall * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-600">Precision:</span>
                      <span className="font-medium">{(metrics.precision * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-600">F1:</span>
                      <span className="font-medium">{(metrics.f1 * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-600">Samples:</span>
                      <span className="font-medium">{metrics.support}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-4 text-sm text-gray-600">
              <p><strong>Recall:</strong> % of actual cases correctly identified (e.g., 85% recall for High means we catch 85% of high toxin events)</p>
              <p><strong>Precision:</strong> % of predictions that were correct (e.g., 70% precision for High means when we predict High, we're right 70% of the time)</p>
            </div>
          </div>
        )}

        {/* Time series plot */}
        {timeSeriesData && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold mb-4">
              Actual vs Predicted Time Series
              {(config.selected_sites || []).length < sites.length && (config.selected_sites || []).length > 0 && (
                <span className="text-sm font-normal text-gray-600 ml-2">
                  ({(config.selected_sites || []).length} of {sites.length} sites)
                </span>
              )}
            </h3>
            <Plot
              data={timeSeriesData.data}
              layout={timeSeriesData.layout}
              config={{
                ...plotConfig,
                toImageButtonOptions: {
                  ...plotConfig.toImageButtonOptions,
                  filename: getPlotFilename(`retrospective_timeseries_${selectedSiteFilter}`)
                }
              }}
              style={{ width: '100%' }}
            />
          </div>
        )}

        {/* Scatter plot */}
        {scatterData && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold mb-4">Model Performance Scatter Plot</h3>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Plot
                data={scatterData.data}
                layout={scatterData.layout}
                config={{
                  ...plotConfig,
                  toImageButtonOptions: {
                    ...plotConfig.toImageButtonOptions,
                    filename: getPlotFilename(`retrospective_scatter_${config.forecast_task}_${selectedSiteFilter}`)
                  }
                }}
                style={{ width: '100%' }}
              />
              <div className="bg-gray-50 p-4 rounded-lg">
                <h4 className="font-medium mb-3">Interpretation Guide</h4>
                <ul className="text-sm space-y-2 text-gray-700">
                  <li>• Points closer to the diagonal line indicate better predictions</li>
                  <li>• Color represents time window (gray=pre-2019 training context, amber=2019-22 validation, green=2022-24 untouched holdout)</li>
                  <li>• Red dotted vertical = spike threshold (actual &gt;20 µg/g)</li>
                  <li>• Orange dotted horizontal = alert threshold (predicted &gt;12 µg/g)</li>
                  <li>• Quadrants: top-right = caught spikes, bottom-right = MISSED spikes, top-left = false alarms</li>
                </ul>
              </div>
            </div>
          </div>
        )}

        {/* Per-site small-multiples scatter — only when viewing All Sites */}
        {config.forecast_task === 'regression' && selectedSiteFilter === 'all' && filteredResults?.results?.length > 0 && (() => {
          // Build per-site mini-data
          const bySite = {}
          filteredResults.results.forEach(r => {
            if (r.actual_da == null || r.predicted_da == null) return
            if (!bySite[r.site]) bySite[r.site] = []
            bySite[r.site].push(r)
          })
          // Sort by N descending so dense sites appear first
          const orderedSites = Object.keys(bySite).sort((a, b) => bySite[b].length - bySite[a].length)
          if (orderedSites.length === 0) return null

          // Compute per-site R² so we can show it in each subtitle
          const r2 = (data) => {
            if (data.length < 3) return null
            const a = data.map(d => d.actual_da)
            const p = data.map(d => d.predicted_da)
            const aBar = a.reduce((s, v) => s + v, 0) / a.length
            const ssRes = a.reduce((s, v, i) => s + (v - p[i]) ** 2, 0)
            const ssTot = a.reduce((s, v) => s + (v - aBar) ** 2, 0)
            return ssTot > 0 ? 1 - ssRes / ssTot : null
          }
          const tierColor = (r) => r == null ? '#94a3b8' : r > 0.4 ? '#10b981' : r > 0 ? '#fbbf24' : '#ef4444'

          // Window-color logic shared with main scatter
          const windowColor = (date) => {
            const d = new Date(date)
            if (d < new Date('2019-01-01')) return '#94a3b8'
            if (d < new Date('2022-01-01')) return '#fbbf24'
            return '#10b981'
          }

          return (
            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-semibold mb-2">Per-Site Performance (small multiples)</h3>
              <p className="text-sm text-gray-600 mb-4">
                Each beach gets its own scatter so you can see per-site performance instead of one pooled chart.
                Color = time window. Diagonal = perfect prediction. Subtitle shows in-sample R².
              </p>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
                {orderedSites.map(site => {
                  const data = bySite[site]
                  const siteR2 = r2(data)
                  const r2Text = siteR2 == null ? 'N/A (too few)' : `R² = ${siteR2.toFixed(2)}`
                  const maxVal = Math.max(
                    ...data.map(d => Math.max(d.actual_da, d.predicted_da)), 5
                  )
                  return (
                    <div
                      key={site}
                      className="border rounded p-2"
                      style={{ borderColor: tierColor(siteR2), borderWidth: '2px' }}
                    >
                      <Plot
                        data={[
                          { x: [0, maxVal], y: [0, maxVal], mode: 'lines',
                            line: { color: '#d1d5db', width: 1, dash: 'dash' },
                            showlegend: false, hoverinfo: 'skip' },
                          { x: [20, 20], y: [0, maxVal], mode: 'lines',
                            line: { color: '#ef4444', width: 0.7, dash: 'dot' },
                            showlegend: false, hoverinfo: 'skip' },
                          { x: [0, maxVal], y: [12, 12], mode: 'lines',
                            line: { color: '#f59e0b', width: 0.7, dash: 'dot' },
                            showlegend: false, hoverinfo: 'skip' },
                          { x: data.map(d => d.actual_da),
                            y: data.map(d => d.predicted_da),
                            mode: 'markers', type: 'scatter',
                            marker: {
                              color: data.map(d => windowColor(d.date)),
                              size: 5, opacity: 0.7,
                            },
                            text: data.map(d => `${d.site}<br>${d.date}`),
                            hovertemplate: '%{text}<br>actual %{x:.1f}, pred %{y:.1f}<extra></extra>',
                            showlegend: false,
                          },
                        ]}
                        layout={{
                          title: { text: `${site}<br><sub>N=${data.length} · ${r2Text}</sub>`, font: { size: 11 } },
                          xaxis: { title: '', range: [0, maxVal], showticklabels: maxVal >= 10 },
                          yaxis: { title: '', range: [0, maxVal], showticklabels: maxVal >= 10 },
                          margin: { l: 30, r: 10, t: 35, b: 25 },
                          height: 200,
                          showlegend: false,
                        }}
                        config={{ displayModeBar: false, responsive: true }}
                        style={{ width: '100%' }}
                      />
                    </div>
                  )
                })}
              </div>
            </div>
          )
        })()}
      </div>
    )
  }

  // Main render
  return (
    <div className="min-h-screen bg-gray-50 py-8">
      {/* Error Display */}
      {error && (
        <div className="max-w-4xl mx-auto mb-6">
          <div className="bg-red-50 border border-red-200 rounded-lg p-4">
            <div className="flex items-center">
              <AlertTriangle className="w-5 h-5 text-red-600 mr-2" />
              <span className="text-red-800">{error}</span>
            </div>
          </div>
        </div>
      )}

      {/* Render current step */}
      {currentStep === 'config' && renderConfigStep()}
      {currentStep === 'realtime' && renderRealtimeStep()}
      {currentStep === 'retrospective' && renderRetrospectiveStep()}
      {currentStep === 'results' && renderResults()}
    </div>
  )
}

export default Dashboard
