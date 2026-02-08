import React, { useState, useEffect } from 'react'
import { Calendar, MapPin, Cpu, AlertTriangle, Settings, Play, BarChart3, TrendingUp, Clock } from 'lucide-react'
import DatePicker from 'react-datepicker'
import Select from 'react-select'
import Plot from 'react-plotly.js'
import { format, subDays } from 'date-fns'
import api from '../services/api'
import { plotConfig, getPlotFilename } from '../utils/plotConfig'
import { SITE_COLORS } from '../utils/constants'
import 'react-datepicker/dist/react-datepicker.css'

const Dashboard = () => {
  const [currentStep, setCurrentStep] = useState('config')
  
  const [config, setConfig] = useState({
    forecast_mode: 'realtime',
    forecast_task: 'regression', 
    forecast_model: 'xgboost',
    selected_sites: []
  })
  const [configLoading, setConfigLoading] = useState(false)
  
  const [sites, setSites] = useState([])
  const [models, setModels] = useState({ regression: [], classification: [] })
  const [dateRange, setDateRange] = useState({ min: null, max: null })
  
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
    
    try {
      const response = await api.post('/api/config', config)
      if (response.data.success) {
        // Update local state
        setTask(config.forecast_task)
        setSelectedModel(config.forecast_model)
        
        // Move to next step based on mode
        if (config.forecast_mode === 'realtime') {
          setCurrentStep('realtime')
        } else {
          // Start retrospective analysis immediately
          setCurrentStep('retrospective')
          await runRetrospectiveAnalysis()
        }
      }
    } catch (err) {
      setError('Failed to update configuration')
    } finally {
      setConfigLoading(false)
    }
  }

  const calculateClassificationMetrics = (results) => {
    const validClassification = results.filter(r => 
      r.actual_category !== null && r.actual_category !== undefined &&
      r.predicted_category !== null && r.predicted_category !== undefined
    )
    
    if (validClassification.length === 0) return {}
    
    const correctPredictions = validClassification.filter(r => 
      r.actual_category === r.predicted_category
    ).length
    const accuracy = correctPredictions / validClassification.length
    
    const classes = [0, 1, 2, 3]
    const classNames = ['Low', 'Moderate', 'High', 'Extreme']
    let balancedAccSum = 0
    let validClasses = 0
    const perClassMetrics = {}
    
    classes.forEach((cls, idx) => {
      const actualInClass = validClassification.filter(r => r.actual_category === cls)
      const predictedInClass = validClassification.filter(r => r.predicted_category === cls)
      const truePositives = validClassification.filter(r => 
        r.actual_category === cls && r.predicted_category === cls
      ).length
      
      if (actualInClass.length > 0) {
        const recall = truePositives / actualInClass.length
        const precision = predictedInClass.length > 0 ? truePositives / predictedInClass.length : 0
        const f1 = (precision + recall) > 0 ? 2 * (precision * recall) / (precision + recall) : 0
        
        balancedAccSum += recall
        validClasses += 1
        
        perClassMetrics[classNames[idx]] = {
          recall, precision, f1,
          support: actualInClass.length
        }
      }
    })
    
    const balancedAccuracy = validClasses > 0 ? balancedAccSum / validClasses : 0
    
    return {
      classification_forecasts: validClassification.length,
      accuracy,
      balanced_accuracy: balancedAccuracy,
      per_class_metrics: perClassMetrics
    }
  }

  const calculateRegressionMetrics = (results) => {
    const validRegression = results.filter(r => 
      r.actual_da !== null && r.predicted_da !== null
    )
    
    if (validRegression.length === 0) return {}
    
    const actuals = validRegression.map(r => r.actual_da)
    const predictions = validRegression.map(r => r.predicted_da)
    
    // R2 calculation
    const meanActual = actuals.reduce((a, b) => a + b, 0) / actuals.length
    const ssTotal = actuals.reduce((sum, val) => sum + Math.pow(val - meanActual, 2), 0)
    const ssResidual = actuals.reduce((sum, val, i) => 
      sum + Math.pow(val - predictions[i], 2), 0
    )
    const r2 = 1 - (ssResidual / ssTotal)
    
    // MAE calculation
    const mae = actuals.reduce((sum, val, i) => 
      sum + Math.abs(val - predictions[i]), 0
    ) / actuals.length
    
    // F1 score for spike detection
    const actualSpikes = actuals.map(val => val > 20 ? 1 : 0)
    const predictedSpikes = predictions.map(val => val > 20 ? 1 : 0)
    
    const truePositives = actualSpikes.reduce((sum, actual, i) => 
      sum + (actual === 1 && predictedSpikes[i] === 1 ? 1 : 0), 0
    )
    const falsePositives = actualSpikes.reduce((sum, actual, i) => 
      sum + (actual === 0 && predictedSpikes[i] === 1 ? 1 : 0), 0
    )
    const falseNegatives = actualSpikes.reduce((sum, actual, i) => 
      sum + (actual === 1 && predictedSpikes[i] === 0 ? 1 : 0), 0
    )
    
    const precision = truePositives + falsePositives > 0 ? truePositives / (truePositives + falsePositives) : 0
    const recall = truePositives + falseNegatives > 0 ? truePositives / (truePositives + falseNegatives) : 0
    const f1Score = precision + recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0
    
    return {
      regression_forecasts: validRegression.length,
      r2_score: r2,
      mae,
      f1_score: f1Score
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
        selected_sites: config.selected_sites
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
  const createRetrospectiveTimeSeries = () => {
    if (!filteredResults?.results) return null

    const results = filteredResults.results
    const isClassification = config.forecast_task === 'classification'
    
    // Group by site for better visualization
    const sites = [...new Set(results.map(r => r.site))] // Show all sites
    const isSingleSite = sites.length === 1
    
    const traces = []
    
    if (isClassification) {
      // Classification time series
      sites.forEach((site, siteIndex) => {
        const siteData = results
          .filter(r => r.site === site && r.actual_category !== null && r.actual_category !== undefined && 
                       r.predicted_category !== null && r.predicted_category !== undefined)
          .sort((a, b) => new Date(a.date) - new Date(b.date))
        
        if (siteData.length === 0) return
        
        const siteColor = SITE_COLORS[siteIndex % SITE_COLORS.length]
        
        // Color logic: single site = blue/red, multiple sites = site-specific colors
        const actualColor = isSingleSite ? 'blue' : siteColor
        const predictedColor = isSingleSite ? 'red' : siteColor
        
        // Actual categories
        traces.push({
          x: siteData.map(d => d.date),
          y: siteData.map(d => d.actual_category),
          mode: 'lines+markers',
          name: `${site} - Actual Category`,
          line: { color: actualColor, width: 2 },
          marker: { size: 6 },
          hovertemplate: '<b>%{text}</b><br>Date: %{x}<br>Actual Category: %{customdata}<extra></extra>',
          text: siteData.map(d => site),
          customdata: siteData.map(d => {
            const categories = ['Low', 'Moderate', 'High', 'Extreme']
            return categories[d.actual_category] || `Category ${d.actual_category}`
          })
        })
        
        // Predicted categories
        traces.push({
          x: siteData.map(d => d.date),
          y: siteData.map(d => d.predicted_category),
          mode: 'lines+markers',
          name: `${site} - Predicted Category`,
          line: { color: predictedColor, width: 2, dash: 'dash' },
          marker: { size: 6, symbol: 'square' },
          hovertemplate: '<b>%{text}</b><br>Date: %{x}<br>Predicted Category: %{customdata}<extra></extra>',
          text: siteData.map(d => site),
          customdata: siteData.map(d => {
            const categories = ['Low', 'Moderate', 'High', 'Extreme']
            return categories[d.predicted_category] || `Category ${d.predicted_category}`
          })
        })
      })
      
      return {
        data: traces,
        layout: {
          title: `DA Category Forecasting Results - ${config.forecast_model} Classification`,
          xaxis: { title: 'Date' },
          yaxis: { 
            title: 'DA Risk Category',
            tickmode: 'array',
            tickvals: [0, 1, 2, 3],
            ticktext: ['Low', 'Moderate', 'High', 'Extreme'],
            range: [-0.5, 3.5]
          },
          height: 500,
          hovermode: 'closest'
        }
      }
    } else {
      // Regression time series (existing code)
      sites.forEach((site, siteIndex) => {
        const siteData = results
          .filter(r => r.site === site && r.actual_da !== null && r.predicted_da !== null)
          .sort((a, b) => new Date(a.date) - new Date(b.date))
        
        if (siteData.length === 0) return
        
        const siteColor = SITE_COLORS[siteIndex % SITE_COLORS.length]
        
        // Color logic: single site = blue/red, multiple sites = site-specific colors
        const actualColor = isSingleSite ? 'blue' : siteColor
        const predictedColor = isSingleSite ? 'red' : siteColor
        
        // Actual values
        traces.push({
          x: siteData.map(d => d.date),
          y: siteData.map(d => d.actual_da),
          mode: 'lines+markers',
          name: `${site} - Actual`,
          line: { color: actualColor, width: 2 },
          marker: { size: 4 }
        })
        
        // Predicted values
        traces.push({
          x: siteData.map(d => d.date),
          y: siteData.map(d => d.predicted_da),
          mode: 'lines+markers',
          name: `${site} - Predicted`,
          line: { color: predictedColor, width: 2, dash: 'dash' },
          marker: { size: 4, symbol: 'square' }
        })
      })

      return {
        data: traces,
        layout: {
          title: `Retrospective Analysis: Actual vs Predicted DA Concentrations (${config.forecast_task})`,
          xaxis: { title: 'Date' },
          yaxis: { title: 'DA Concentration (μg/g)' },
          height: 500,
          hovermode: 'closest'
        }
      }
    }
  }

  // Create scatter plot for retrospective results
  const createRetrospectiveScatter = () => {
    if (!filteredResults?.results) return null
    
    const isClassification = config.forecast_task === 'classification'
    
    if (isClassification) {
      // Classification scatter plot (confusion matrix style)
      const validData = filteredResults.results.filter(r => 
        r.actual_category !== null && r.actual_category !== undefined &&
        r.predicted_category !== null && r.predicted_category !== undefined
      )
      
      if (validData.length === 0) return null
      
      // Calculate accuracy
      const correctPredictions = validData.filter(d => d.actual_category === d.predicted_category).length
      const accuracy = correctPredictions / validData.length
      
      // Add jitter to see overlapping points better
      const jitterStrength = 0.1
      const jitteredData = validData.map(d => ({
        ...d,
        x_jitter: d.actual_category + (Math.random() - 0.5) * jitterStrength * 2,
        y_jitter: d.predicted_category + (Math.random() - 0.5) * jitterStrength * 2
      }))
      
      // Group by site for colors
      const siteGroups = {}
      jitteredData.forEach(d => {
        if (!siteGroups[d.site]) {
          siteGroups[d.site] = []
        }
        siteGroups[d.site].push(d)
      })
      
      const traces = []
      
      // Add diagonal reference line for perfect predictions
      traces.push({
        x: [-0.5, 3.5],
        y: [-0.5, 3.5],
        mode: 'lines',
        line: { color: 'red', width: 2, dash: 'dash' },
        name: 'Perfect Prediction',
        hoverinfo: 'skip'
      })
      
      // Add scatter points for each site
      Object.entries(siteGroups).forEach(([site, data], index) => {
        traces.push({
          x: data.map(d => d.x_jitter),
          y: data.map(d => d.y_jitter),
          mode: 'markers',
          type: 'scatter',
          name: site,
          marker: { 
            color: SITE_COLORS[index % SITE_COLORS.length],
            size: 8,
            opacity: 0.6
          },
          hovertemplate: '<b>%{text}</b><br>Actual Category: %{customdata[0]}<br>Predicted Category: %{customdata[1]}<extra></extra>',
          text: data.map(d => `${d.site}<br>${d.date}`),
          customdata: data.map(d => {
            const categories = ['Low', 'Moderate', 'High', 'Extreme']
            return [
              categories[d.actual_category] || `Category ${d.actual_category}`,
              categories[d.predicted_category] || `Category ${d.predicted_category}`
            ]
          })
        })
      })
      
      return {
        data: traces,
        layout: {
          title: `Actual vs Predicted Category - ${config.forecast_model} (Accuracy = ${(accuracy * 100).toFixed(1)}%)`,
          xaxis: {
            title: 'Actual DA Category',
            tickmode: 'array',
            tickvals: [0, 1, 2, 3],
            ticktext: ['Low', 'Moderate', 'High', 'Extreme'],
            range: [-0.5, 3.5]
          },
          yaxis: {
            title: 'Predicted DA Category',
            tickmode: 'array',
            tickvals: [0, 1, 2, 3],
            ticktext: ['Low', 'Moderate', 'High', 'Extreme'],
            range: [-0.5, 3.5]
          },
          height: 500,
          showlegend: true,
          legend: { 
            x: 0.02, 
            y: 0.98,
            bgcolor: 'rgba(255,255,255,0.8)'
          }
        }
      }
    } else {
      // Regression scatter plot (existing code)
      const validData = filteredResults.results.filter(r => 
        r.actual_da !== null && r.predicted_da !== null
      )

      if (validData.length === 0) return null

      // Calculate range for diagonal line
      const allValues = [...validData.map(d => d.actual_da), ...validData.map(d => d.predicted_da)]
      const minVal = Math.min(...allValues)
      const maxVal = Math.max(...allValues)
      const range = [Math.max(0, minVal - 0.1), maxVal + 0.1]

      // Group data by site for different colors
      const siteGroups = {}
      validData.forEach(d => {
        if (!siteGroups[d.site]) {
          siteGroups[d.site] = []
        }
        siteGroups[d.site].push(d)
      })

      const traces = []

      // Add diagonal reference line
      traces.push({
        x: range,
        y: range,
        mode: 'lines',
        line: { color: 'red', width: 2, dash: 'dash' },
        name: 'Perfect Prediction',
        hovertemplate: 'Perfect prediction line<extra></extra>'
      })

      // Add scatter points for each site
      Object.entries(siteGroups).forEach(([site, data], index) => {
        traces.push({
          x: data.map(d => d.actual_da),
          y: data.map(d => d.predicted_da),
          mode: 'markers',
          type: 'scatter',
          name: site,
          marker: { 
            color: SITE_COLORS[index % SITE_COLORS.length],
            size: 8,
            opacity: 0.7
          },
          text: data.map(d => `${d.site}<br>${d.date}`),
          hovertemplate: '%{text}<br>Actual: %{x:.2f} μg/g<br>Predicted: %{y:.2f} μg/g<extra></extra>'
        })
      })

      return {
        data: traces,
        layout: {
          title: 'Model Performance: Actual vs Predicted DA Concentrations',
          xaxis: { 
            title: 'Actual DA Concentration (μg/g)',
            range: range
          },
          yaxis: { 
            title: 'Predicted DA Concentration (μg/g)',
            range: range
          },
          height: 500,
          showlegend: true,
          legend: { 
            x: 0.02, 
            y: 0.98,
            bgcolor: 'rgba(255,255,255,0.8)'
          }
        }
      }
    }
  }

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
                    <option value="ensemble">Ensemble - XGBoost + RF + Naive combined (Recommended)</option>
                    <option value="naive">Naive Baseline - Last known DA value (persistence)</option>
                    <option value="linear">Linear / Logistic - Interpretable linear models</option>
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
    return null
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

      {forecast && forecast.success && (
        <div className="space-y-6">
          {/* Summary */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
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

            </div>
          </div>

          {/* Ensemble Breakdown */}
          {forecast.ensemble && (
            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-medium text-purple-800 mb-4">🔬 Ensemble Model Breakdown</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
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
                <div className="bg-purple-50 p-4 rounded-lg text-center">
                  <p className="text-sm text-gray-600 mb-1">Naive Baseline</p>
                  <p className="text-xl font-bold text-purple-700">
                    {forecast.ensemble.naive_prediction?.toFixed(2)} μg/g
                  </p>
                  {forecast.ensemble.ensemble_weights && (
                    <p className="text-xs text-gray-500 mt-1">
                      weight: {(forecast.ensemble.ensemble_weights[2] * 100).toFixed(0)}%
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
            </div>
          )}

          {/* Level Range and Category Range Graphs - Match modular-forecast exactly */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Level Range Graph - Advanced Gradient Visualization */}
              {forecast.graphs && forecast.graphs.level_range && (
                <div>
                  {forecast.graphs.level_range.type === 'gradient_uncertainty' && forecast.graphs.level_range.gradient_plot ? (
                    // Use the advanced gradient plot from backend
                    <Plot
                      {...JSON.parse(forecast.graphs.level_range.gradient_plot)}
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
                      x: forecast.graphs.category_range.category_labels || ['Low (≤5)', 'Moderate (5-20]', 'High (20-40]', 'Extreme (>40)'],
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
                  x: (forecast.regression?.feature_importance || forecast.classification?.feature_importance)
                    ?.slice(0, 15)
                    ?.map(f => f.importance) || [],
                  y: (forecast.regression?.feature_importance || forecast.classification?.feature_importance)
                    ?.slice(0, 15)
                    ?.map(f => f.feature) || [],
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
    const timeSeriesData = createRetrospectiveTimeSeries()
    const scatterData = createRetrospectiveScatter()
    
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

        {/* Site filtering controls - Simple dropdown that filters existing results */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center gap-4">
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
              {sites.map(site => (
                <option key={site} value={site}>{site}</option>
              ))}
            </select>
            <span className="text-sm text-gray-600">
              Showing {filteredResults?.results?.length || 0} forecasts
            </span>
          </div>
        </div>

        {/* Summary statistics */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4">Performance Summary</h3>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-blue-50 p-4 rounded-lg text-center">
              <div className="text-2xl font-bold text-blue-600">{filteredResults?.summary?.total_forecasts || 0}</div>
              <div className="text-sm text-gray-600">Total Forecasts</div>
            </div>
            {filteredResults?.summary?.r2_score !== undefined && (
              <div className="bg-green-50 p-4 rounded-lg text-center">
                <div className="text-2xl font-bold text-green-600">{filteredResults.summary.r2_score.toFixed(3)}</div>
                <div className="text-sm text-gray-600">R² Score</div>
              </div>
            )}
            {filteredResults?.summary?.mae !== undefined && (
              <div className="bg-yellow-50 p-4 rounded-lg text-center">
                <div className="text-2xl font-bold text-yellow-600">{filteredResults.summary.mae.toFixed(2)}</div>
                <div className="text-sm text-gray-600">MAE (μg/g)</div>
              </div>
            )}
            {filteredResults?.summary?.f1_score !== undefined && (
              <div className="bg-orange-50 p-4 rounded-lg text-center">
                <div className="text-2xl font-bold text-orange-600">{filteredResults.summary.f1_score.toFixed(3)}</div>
                <div className="text-sm text-gray-600">F1 Score</div>
                <div className="text-xs text-gray-500 mt-1">Spike Detection (&gt;20 μg/g)</div>
              </div>
            )}
            {filteredResults?.summary?.accuracy !== undefined && (
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
        {filteredResults?.summary?.per_class_metrics && (
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
                  <li>• Scattered points suggest higher prediction uncertainty</li>
                  <li>• Color represents different monitoring sites</li>
                  <li>• R² closer to 1.0 indicates better model performance</li>
                </ul>
              </div>
            </div>
          </div>
        )}
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
