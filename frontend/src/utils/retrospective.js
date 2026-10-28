// Pure helpers for the Dashboard's retrospective view: metrics, plots and CSV export.
// They depend only on their arguments, so they can be tested and reused outside the page.
import { getSiteColor } from './constants'

export const exportResultsCSV = (filtered, siteFilter, task) => {
  if (!filtered?.results?.length) return
  const rows = filtered.results
  // Build union of all keys we want, in a stable order
  const baseCols = ['date', 'site', 'predicted_da', 'actual_da', 'predicted_category',
                    'actual_category', 'spike_alert', 'spike_probability',
                    'forecast_window']
  const cols = baseCols.filter(c => rows.some(r => r[c] !== undefined))
  const esc = v => v === null || v === undefined ? ''
    : (typeof v === 'string' && (v.includes(',') || v.includes('"')))
      ? `"${v.replace(/"/g, '""')}"`
      : String(v)
  const csv = [cols.join(','), ...rows.map(r => cols.map(c => esc(r[c])).join(','))].join('\n')
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  const ts = new Date().toISOString().slice(0, 10)
  a.href = url
  a.download = `datect_${task || 'results'}_${siteFilter || 'all'}_${ts}.csv`
  document.body.appendChild(a); a.click(); document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

export const parsePlotlyJson = (plotJson) => {
  if (!plotJson) return null
  if (typeof plotJson === 'object') return plotJson
  try {
    return JSON.parse(plotJson)
  } catch (err) {
    console.warn('Failed to parse plot JSON', err)
    return null
  }
}

export const calculateClassificationMetrics = (results) => {
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

export const calculateRegressionMetrics = (results) => {
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
  
  // Spike detection metrics — match backend thresholds (reaching a threshold counts):
  //   actual >= 20 µg/g (SPIKE_THRESHOLD)
  //   predicted >= 12 µg/g (SPIKE_REGRESSION_ALERT_THRESHOLD)
  const SPIKE_ACTUAL = 20
  const SPIKE_PREDICTED = 12
  const actualSpikes = actuals.map(val => val >= SPIKE_ACTUAL ? 1 : 0)
  const predictedSpikes = predictions.map(val => val >= SPIKE_PREDICTED ? 1 : 0)

  const truePositives = actualSpikes.reduce((sum, actual, i) =>
    sum + (actual === 1 && predictedSpikes[i] === 1 ? 1 : 0), 0
  )
  const falsePositives = actualSpikes.reduce((sum, actual, i) =>
    sum + (actual === 0 && predictedSpikes[i] === 1 ? 1 : 0), 0
  )
  const falseNegatives = actualSpikes.reduce((sum, actual, i) =>
    sum + (actual === 1 && predictedSpikes[i] === 0 ? 1 : 0), 0
  )

  const nSpikeEvents = actualSpikes.reduce((a, b) => a + b, 0)
  const nAlerts = predictedSpikes.reduce((a, b) => a + b, 0)
  const precision = truePositives + falsePositives > 0 ? truePositives / (truePositives + falsePositives) : 0
  const recall = truePositives + falseNegatives > 0 ? truePositives / (truePositives + falseNegatives) : 0
  const f1Score = precision + recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0
  // F2 weights recall 4x precision (matches backend fbeta_score(beta=2.0))
  const f2Score = precision + recall > 0
    ? (5 * precision * recall) / (4 * precision + recall)
    : 0

  return {
    regression_forecasts: validRegression.length,
    r2_score: r2,
    mae,
    f1_score: f1Score,
    // New per-filter spike metrics (override pooled server values)
    spike_recall: recall,
    spike_precision: precision,
    spike_f2: f2Score,
    n_spike_events: nSpikeEvents,
    n_alerts_fired: nAlerts,
    spike_threshold_actual: SPIKE_ACTUAL,
    spike_threshold_predicted: SPIKE_PREDICTED,
  }
}

// Time series of actual vs predicted for the retrospective results
export const buildRetrospectiveTimeSeries = (filteredResults, config) => {
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
      
      const siteColor = getSiteColor(siteIndex)
      
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
      
      const siteColor = getSiteColor(siteIndex)
      
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

      // Spike alert markers (where classifier flagged spike risk)
      const spikeAlerts = siteData.filter(d => d.spike_alert === true)
      if (spikeAlerts.length > 0) {
        traces.push({
          x: spikeAlerts.map(d => d.date),
          y: spikeAlerts.map(d => d.predicted_da),
          mode: 'markers',
          name: `${site} - Spike Alert`,
          marker: {
            size: 12,
            symbol: 'triangle-up',
            color: 'rgba(220, 38, 38, 0.8)',
            line: { color: 'darkred', width: 1 }
          },
          text: spikeAlerts.map(d => `Spike prob: ${(d.spike_probability * 100).toFixed(1)}%`),
          hovertemplate: '%{text}<br>Predicted: %{y:.1f} μg/g<br>Date: %{x}<extra></extra>'
        })
      }
    })

    // Regulatory threshold line at DA = 20
    traces.push({
      x: [results[0]?.date, results[results.length - 1]?.date],
      y: [20, 20],
      mode: 'lines',
      name: 'Regulatory Threshold (20 μg/g)',
      line: { color: 'rgba(220, 38, 38, 0.4)', width: 2, dash: 'dot' },
      hoverinfo: 'skip'
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

// Scatter plot of predicted vs actual for the retrospective results
export const buildRetrospectiveScatter = (filteredResults, config) => {
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
          color: getSiteColor(index),
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

    // Group by TIME WINDOW (not site) so users see which dots are the
    // unbiased 2022-2023 holdout vs training-period predictions.
    // Site stays in hover text.
    const windowOf = (dateStr) => {
      const d = new Date(dateStr)
      if (d < new Date('2019-01-01')) return 'pre-2019 (training context)'
      if (d < new Date('2022-01-01')) return 'val 2019-2022 (tuning)'
      return 'holdout 2022-2023 (untouched)'
    }
    const windowColor = {
      'pre-2019 (training context)':  '#94a3b8',  // gray
      'val 2019-2022 (tuning)':       '#fbbf24',  // amber
      'holdout 2022-2023 (untouched)':'#10b981',  // green
    }
    const windowGroups = {}
    validData.forEach(d => {
      const w = windowOf(d.date)
      if (!windowGroups[w]) windowGroups[w] = []
      windowGroups[w].push(d)
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
    // Add regulatory + alert threshold lines (operational reference)
    traces.push({
      x: [20, 20], y: range, mode: 'lines',
      line: { color: '#ef4444', width: 1, dash: 'dot' },
      name: 'Spike threshold (actual >20)',
      hoverinfo: 'skip'
    })
    traces.push({
      x: range, y: [12, 12], mode: 'lines',
      line: { color: '#f59e0b', width: 1, dash: 'dot' },
      name: 'Alert threshold (predicted >12)',
      hoverinfo: 'skip'
    })

    // Add scatter points colored by time window (preserves site in hover)
    const orderedWindows = ['pre-2019 (training context)', 'val 2019-2022 (tuning)', 'holdout 2022-2023 (untouched)']
    orderedWindows.forEach(window => {
      const data = windowGroups[window]
      if (!data || data.length === 0) return
      traces.push({
        x: data.map(d => d.actual_da),
        y: data.map(d => d.predicted_da),
        mode: 'markers',
        type: 'scatter',
        name: `${window} (n=${data.length})`,
        marker: {
          color: windowColor[window],
          size: 8,
          opacity: window.startsWith('holdout') ? 0.85 : 0.45,
          line: window.startsWith('holdout') ? { color: '#047857', width: 1 } : undefined,
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
