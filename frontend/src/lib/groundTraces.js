import { parseISO, subMonths, subYears, addDays, format as fmtDate } from 'date-fns'
import { BLUE, INK, LEVEL_COLORS, RED } from './colors'
import { daysBetween, formatDate, formatPct, formatValue } from './format'

const GAP_DAYS = 21
const END_PAD_DAYS = 12
const MONTHS_BACK = { '6m': 6 }
const YEARS_BACK = { '1y': 1, '2y': 2 }

const iso = (d) => fmtDate(d, 'yyyy-MM-dd')

/** Earliest ISO date a range shows, counted back from `end` (an ISO date); null for 'all'. */
function windowStart(range, end) {
  if (MONTHS_BACK[range]) return iso(subMonths(parseISO(end), MONTHS_BACK[range]))
  if (YEARS_BACK[range]) return iso(subYears(parseISO(end), YEARS_BACK[range]))
  return null
}

const hasRange = (r) => r.lo != null && r.hi != null
const isInside = (r) => r.actual >= r.lo && r.actual <= r.hi

function tip(r) {
  const lines = [formatDate(r.date)]
  lines.push(r.actual == null ? 'No measurement' : `Measured ${formatValue(r.actual)} µg/g`)
  if (r.predicted != null) {
    lines.push(`Forecast ${formatValue(r.predicted)}${hasRange(r) ? ` [${formatValue(r.lo)}–${formatValue(r.hi)}]` : ''}`)
  }
  if (r.actual != null && hasRange(r)) lines.push(`${isInside(r) ? 'Inside' : 'Outside'} the range`)
  if (r.probability != null) lines.push(`Classifier ${formatPct(r.probability)}`)
  return lines.join('<br>')
}

/**
 * Traces and layout for the ground-data chart: lab measurements, the forecast made a
 * week before each, its range, and level ticks. Pure; thresholds come from `config`.
 * @param {import('../services/contract').HistoryRow[]} rows oldest first
 * @param {import('../services/contract').Forecast | null} forecast null for no anchor and no next-week point
 * @param {import('../services/contract').Config} config
 * @param {{ range: '6m' | '1y' | '2y' | 'all', showAfter?: boolean }} options
 */
export function buildGroundTraces(rows, forecast, config, { range, showAfter = false }) {
  const end = forecast ? forecast.date : rows.length ? rows[rows.length - 1].date : null
  const start = end ? windowStart(range, end) : null
  const shown = rows.filter(
    (r) => (start === null || r.date >= start) && (!forecast || showAfter || r.date <= forecast.anchorDate),
  )

  // Break the line and band at gaps and at rows with no range, so they never imply forecasts that were not made.
  const xs = []
  const lo = []
  const hi = []
  const pred = []
  let prev = null
  const lineBreak = () => {
    if (xs.length && xs[xs.length - 1] !== null) {
      xs.push(null)
      lo.push(null)
      hi.push(null)
      pred.push(null)
    }
  }
  for (const r of shown) {
    if (!hasRange(r) || r.predicted == null) {
      lineBreak()
      continue
    }
    if (prev && daysBetween(prev, r.date) > GAP_DAYS) lineBreak()
    xs.push(r.date)
    lo.push(r.lo)
    hi.push(r.hi)
    pred.push(r.predicted)
    prev = r.date
  }

  const dots = (list, name, marker) => ({
    x: list.map((r) => r.date),
    y: list.map((r) => r.actual),
    mode: 'markers',
    name,
    marker,
    text: list.map(tip),
    hovertemplate: '%{text}<extra></extra>',
  })
  const measured = shown.filter((r) => r.actual != null)
  const outside = measured.filter((r) => hasRange(r) && !isInside(r))

  const traces = [
    { x: xs, y: hi, mode: 'lines', line: { width: 0 }, hoverinfo: 'skip', showlegend: false, connectgaps: false },
    { x: xs, y: lo, mode: 'lines', line: { width: 0 }, fill: 'tonexty', fillcolor: 'rgba(37,99,235,0.12)', name: '80% range', hoverinfo: 'skip', connectgaps: false },
    { x: xs, y: pred, mode: 'lines', line: { color: BLUE, width: 1.5 }, name: 'Forecast, made a week ahead', hoverinfo: 'skip', connectgaps: false },
    dots(measured.filter((r) => !outside.includes(r)), 'Measurement', { color: INK, size: 6 }),
    dots(outside, 'Measurement outside the range', { color: 'white', size: 7, line: { color: INK, width: 1.5 } }),
  ]

  if (forecast && forecast.value != null) {
    const band = forecast.band
    const next = {
      x: [forecast.date],
      y: [forecast.value],
      mode: 'markers',
      name: 'Next week',
      marker: { color: BLUE, size: 9, symbol: 'diamond' },
      hovertemplate: `Next week (${formatDate(forecast.date)})<br>Forecast ${formatValue(forecast.value)}${
        band ? ` [${formatValue(band.lo)}–${formatValue(band.hi)}]` : ''
      }<extra></extra>`,
    }
    if (band) {
      next.error_y = {
        type: 'data',
        symmetric: false,
        array: [band.hi - forecast.value],
        arrayminus: [forecast.value - band.lo],
        color: BLUE,
        thickness: 1.5,
        width: 5,
      }
    }
    traces.push(next)
  }

  for (const level of config.levels.filter((l) => l.cutoff != null)) {
    const flagged = shown.filter((r) => r.level === level.name)
    const color = LEVEL_COLORS[level.name] ?? INK
    traces.push({
      x: flagged.map((r) => r.date),
      y: flagged.map(() => 0.5),
      yaxis: 'y2',
      mode: 'markers',
      name: level.name,
      marker: { color, symbol: 'line-ns', size: 10, line: { color, width: 1.5 } },
      text: flagged.map(tip),
      hovertemplate: '%{text}<extra></extra>',
    })
  }

  const threshold = config.spikeThreshold
  const ymax =
    Math.max(
      threshold * 1.3,
      ...shown.map((r) => Math.max(r.actual ?? 0, r.hi ?? 0)),
      forecast?.band?.hi ?? 0,
      forecast?.value ?? 0,
    ) * 1.05

  const last = shown.length ? shown[shown.length - 1].date : null
  const first = shown.length ? shown[0].date : null
  const xEndBase = forecast && !showAfter ? forecast.date : [forecast?.date, last].filter(Boolean).sort().pop()
  const xRange = xEndBase ? [first ?? xEndBase, iso(addDays(parseISO(xEndBase), END_PAD_DAYS))] : undefined

  const shapes = [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: threshold, y1: threshold, line: { color: RED, width: 1, dash: 'dot' } }]
  if (forecast && showAfter) {
    shapes.push({ type: 'line', xref: 'x', yref: 'paper', x0: forecast.anchorDate, x1: forecast.anchorDate, y0: 0, y1: 1, line: { color: '#9ca3af', width: 1 } })
  }

  const layout = {
    margin: { l: 48, r: 12, t: 6, b: 36 },
    hovermode: 'closest',
    plot_bgcolor: 'white',
    paper_bgcolor: 'white',
    font: { family: '-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif', size: 11, color: '#4b5563' },
    hoverlabel: { bgcolor: 'white', bordercolor: '#d1d5db', font: { color: INK } },
    legend: { orientation: 'h', y: -0.13, x: 0, font: { size: 11 }, traceorder: 'normal' },
    xaxis: { showgrid: false, linecolor: '#d1d5db', ticks: 'outside', tickcolor: '#d1d5db', ...(xRange ? { range: xRange } : {}) },
    yaxis: { domain: [0.1, 1], title: { text: 'DA (µg/g)', font: { size: 11 } }, gridcolor: '#f3f4f6', zeroline: false, range: [0, ymax], linecolor: '#d1d5db' },
    yaxis2: { domain: [0, 0.05], range: [0, 1], showticklabels: false, showgrid: false, zeroline: false },
    shapes,
    annotations: [{ xref: 'paper', x: 1, y: threshold, yanchor: 'bottom', xanchor: 'right', text: `${threshold} µg/g`, showarrow: false, font: { size: 10, color: RED } }],
  }
  return { traces, layout }
}
