import { parseISO, subMonths, subYears, addDays, format as fmtDate } from 'date-fns'
import { BLUE, INK, LEVEL_COLORS, RED } from './colors'
import { daysBetween, formatDate, formatPct, formatScore, formatValue } from './format'

const GAP_DAYS = 21
const END_PAD_DAYS = 12
const MONTHS_BACK = { '6m': 6 }
const YEARS_BACK = { '1y': 1, '2y': 2 }

const BAND_FILL = 'rgba(37,99,235,0.12)'
const LONE_BAR = 'rgba(37,99,235,0.35)'

const iso = (d) => fmtDate(d, 'yyyy-MM-dd')

/** Earliest ISO date a range shows, counted back from `end` (an ISO date); null for 'all'. */
function windowStart(range, end) {
  if (MONTHS_BACK[range]) return iso(subMonths(parseISO(end), MONTHS_BACK[range]))
  if (YEARS_BACK[range]) return iso(subYears(parseISO(end), YEARS_BACK[range]))
  return null
}

const hasRange = (r) => r.lo != null && r.hi != null
const isInside = (r) => r.actual >= r.lo && r.actual <= r.hi

/**
 * Runs of rows with a forecast and a range. A gap or a row without them ends a run, so
 * neither the line nor the range implies forecasts that were not made.
 */
function forecastSegments(rows) {
  const segments = []
  let current = null
  for (const r of rows) {
    if (!hasRange(r) || r.predicted == null) {
      current = null
      continue
    }
    if (!current || daysBetween(current[current.length - 1].date, r.date) > GAP_DAYS) {
      current = []
      segments.push(current)
    }
    current.push(r)
  }
  return segments
}

/** Plotly coordinates for several paths: each path's values, with a null between paths. */
const joinPaths = (paths) => paths.flatMap((path, i) => (i ? [null, ...path] : path))
const pick = (seg, key) => seg.map((r) => r[key])
const reversed = (list) => [...list].reverse()

function tip(r) {
  const lines = [formatDate(r.date)]
  lines.push(r.actual == null ? 'No measurement' : `Measured ${formatValue(r.actual)} µg/g`)
  if (r.predicted != null) {
    lines.push(`Forecast ${formatValue(r.predicted)}${hasRange(r) ? ` [${formatValue(r.lo)}–${formatValue(r.hi)}]` : ''}`)
  }
  if (r.actual != null && hasRange(r)) lines.push(`${isInside(r) ? 'Inside' : 'Outside'} the range`)
  if (r.probability != null) lines.push(`Classifier score ${formatScore(r.probability)}`)
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
  // A forecast that failed outright may not report its anchor; its own date then bounds the rows.
  const cutoff = forecast?.anchorDate ?? forecast?.date
  const shown = rows.filter(
    (r) => (start === null || r.date >= start) && (!forecast || showAfter || r.date <= cutoff),
  )

  const segments = forecastSegments(shown)
  const polygons = segments.filter((seg) => seg.length > 1)
  const lone = segments.filter((seg) => seg.length === 1).map(([r]) => r)

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
    {
      x: joinPaths(polygons.map((seg) => [...pick(seg, 'date'), ...reversed(pick(seg, 'date'))])),
      y: joinPaths(polygons.map((seg) => [...pick(seg, 'hi'), ...reversed(pick(seg, 'lo'))])),
      mode: 'lines',
      line: { width: 0 },
      fill: 'toself',
      fillcolor: BAND_FILL,
      name: `${formatPct(config.bandLevel)} range`,
      hoverinfo: 'skip',
      connectgaps: false,
    },
    {
      x: joinPaths(segments.map((seg) => pick(seg, 'date'))),
      y: joinPaths(segments.map((seg) => pick(seg, 'predicted'))),
      mode: 'lines',
      line: { color: BLUE, width: 1.5 },
      name: 'Forecast, made a week ahead',
      legendgroup: 'forecast',
      hoverinfo: 'skip',
      connectgaps: false,
    },
    // A one-row segment has no area to shade and no line to draw: a thin bar and a small marker instead.
    {
      x: pick(lone, 'date'),
      y: pick(lone, 'predicted'),
      mode: 'markers',
      name: 'Lone forecast',
      legendgroup: 'forecast',
      showlegend: false,
      hoverinfo: 'skip',
      marker: { color: BLUE, size: 4 },
      error_y: {
        type: 'data',
        symmetric: false,
        array: lone.map((r) => r.hi - r.predicted),
        arrayminus: lone.map((r) => r.predicted - r.lo),
        color: LONE_BAR,
        thickness: 2,
        width: 0,
      },
    },
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
  if (forecast?.anchorDate && showAfter) {
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
    yaxis: { domain: [0.18, 1], title: { text: 'DA (µg/g)', font: { size: 11 } }, gridcolor: '#f3f4f6', zeroline: false, range: [0, ymax], linecolor: '#d1d5db' },
    yaxis2: { domain: [0, 0.04], range: [0, 1], showticklabels: false, showgrid: false, zeroline: false },
    shapes,
    annotations: [{ xref: 'paper', x: 1, y: threshold, yanchor: 'bottom', xanchor: 'right', text: `${threshold} µg/g`, showarrow: false, font: { size: 10, color: RED } }],
  }
  return { traces, layout, empty: shown.length === 0 }
}

/**
 * What to say when the chart has no rows to show.
 * @param {import('../services/contract').HistoryRow[]} rows the site's whole history, oldest first
 * @param {{ date: string }} forecast
 * @param {string} since ISO date the history was requested from
 * @returns {string}
 */
export function noRowsNote(rows, forecast, since) {
  const first = rows.length ? rows[0].date : since
  return forecast.date < first
    ? `History on this chart starts in ${first.slice(0, 4)}.`
    : 'No measurements at this site in this range.'
}
