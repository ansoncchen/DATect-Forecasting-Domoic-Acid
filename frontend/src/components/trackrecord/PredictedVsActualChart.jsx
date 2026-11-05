import React, { useMemo } from 'react'
import Plot from '../../utils/Plot'
import { BLUE, INK, RED } from '../../lib/colors'
import { formatDate, formatValue } from '../../lib/format'

const PLOT_CONFIG = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ['lasso2d', 'select2d'] }

function tip(r) {
  const range = r.lo != null && r.hi != null ? ` [${formatValue(r.lo)}–${formatValue(r.hi)}]` : ''
  return [r.site, formatDate(r.date), `Measured ${formatValue(r.actual)} µg/g`, `Forecast ${formatValue(r.ensemble)}${range}`].join('<br>')
}

/**
 * Measured against the ensemble's forecast, each point with its range as a vertical bar.
 * @param {{ rows: import('../../services/contract').TrackRecord['rows'],
 *   config: import('../../services/contract').Config }} props
 */
export default function PredictedVsActualChart({ rows, config }) {
  const { traces, layout } = useMemo(() => {
    const pts = rows.filter((r) => r.actual != null && r.ensemble != null)
    const t = config.spikeThreshold
    const top = Math.max(t * 1.3, ...pts.map((r) => Math.max(r.actual, r.ensemble, r.hi ?? 0))) * 1.05
    const axis = { range: [0, top], gridcolor: '#f3f4f6', zeroline: false, linecolor: '#d1d5db' }
    const line = (color, dash) => ({ color, width: 1, dash })
    return {
      traces: [
        { x: [0, top], y: [0, top], mode: 'lines', line: line('#9ca3af'), hoverinfo: 'skip', showlegend: false },
        {
          x: pts.map((r) => r.actual),
          y: pts.map((r) => r.ensemble),
          mode: 'markers',
          name: 'Measurement',
          marker: { color: INK, size: 6, opacity: 0.35 },
          error_y: {
            type: 'data',
            symmetric: false,
            array: pts.map((r) => (r.lo != null && r.hi != null ? r.hi - r.ensemble : null)),
            arrayminus: pts.map((r) => (r.lo != null && r.hi != null ? r.ensemble - r.lo : null)),
            color: BLUE,
            thickness: 1,
            width: 0,
          },
          text: pts.map(tip),
          hovertemplate: '%{text}<extra></extra>',
          showlegend: false,
        },
      ],
      layout: {
        margin: { l: 52, r: 12, t: 6, b: 44 },
        hovermode: 'closest',
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        font: { family: '-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif', size: 11, color: '#4b5563' },
        hoverlabel: { bgcolor: 'white', bordercolor: '#d1d5db', font: { color: INK } },
        xaxis: { ...axis, title: { text: 'Measured DA (µg/g)', font: { size: 11 } } },
        yaxis: { ...axis, title: { text: 'Forecast a week ahead (µg/g)', font: { size: 11 } } },
        shapes: [
          { type: 'line', xref: 'x', yref: 'paper', x0: t, x1: t, y0: 0, y1: 1, line: line(RED, 'dot') },
          { type: 'line', xref: 'paper', yref: 'y', x0: 0, x1: 1, y0: t, y1: t, line: line(RED, 'dot') },
        ],
        annotations: [{ xref: 'x', yref: 'paper', x: t, y: 1, xanchor: 'left', yanchor: 'top', text: `${t} µg/g`, showarrow: false, font: { size: 10, color: RED } }],
      },
    }
  }, [rows, config])

  return <Plot data={traces} layout={layout} config={PLOT_CONFIG} useResizeHandler style={{ width: '100%', height: 420 }} />
}
