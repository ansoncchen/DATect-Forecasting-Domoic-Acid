import { featureLabel } from './featureNames'
import { INK } from './colors'

/**
 * The heatmap's default variables: toxin, algae, satellite, climate and river data, plus one
 * ocean-anomaly score per scope. The other 14 ocean-anomaly columns are lags and averages of
 * those two; "Show all variables" brings them back.
 */
export const CORE_VARIABLES = [
  'da', 'pn', 'modis-chla', 'chla-anom', 'modis-flr', 'modis-sst', 'sst-anom', 'modis-par', 'modis-k490',
  'beuti', 'pdo', 'oni', 'discharge', 'oad_score', 'oad_overall_score',
]

const relabel = (values) =>
  Array.isArray(values) ? values.map((v) => (typeof v === 'string' ? featureLabel(v) : v)) : values

/**
 * The correlation heatmap with plain variable names, no number printed in every cell (hover
 * shows it), and by default only CORE_VARIABLES.
 * @param {{ data: object[], layout: object }} plot
 * @param {{ all?: boolean }} [options]
 */
export function readableHeatmap(plot, { all = false } = {}) {
  const data = plot.data.map((trace) => {
    if (trace.type !== 'heatmap') return trace
    const names = trace.x
    const keep = names.map((n, i) => [n, i]).filter(([n]) => all || CORE_VARIABLES.includes(n)).map(([, i]) => i)
    const pick = (row) => keep.map((i) => row[i])
    return {
      ...trace,
      x: relabel(pick(names)),
      y: relabel(keep.map((i) => trace.y[i])),
      z: keep.map((i) => pick(trace.z[i])),
      hovertemplate: '%{y}<br>%{x}<br>r = %{z:.2f}<extra></extra>',
    }
  })
  const title = plot.layout?.title?.text ?? ''
  const site = title.includes(' - ') ? title.split(' - ').pop() : 'all sites'
  return {
    data,
    layout: {
      ...plot.layout,
      annotations: [],
      height: all ? 900 : 640,
      title: { text: `Which measurements rise and fall together, ${site}` },
      xaxis: { ...plot.layout.xaxis, automargin: true, tickangle: -40 },
      yaxis: { ...plot.layout.yaxis, automargin: true, autorange: 'reversed' },
      margin: { l: 20, r: 20, t: 60, b: 20 },
    },
  }
}

/**
 * Plain names on any axis that lists model inputs (the sensitivity charts' bars).
 * @param {{ data: object[], layout: object }} plot
 */
export function readableLabels(plot) {
  return {
    data: plot.data.map((trace) => ({ ...trace, x: relabel(trace.x), y: relabel(trace.y) })),
    layout: {
      ...plot.layout,
      xaxis: { ...plot.layout?.xaxis, automargin: true },
      yaxis: { ...plot.layout?.yaxis, automargin: true },
    },
  }
}

/**
 * The site map in one colour: the API colours sites by their last reading and titles them
 * "Risk Levels", against the dashboard's colour rules. The hover text stays.
 * @param {{ data: object[], layout: object }} plot
 */
export function plainSiteMap(plot) {
  return {
    data: plot.data.map((trace) => ({ ...trace, marker: { ...trace.marker, color: INK } })),
    layout: { ...plot.layout, title: { text: 'Monitoring sites' } },
  }
}
