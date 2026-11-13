import { describe, it, expect } from 'vitest'
import { CORE_VARIABLES, readableHeatmap, readableLabels, plainSiteMap } from './figures'
import { INK } from './colors'

const names = ['oni', 'da', 'oad_score', 'oad_score_lag1week']
const heatmap = {
  data: [{ type: 'heatmap', x: names, y: names, z: names.map((_, i) => names.map((__, j) => (i === j ? 1 : i + j / 10))) }],
  layout: { title: { text: 'Correlation Heatmap - Newport' }, annotations: [{ text: '1.00' }], xaxis: {}, yaxis: {} },
}

describe('readableHeatmap', () => {
  it('keeps the core variables, with plain names and the matching rows and columns', () => {
    const { data, layout } = readableHeatmap(heatmap)
    const [t] = data
    expect(t.x).toEqual(['El Niño index (ONI)', 'Domoic acid (the toxin)', 'Ocean anomaly (this site’s region)'])
    expect(t.y).toEqual(t.x)
    expect(t.z).toEqual([[1, 0.1, 0.2], [1, 1, 1.2], [2, 2.1, 1]])
    expect(layout.annotations).toEqual([])
    expect(layout.title.text).toBe('Which measurements rise and fall together, Newport')
  })
  it('shows every variable on request', () => {
    const [t] = readableHeatmap(heatmap, { all: true }).data
    expect(t.x).toHaveLength(4)
    expect(t.z[3][3]).toBe(1)
  })
  it('lists only columns the dataset has', () => {
    expect(CORE_VARIABLES).toContain('da')
    expect(new Set(CORE_VARIABLES).size).toBe(CORE_VARIABLES.length)
  })
})

describe('readableLabels', () => {
  it('renames input names and leaves numbers alone', () => {
    const { data } = readableLabels({ data: [{ x: ['pdo', 'beuti'], y: [0.4, 0.2] }], layout: {} })
    expect(data[0].x).toEqual(['Pacific Decadal Oscillation', 'Coastal upwelling (BEUTI)'])
    expect(data[0].y).toEqual([0.4, 0.2])
  })
})

describe('plainSiteMap', () => {
  it('draws every site in one colour under a neutral title', () => {
    const { data, layout } = plainSiteMap({ data: [{ marker: { color: ['red', 'green'], size: 9 } }], layout: {} })
    expect(data[0].marker).toEqual({ color: INK, size: 9 })
    expect(layout.title.text).toBe('Monitoring sites')
  })
})
