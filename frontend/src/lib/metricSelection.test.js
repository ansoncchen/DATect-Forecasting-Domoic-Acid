import { describe, it, expect } from 'vitest'
import { METRICS, DEFAULT_METRICS, loadMetricSelection, saveMetricSelection } from './metricSelection'

const store = (data) => ({
  getItem: (k) => (k in data ? data[k] : null),
  setItem: (k, v) => {
    data[k] = v
  },
})
const throwingStore = {
  getItem() {
    throw new Error('blocked')
  },
  setItem() {
    throw new Error('blocked')
  },
}

describe('loadMetricSelection', () => {
  it('falls back to the defaults', () => {
    expect(loadMetricSelection(store({}))).toEqual(DEFAULT_METRICS)
    expect(loadMetricSelection(store({ 'datect.metrics': 'not json' }))).toEqual(DEFAULT_METRICS)
    expect(loadMetricSelection(store({ 'datect.metrics': '{"a":1}' }))).toEqual(DEFAULT_METRICS)
    expect(loadMetricSelection(throwingStore)).toEqual(DEFAULT_METRICS)
    expect(loadMetricSelection(null)).toEqual(DEFAULT_METRICS)
  })
  it('drops unknown keys and keeps the ranked order', () => {
    expect(loadMetricSelection(store({ 'datect.metrics': '["r2","bogus"]' }))).toEqual(['r2'])
    expect(loadMetricSelection(store({ 'datect.metrics': '["r2","alert"]' }))).toEqual(['alert', 'r2'])
  })
  it('keeps an empty selection', () => {
    expect(loadMetricSelection(store({ 'datect.metrics': '[]' }))).toEqual([])
  })
})

describe('saveMetricSelection', () => {
  it('round-trips and survives a blocked store', () => {
    const s = store({})
    saveMetricSelection(s, ['mae', 'prov'])
    expect(loadMetricSelection(s)).toEqual(['mae', 'prov'])
    expect(() => saveMetricSelection(throwingStore, ['mae'])).not.toThrow()
  })
})

describe('METRICS', () => {
  it('has ten tiles and the defaults are among them', () => {
    expect(METRICS).toHaveLength(10)
    expect(DEFAULT_METRICS.every((k) => METRICS.some(([m]) => m === k))).toBe(true)
  })
})
