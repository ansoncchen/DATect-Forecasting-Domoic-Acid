import { describe, expect, it } from 'vitest'
import { buildGroundTraces } from './groundTraces'

const row = (o = {}) => ({
  date: '2023-01-02', anchorDate: '2022-12-26', actual: 4, predicted: 5, lo: 1, hi: 10,
  probability: 0.1, fired: false, level: 'Quiet', ...o,
})
const rows = (dates) => dates.map((date) => row({ date }))
const fc = {
  site: 'Newport', date: '2023-11-27', anchorDate: '2023-11-20', error: null, value: 6,
  band: { lo: 2, hi: 14, nCalibration: 50 }, bandReason: null,
}
const cfg = {
  spikeThreshold: 20,
  levels: [
    { name: 'Quiet', cutoff: null, placeholder: false },
    { name: 'Watch', cutoff: 12, placeholder: false },
    { name: 'Warning', cutoff: 0.3, placeholder: false },
  ],
}
const allX = (traces) => traces.flatMap((t) => t.x ?? [])
const named = (traces, name) => traces.find((t) => t.name === name)

describe('buildGroundTraces', () => {
  it('breaks line and band across gaps longer than 21 days', () => {
    const { traces } = buildGroundTraces(rows(['2023-01-02', '2023-01-09', '2023-03-06']), fc, cfg, { range: 'all' })
    expect(named(traces, 'Forecast, made a week ahead').x).toEqual(['2023-01-02', '2023-01-09', null, '2023-03-06'])
  })

  it('draws measurements outside the range hollow', () => {
    const { traces } = buildGroundTraces([row({ actual: 30, lo: 1, hi: 10 }), row({ actual: 5, lo: 1, hi: 10 })], fc, cfg, { range: 'all' })
    expect(named(traces, 'Measurement outside the range').x).toHaveLength(1)
    expect(named(traces, 'Measurement outside the range').marker.color).toBe('white')
  })

  it('counts a measurement exactly on the range edge as inside', () => {
    const { traces } = buildGroundTraces([row({ actual: 10 }), row({ actual: 1 })], fc, cfg, { range: 'all' })
    expect(named(traces, 'Measurement outside the range').x).toHaveLength(0)
    expect(named(traces, 'Measurement').x).toHaveLength(2)
  })

  it('treats a missing range as no band, not as outside it', () => {
    const rs = [row({ date: '2023-01-02' }), row({ date: '2023-01-09', actual: 30, lo: null, hi: null }), row({ date: '2023-01-16' })]
    const { traces } = buildGroundTraces(rs, fc, cfg, { range: 'all' })
    expect(named(traces, 'Measurement outside the range').x).toHaveLength(0)
    expect(named(traces, 'Measurement').x).toEqual(['2023-01-02', '2023-01-09', '2023-01-16'])
    expect(named(traces, '80% range').x).toEqual(['2023-01-02', null, '2023-01-16'])
    expect(named(traces, 'Forecast, made a week ahead').y).toEqual([5, null, 5])
  })

  it('keeps rows after the anchor out unless showAfter', () => {
    const after = row({ date: '2023-12-04' })
    expect(allX(buildGroundTraces([after], fc, cfg, { range: 'all' }).traces)).not.toContain('2023-12-04')
    const shown = buildGroundTraces([after], fc, cfg, { range: 'all', showAfter: true })
    expect(allX(shown.traces)).toContain('2023-12-04')
    expect(shown.layout.shapes.some((s) => s.xref === 'x' && s.x0 === fc.anchorDate)).toBe(true)
  })

  it('counts ranges back from the forecast week', () => {
    const rs = rows(['2021-11-22', '2022-06-06', '2023-09-04'])
    const x = (range) => named(buildGroundTraces(rs, fc, cfg, { range }).traces, 'Measurement').x
    expect(x('6m')).toEqual(['2023-09-04'])
    expect(x('1y')).toEqual(['2023-09-04'])
    expect(x('6m')).not.toContain('2022-06-06')
    expect(x('2y')).toEqual(['2022-06-06', '2023-09-04'])
    expect(x('all')).toHaveLength(3)
  })

  it('adds next week with an asymmetric error bar, none without a band', () => {
    const nw = named(buildGroundTraces([], fc, cfg, { range: '2y' }).traces, 'Next week')
    expect(nw.error_y.array[0]).toBeCloseTo(fc.band.hi - fc.value)
    expect(nw.error_y.arrayminus[0]).toBeCloseTo(fc.value - fc.band.lo)
    expect(named(buildGroundTraces([], { ...fc, band: null }, cfg, { range: '2y' }).traces, 'Next week').error_y).toBeUndefined()
  })

  it('omits next week for a failed forecast and for no forecast at all', () => {
    expect(named(buildGroundTraces([], { ...fc, value: null, band: null }, cfg, { range: '2y' }).traces, 'Next week')).toBeUndefined()
    const { traces } = buildGroundTraces([row({ date: '2023-12-04' })], null, cfg, { range: 'all' })
    expect(named(traces, 'Next week')).toBeUndefined()
    expect(allX(traces)).toContain('2023-12-04')
  })

  it('without a forecast, ranges count back from the newest row', () => {
    const rs = rows(['2020-01-06', '2023-06-05', '2023-12-04'])
    const { traces } = buildGroundTraces(rs, null, cfg, { range: '6m' })
    expect(named(traces, 'Measurement').x).toEqual(['2023-06-05', '2023-12-04'])
  })

  it('puts ticks only for levels with a cutoff, one trace per level, on the strip axis', () => {
    const rs = [row({ date: '2023-01-02', level: 'Watch' }), row({ date: '2023-01-09', level: 'Warning' }), row({ date: '2023-01-16' })]
    const { traces } = buildGroundTraces(rs, fc, cfg, { range: 'all' })
    expect(named(traces, 'Quiet')).toBeUndefined()
    expect(named(traces, 'Watch').x).toEqual(['2023-01-02'])
    expect(named(traces, 'Warning').x).toEqual(['2023-01-09'])
    expect(named(traces, 'Warning').yaxis).toBe('y2')
  })

  it('labels the threshold line from config', () => {
    const { layout } = buildGroundTraces([], fc, cfg, { range: 'all' })
    expect(layout.annotations[0].text).toBe('20 µg/g')
    expect(layout.shapes[0].y0).toBe(20)
  })
})
