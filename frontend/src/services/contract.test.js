import { describe, it, expect } from 'vitest'
import {
  toForecast, toConfig, toSummary, toStat, toAllSites, toHistory, toTrackRecord,
} from './contract'
import configFixture from './__fixtures__/config.json'
import sitesFixture from './__fixtures__/sites.json'
import forecastFixture from './__fixtures__/forecast.json'
import forecastNoBandFixture from './__fixtures__/forecast_noband.json'
import forecastFailedFixture from './__fixtures__/forecast_failed.json'
import summaryFixture from './__fixtures__/summary.json'
import allSitesFixture from './__fixtures__/all_sites.json'
import historyFixture from './__fixtures__/history.json'
import trackRecordFixture from './__fixtures__/track_record.json'

describe('contract adapters', () => {
  it('maps a forecast with its band and alert parts', () => {
    const f = toForecast(forecastFixture)
    expect(f.site).toBe('Newport')
    expect(f.value).toBeCloseTo(forecastFixture.regression.predicted_da)
    expect(f.error).toBeNull()
    expect(f.band.lo).toBeLessThan(f.value)
    expect(f.band.hi).toBeGreaterThan(f.value)
    expect(f.categoryChances.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6)
    expect(typeof f.alert.byClassifier).toBe('boolean')
    expect(['Quiet', 'Watch', 'Warning', null]).toContain(f.alert.level)
    expect(f.lastMeasurement.date <= f.anchorDate).toBe(true)
    expect(f.recentMeasurements.length).toBeLessThanOrEqual(6)
    expect(f.recentMeasurements.at(-1)).toEqual(f.lastMeasurement)
  })

  it('maps parts, drivers and provenance', () => {
    const f = toForecast(forecastFixture)
    expect(f.parts).toEqual({ xgb: 5.040693283081055, rf: 5.661140230779626, naive: 6.3, weights: [1, 0, 0] })
    expect(f.drivers[0].importance).toBeGreaterThanOrEqual(f.drivers[1].importance)
    expect(f.provenance).toEqual({ source: 'computed', modelVersion: null, trainingSamples: 1093 })
    expect(f.bandPExceed).toBe(0.046)
  })

  it('leaves band fields null when the API sends no band', () => {
    const f = toForecast(forecastNoBandFixture)
    expect(f.band).toBeNull(); expect(f.categoryChances).toBeNull(); expect(f.bandPExceed).toBeNull()
    expect(f.bandReason).toBe('fewer than 30 earlier forecast errors')
    expect(f.alert.level).toBeNull()
    expect(f.alert.reason).toBe('already_at_or_above_threshold')
  })

  it('keeps measurements on a failed forecast', () => {
    const f = toForecast(forecastFailedFixture)
    expect(f.error).toBeTruthy(); expect(f.value).toBeNull(); expect(f.alert).toBeNull(); expect(f.band).toBeNull()
    expect(f.parts).toBeNull(); expect(f.drivers).toEqual([])
    expect(f.anchorDate).toBe('2002-12-30')
    expect(f.lastMeasurement).toBeTruthy(); expect(f.recentMeasurements).toHaveLength(6)
  })

  it('maps the API\'s bare failure, which has no anchor or measurements, to nulls', () => {
    const f = toForecast({ success: false, forecast_date: '2023-12-25', site: 'Newport', error: 'boom' })
    expect(f.anchorDate).toBeNull()
    expect(f.error).toBe('boom')
    expect(f.value).toBeNull(); expect(f.band).toBeNull(); expect(f.alert).toBeNull()
    expect(f.lastMeasurement).toBeNull(); expect(f.recentMeasurements).toEqual([])
  })

  it('keeps a missing classifier as null beside the regression-driven level', () => {
    const noClassifier = {
      ...forecastFixture,
      alert: { classifier_probability: null, by_classifier: false, by_regression: true, alert: true, level: 'Watch', level_reason: 'no_classifier' },
    }
    const f = toForecast(noClassifier)
    expect(f.alert).toMatchObject({ probability: null, level: 'Watch', reason: 'no_classifier', byRegression: true })
    expect(toForecast({ ...noClassifier, alert: { level: 'Quiet' } }).alert.probability).toBeNull()
  })

  it('builds config thresholds, levels, categories and site order', () => {
    const c = toConfig(configFixture, sitesFixture)
    expect(c.spikeThreshold).toBe(20); expect(c.regressionAlert).toBe(12); expect(c.classifierCutoff).toBe(0.1)
    expect(c.levels.find(l => l.name === 'Warning')).toMatchObject({ cutoff: 0.35, placeholder: false })
    expect(c.windows.holdout.label).toBe('2022–2023'); expect(c.windows.validation.label).toBe('2019–2021')
    expect(c.categories.map(k => k.lo)).toEqual([0, 5, 20, 40]); expect(c.categories.at(-1).hi).toBe(Infinity)
    expect(c.categories.map(k => k.label)).toEqual(['Low', 'Moderate', 'High', 'Extreme'])
    expect(c.sites[0]).toBe('Kalaloch'); expect(c.sites.at(-1)).toBe('Gold Beach')
    expect(c.dateRange).toEqual({ min: '2003-01-06', max: '2023-12-25' })
    expect(c.bandLevel).toBe(0.8)
  })

  it('maps the holdout reliability summary', () => {
    const r = toSummary(summaryFixture).reliability.holdout
    expect(r.overall.onsets).toEqual({ n: 27, caught: 19, quiet: 292, quietAlarmed: 87 })
    expect(r.overall.alarms).toEqual({ n: 106, followed: 19, falseBelow5: 21, false5To10: 18, falseNearMiss: 48 })
    expect(r.overall.levels.Watch).toEqual({ weeks: 68, followed: 9 })
    expect(r.overall.levels.Warning).toEqual({ weeks: 38, followed: 10 })
    expect(r.perSite.Newport.band).toMatchObject({ inside: 23, nBanded: 48 })
    expect(r.perSite.Newport.onsets).toEqual({ n: 6, caught: 5, quiet: 24, quietAlarmed: 8 })
    expect(r.perSite['Cannon Beach'].n).toBe(2); expect(r.perSite['Cannon Beach'].r2).toBeNull()
  })

  it('maps the headline from overall and per_site', () => {
    const s = toSummary(summaryFixture)
    expect(s.headline.overall.n).toBe(409)
    expect(s.headline.overall.r2.value).toBeCloseTo(0.516, 3)
    expect(s.headline.overall.r2.ci).toHaveLength(2)
    expect(s.headline.overall.spike.recall).toBeCloseTo(0.845, 3)
    expect(s.headline.perSite.Newport.nSpikes).toBe(16)
    expect(s.source).toEqual(summaryFixture.source)
  })

  it('normalizes r2 sent as a number or as [v, lo, hi]', () => {
    expect(toStat(0.516)).toEqual({ value: 0.516, ci: null })
    expect(toStat([0.516, 0.308, 0.654])).toEqual({ value: 0.516, ci: [0.308, 0.654] })
    expect(toStat([0.516])).toEqual({ value: 0.516, ci: null })
    expect(toStat(null)).toBeNull()
  })

  it('maps all-sites rows with the fields the endpoint lacks as null', () => {
    const rows = toAllSites(allSitesFixture)
    expect(rows).toHaveLength(10)
    expect(rows[0]).toMatchObject({ site: 'Kalaloch', date: '2023-12-25', error: null, parts: null, categoryChances: expect.any(Array) })
    expect(rows[0].recentMeasurements).toEqual([])
    expect(rows[0].drivers).toEqual([])
  })

  it('maps an all-sites row with no anchor to null', () => {
    const [row] = toAllSites({ date: '2023-12-25', sites: [{ site: 'Newport', error: 'boom' }] })
    expect(row.anchorDate).toBeNull()
  })

  it('maps history rows', () => {
    const rows = toHistory(historyFixture)
    expect(rows[0]).toMatchObject({ date: '2019-01-09', anchorDate: '2019-01-02', actual: 0, level: 'Quiet', fired: false })
    expect(rows[0].hi).toBeCloseTo(9.77, 2)
  })

  it('maps the track record', () => {
    const t = toTrackRecord(trackRecordFixture)
    expect(t.window).toMatchObject({ name: 'holdout', start: '2022-01-01', label: '2022–2023' })
    expect(t.overall.n).toBe(409)
    expect(t.overall.models.naive.r2).toBeCloseTo(0.225, 3)
    expect(t.perSite['Cannon Beach'].r2).toBeNull()
    expect(t.rows).toHaveLength(trackRecordFixture.rows.length)
    expect(t.rows[0]).toMatchObject({ site: 'Long Beach', actual: 1, naive: 1 })
  })
})
