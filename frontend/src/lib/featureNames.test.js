import { describe, it, expect } from 'vitest'
import { featureLabel } from './featureNames'

describe('featureLabel', () => {
  it('names the examples in the plan', () => {
    expect(featureLabel('oad_overall_score_30day_mean')).toBe('Ocean anomaly (all regions), 30-day mean')
    expect(featureLabel('last_observed_da_raw')).toBe('Last measured DA')
    expect(featureLabel('da_raw_prev_obs_2')).toBe('Measurement before last (2 back)')
    expect(featureLabel('modis-sst')).toBe('Sea surface temperature (MODIS)')
    expect(featureLabel('something_new')).toBe('something_new')
  })
  it('covers every OAD feature by rule', () => {
    for (const scope of ['oad_', 'oad_overall_']) {
      for (const stem of ['score', 'score_30day_max', 'score_30day_trend', 'score_lag1week', 'score_lag2week', 'score_zscore_doy', 'valid_frac']) {
        const name = scope + stem
        expect(featureLabel(name), name).not.toBe(name)
      }
    }
    expect(featureLabel('oad_score')).toBe('Ocean anomaly (this site’s region)')
    expect(featureLabel('oad_valid_frac')).toBe('Satellite coverage (this site’s region)')
    expect(featureLabel('oad_score_bogus')).toBe('oad_score_bogus')
  })
  it('covers the other engine and parquet features', () => {
    for (const n of ['da_raw_prev_obs_1', 'month', 'weeks_since_last_spike', 'pdo', 'oni', 'discharge', 'pn', 'beuti', 'chla-anom', 'modis-chla', 'modis-flr', 'modis-k490', 'modis-par', 'sst-anom']) {
      expect(featureLabel(n), n).not.toBe(n)
    }
  })
})
