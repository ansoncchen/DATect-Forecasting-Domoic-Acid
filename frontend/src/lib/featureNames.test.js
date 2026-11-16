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
  it('names every feature a forecast reports (Copalis, 2023-06-12)', () => {
    const reported = ['oni', 'pdo', 'discharge', 'pn', 'beuti', 'modis-flr', 'modis-sst', 'sst-anom',
      'oad_score', 'oad_score_lag1week', 'oad_score_lag2week', 'oad_score_30day_mean', 'oad_score_30day_max',
      'oad_score_30day_trend', 'oad_score_zscore_doy', 'oad_overall_score', 'oad_overall_score_lag1week',
      'oad_overall_score_lag2week', 'oad_overall_score_30day_mean', 'oad_overall_score_30day_max',
      'oad_overall_score_30day_trend', 'oad_overall_score_zscore_doy', 'oad_valid_frac', 'oad_overall_valid_frac',
      'sin_day_of_year', 'cos_day_of_year', 'month', 'pn_log', 'last_observed_da_raw', 'weeks_since_last_spike',
      'raw_obs_roll_mean_4', 'raw_obs_roll_std_4', 'raw_obs_roll_max_4', 'raw_obs_roll_std_8', 'raw_obs_roll_max_8',
      'raw_obs_roll_std_12', 'raw_obs_roll_max_12', 'da_raw_prev_obs_1', 'da_raw_prev_obs_2',
      'da_raw_prev_obs_2_weeks_ago', 'da_raw_prev_obs_3', 'da_raw_prev_obs_3_weeks_ago', 'da_raw_prev_obs_4',
      'da_raw_prev_obs_diff_1_2']
    for (const n of reported) expect(featureLabel(n), n).not.toBe(n)
    expect(featureLabel('raw_obs_roll_max_4')).toBe('Highest measured DA, last 4 weeks')
    expect(featureLabel('raw_obs_roll_std_12')).toBe('Swings in measured DA, last 12 weeks')
    expect(featureLabel('da_raw_prev_obs_3_weeks_ago')).toBe('Weeks since measurement 3 back')
  })
  it('covers the other engine and parquet features', () => {
    for (const n of ['da_raw_prev_obs_1', 'month', 'weeks_since_last_spike', 'pdo', 'oni', 'discharge', 'pn', 'beuti', 'chla-anom', 'modis-chla', 'modis-flr', 'modis-k490', 'modis-par', 'sst-anom']) {
      expect(featureLabel(n), n).not.toBe(n)
    }
  })
})
