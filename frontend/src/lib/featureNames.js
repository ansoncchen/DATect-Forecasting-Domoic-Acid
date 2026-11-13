const NAMES = {
  da: 'Domoic acid (the toxin)',
  last_observed_da_raw: 'Last measured DA',
  weeks_since_last_spike: 'Weeks since last spike',
  month: 'Month of year',
  'modis-sst': 'Sea surface temperature (MODIS)',
  'modis-chla': 'Chlorophyll-a (MODIS)',
  'modis-flr': 'Fluorescence line height (MODIS)',
  'modis-k490': 'Water clarity, K490 (MODIS)',
  'modis-par': 'Sunlight at the surface, PAR (MODIS)',
  'sst-anom': 'Sea surface temperature anomaly',
  'chla-anom': 'Chlorophyll-a anomaly',
  pdo: 'Pacific Decadal Oscillation',
  oni: 'El Niño index (ONI)',
  beuti: 'Coastal upwelling (BEUTI)',
  discharge: 'River discharge',
  pn: 'Pseudo-nitzschia count',
}

const OAD_SUFFIX = {
  '30day_max': ', 30-day maximum',
  '30day_mean': ', 30-day mean',
  '30day_trend': ', 30-day trend',
  lag1week: ', 1 week earlier',
  lag2week: ', 2 weeks earlier',
  zscore_doy: ', against the usual for the date',
}

const ORDINAL_BACK = { 1: 'Previous measurement (1 back)', 2: 'Measurement before last (2 back)' }

/**
 * Plain words for a model feature name; an unknown name comes back unchanged.
 * @param {string} name
 * @returns {string}
 */
export function featureLabel(name) {
  if (NAMES[name]) return NAMES[name]
  const prev = /^da_raw_prev_obs_(\d+)$/.exec(name)
  if (prev) return ORDINAL_BACK[prev[1]] ?? `Measurement ${prev[1]} back`
  const oad = /^oad_(overall_)?(score|valid_frac)(?:_(.+))?$/.exec(name)
  if (oad) {
    const [, overall, kind, rest] = oad
    const scope = overall ? 'all regions' : 'this site’s region'
    if (kind === 'valid_frac') return rest ? name : `Satellite coverage (${scope})`
    if (rest && !OAD_SUFFIX[rest]) return name
    return `Ocean anomaly (${scope})${rest ? OAD_SUFFIX[rest] : ''}`
  }
  return name
}
