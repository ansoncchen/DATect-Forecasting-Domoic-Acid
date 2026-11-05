export const METRICS = [
  ['alert', 'Alert track record'],
  ['coverage', 'Range reliability'],
  ['mae', 'Typical error'],
  ['cats', 'Category chances'],
  ['r2', 'Fit (R²)'],
  ['line', 'Spike detection by the forecast'],
  ['parts', 'Model parts'],
  ['drivers', "XGBoost's feature importances"],
  ['width', 'Range width'],
  ['prov', 'Data and model'],
]

export const DEFAULT_METRICS = ['alert', 'coverage', 'mae', 'cats']

const KEY = 'datect.metrics'

/**
 * Saved tile selection in METRICS order; unknown keys are dropped, and anything
 * unreadable gives the defaults.
 * @param {Storage | null | undefined} storage
 * @returns {string[]}
 */
export function loadMetricSelection(storage) {
  try {
    const saved = JSON.parse(storage.getItem(KEY))
    if (!Array.isArray(saved)) return DEFAULT_METRICS
    return METRICS.map(([k]) => k).filter((k) => saved.includes(k))
  } catch {
    return DEFAULT_METRICS
  }
}

/** @param {Storage | null | undefined} storage @param {string[]} keys */
export function saveMetricSelection(storage, keys) {
  try {
    storage.setItem(KEY, JSON.stringify(keys))
  } catch {
    // storage unavailable
  }
}
