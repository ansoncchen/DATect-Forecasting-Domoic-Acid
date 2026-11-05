/** Test-only: Config.categories for bins [0, 5, 20, 40, inf). */
export const cats = [
  { lo: 0, hi: 5, label: 'Low' },
  { lo: 5, hi: 20, label: 'Moderate' },
  { lo: 20, hi: 40, label: 'High' },
  { lo: 40, hi: Infinity, label: 'Extreme' },
]
