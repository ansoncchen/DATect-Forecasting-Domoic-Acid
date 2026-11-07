/**
 * Position 0..100 on a scale where each category takes a quarter of the width;
 * the open last category is drawn out to twice its lower bound.
 * @param {number} v
 * @param {import('../services/contract').Config['categories']} categories
 * @returns {number}
 */
export function categoryScaleX(v, categories) {
  const width = 100 / categories.length
  const i = categories.findIndex((c) => v < c.hi)
  const k = i === -1 ? categories.length - 1 : i
  const { lo, hi } = categories[k]
  const top = Number.isFinite(hi) ? hi : 2 * lo
  const frac = Math.min((v - lo) / (top - lo), 1)
  return width * (k + frac)
}
