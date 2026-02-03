/**
 * Shared constants for the DATect frontend
 */

// Color palette for site visualization (consistent across all charts)
export const SITE_COLORS = [
  '#1f77b4', // blue
  '#ff7f0e', // orange
  '#2ca02c', // green
  '#d62728', // red
  '#9467bd', // purple
  '#8c564b', // brown
  '#e377c2', // pink
  '#7f7f7f', // gray
  '#bcbd22', // olive
  '#17becf'  // cyan
]

/**
 * Get color for a site based on its index
 * @param {number} index - Site index
 * @returns {string} Hex color code
 */
export const getSiteColor = (index) => SITE_COLORS[index % SITE_COLORS.length]

// DA risk category labels
export const DA_CATEGORIES = {
  0: { label: 'Low', color: '#22c55e', range: '0-5' },
  1: { label: 'Moderate', color: '#eab308', range: '5-20' },
  2: { label: 'High', color: '#f97316', range: '20-40' },
  3: { label: 'Extreme', color: '#dc2626', range: '>40' }
}

// Get category info by value
export const getCategoryInfo = (category) => DA_CATEGORIES[category] || DA_CATEGORIES[0]
