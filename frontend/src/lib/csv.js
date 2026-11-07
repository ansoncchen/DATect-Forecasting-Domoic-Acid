const escape = (v) => {
  if (v === null || v === undefined) return ''
  const s = String(v)
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
}

/**
 * CSV text: a header line, then one line per row, values in `columns` order.
 * @param {object[]} rows
 * @param {string[]} columns
 * @param {Object<string, string>} [headers] header text for the columns that differ from their field name
 * @returns {string}
 */
export function toCsv(rows, columns, headers = {}) {
  const header = columns.map((c) => headers[c] ?? c).join(',')
  return [header, ...rows.map((r) => columns.map((c) => escape(r[c])).join(','))].join('\n')
}

export const TRACK_RECORD_COLUMNS = ['date', 'site', 'actual', 'ensemble', 'naive', 'linear', 'lo', 'hi', 'probability', 'fired']
export const TRACK_RECORD_HEADERS = { probability: 'classifier_score' }

/** Download `rows` as a CSV file named `filename`. */
export function exportCsv(rows, filename, columns = TRACK_RECORD_COLUMNS, headers = TRACK_RECORD_HEADERS) {
  const blob = new Blob([toCsv(rows, columns, headers)], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}
