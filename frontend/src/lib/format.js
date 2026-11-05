import { parseISO, subDays } from 'date-fns'

/**
 * Label for an evaluation window whose end is exclusive, with an en dash and the
 * inclusive end year: ('2022-01-01', '2024-01-01') -> '2022–2023'.
 * @param {string} start ISO date
 * @param {string} end ISO date, exclusive
 * @returns {string}
 */
export function windowLabel(start, end) {
  return `${parseISO(start).getFullYear()}–${subDays(parseISO(end), 1).getFullYear()}`
}
