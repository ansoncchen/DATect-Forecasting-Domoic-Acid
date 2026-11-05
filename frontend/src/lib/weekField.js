import { format, isValid, parseISO, startOfWeek } from 'date-fns'
import { formatDate } from './format'

/**
 * The forecast week a typed date commits to: the Monday of its week, or null when the
 * date is invalid or that Monday is outside the range.
 * @param {string | undefined} value ISO date from the field; '' while it is incomplete
 * @param {{ min: string, max: string }} range ISO Mondays, inclusive
 * @returns {string | null}
 */
export function resolveWeek(value, { min, max }) {
  if (!value) return null
  const date = parseISO(value)
  if (!isValid(date)) return null
  const monday = format(startOfWeek(date, { weekStartsOn: 1 }), 'yyyy-MM-dd')
  return monday >= min && monday <= max ? monday : null
}

/**
 * @param {{ min: string, max: string }} range
 * @returns {string}
 */
export function weekRangeNote({ min, max }) {
  return `Not a forecast week. Choose a week from ${formatDate(min)} to ${formatDate(max)}.`
}
