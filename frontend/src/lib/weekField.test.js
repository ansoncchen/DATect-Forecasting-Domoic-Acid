import { describe, expect, it } from 'vitest'
import { resolveWeek, weekRangeNote } from './weekField'

const range = { min: '2003-01-06', max: '2023-12-25' }

describe('resolveWeek', () => {
  it('snaps a date to the Monday of its week', () => {
    expect(resolveWeek('2023-06-15', range)).toBe('2023-06-12')
    expect(resolveWeek('2023-06-18', range)).toBe('2023-06-12') // a Sunday ends its week
  })
  it('keeps a Monday', () => {
    expect(resolveWeek('2023-06-12', range)).toBe('2023-06-12')
  })
  it('accepts the first and last weeks, and any day of them', () => {
    expect(resolveWeek('2003-01-06', range)).toBe('2003-01-06')
    expect(resolveWeek('2023-12-25', range)).toBe('2023-12-25')
    expect(resolveWeek('2023-12-31', range)).toBe('2023-12-25')
  })
  it('rejects weeks outside the range', () => {
    expect(resolveWeek('2003-01-05', range)).toBeNull()
    expect(resolveWeek('2024-01-01', range)).toBeNull()
    expect(resolveWeek('2030-01-07', range)).toBeNull()
  })
  it('rejects the years a person passes through while typing', () => {
    expect(resolveWeek('0002-12-23', range)).toBeNull()
    expect(resolveWeek('0023-04-24', range)).toBeNull()
    expect(resolveWeek('0202-06-15', range)).toBeNull()
  })
  it('rejects a date that is missing or not a date', () => {
    expect(resolveWeek('', range)).toBeNull()
    expect(resolveWeek(undefined, range)).toBeNull()
    expect(resolveWeek('2023-13-45', range)).toBeNull()
    expect(resolveWeek('275760-09-13', range)).toBeNull()
  })
})

describe('weekRangeNote', () => {
  it('gives the valid range', () => {
    expect(weekRangeNote(range)).toBe('Not a forecast week. Choose a week from 6 Jan 2003 to 25 Dec 2023.')
  })
})
