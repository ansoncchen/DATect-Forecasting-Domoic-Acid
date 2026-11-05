import { describe, it, expect } from 'vitest'
import {
  categoryOf, reachesThreshold, levelKey, isStale, siteReliability, dataCurrencyNotice,
} from './risk'
import { cats } from './testCategories'

describe('categoryOf', () => {
  it('uses left-closed bins', () => {
    expect(categoryOf(4.99, cats).label).toBe('Low')
    expect(categoryOf(5, cats).label).toBe('Moderate')
    expect(categoryOf(20, cats).label).toBe('High')
    expect(categoryOf(40, cats).label).toBe('Extreme')
    expect(categoryOf(500, cats).label).toBe('Extreme')
  })
})

describe('reachesThreshold', () => {
  it('counts a range top exactly at the threshold', () => {
    expect(reachesThreshold({ lo: 2, hi: 20 }, 20)).toBe(true)
    expect(reachesThreshold({ lo: 2, hi: 19.9 }, 20)).toBe(false)
    expect(reachesThreshold(null, 20)).toBe(false)
  })
})

describe('levelKey', () => {
  it('maps the API level and never decides one', () => {
    expect(levelKey({ level: 'Watch', reason: null })).toBe('watch')
    expect(levelKey({ level: 'Warning', reason: null })).toBe('warning')
    expect(levelKey({ level: 'Quiet', reason: null })).toBe('quiet')
    expect(levelKey({ level: null, reason: 'already_at_or_above_threshold' })).toBe('already')
    expect(levelKey({ level: 'Watch', reason: 'already_at_or_above_threshold' })).toBe('already')
    expect(levelKey({ level: null, reason: null })).toBe('unavailable')
    expect(levelKey(null)).toBe('unavailable')
  })
})

describe('isStale', () => {
  it('is stale only beyond 21 days, or with no measurement', () => {
    expect(isStale('2023-10-30', '2023-11-20')).toBe(false)
    expect(isStale('2023-10-29', '2023-11-20')).toBe(true)
    expect(isStale(null, '2023-11-20')).toBe(true)
  })
})

describe('siteReliability', () => {
  it('flags weak below 0.70 of at least 10 rows', () => {
    expect(siteReliability({ band: { coverage: 0.479, nBanded: 48 } })).toBe('weak')
    expect(siteReliability({ band: { coverage: 0.7, nBanded: 40 } })).toBe('ok')
    expect(siteReliability({ band: { coverage: 0.5, nBanded: 10 } })).toBe('weak')
    expect(siteReliability({ band: { coverage: 0.5, nBanded: 9 } })).toBe('too-few')
    expect(siteReliability({ band: { coverage: 1, nBanded: 2 } })).toBe('too-few')
    expect(siteReliability(undefined)).toBe('too-few')
  })
})

describe('dataCurrencyNotice', () => {
  it('speaks only when the newest measurement is more than 30 days old', () => {
    expect(dataCurrencyNotice('2023-11-30', '2026-11-03')).toBe('Forecasts use data through 30 Nov 2023; they are not live.')
    expect(dataCurrencyNotice('2026-10-20', '2026-11-03')).toBeNull()
    expect(dataCurrencyNotice('2026-10-04', '2026-11-03')).toBeNull()
    expect(dataCurrencyNotice('2026-10-03', '2026-11-03')).not.toBeNull()
  })
})
