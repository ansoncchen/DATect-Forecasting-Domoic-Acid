import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { getPlotFilename } from './plotConfig'

describe('getPlotFilename', () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => vi.useRealTimers())

  it('uses the local date, not the UTC one', () => {
    vi.setSystemTime(new Date('2026-11-05T03:00:00Z')) // the evening of the 4th in Los Angeles
    expect(getPlotFilename('map')).toBe('map_2026-11-04')
  })
})
