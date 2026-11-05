import { describe, it, expect } from 'vitest'
import { windowLabel } from './format'

describe('windowLabel', () => {
  it('uses an en dash and the inclusive end year', () => {
    expect(windowLabel('2022-01-01', '2024-01-01')).toBe('2022–2023')
    expect(windowLabel('2019-01-01', '2022-01-01')).toBe('2019–2021')
  })
})
