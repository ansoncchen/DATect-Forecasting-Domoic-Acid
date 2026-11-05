import { describe, it, expect } from 'vitest'
import { categoryScaleX } from './scale'
import { cats } from './testCategories'

describe('categoryScaleX', () => {
  it('gives each category a quarter and draws the open one to 2 x lo', () => {
    expect(categoryScaleX(0, cats)).toBe(0)
    expect(categoryScaleX(5, cats)).toBe(25)
    expect(categoryScaleX(20, cats)).toBe(50)
    expect(categoryScaleX(200, cats)).toBe(100)
  })
  it('interpolates inside a category', () => {
    expect(categoryScaleX(12.5, cats)).toBe(37.5)
    expect(categoryScaleX(60, cats)).toBe(87.5)
  })
})
