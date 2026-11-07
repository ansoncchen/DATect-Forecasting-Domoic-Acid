import React from 'react'
import { AMBER } from '../../lib/colors'

export default function Caution({ children }) {
  return (
    <p className="mt-3 text-xs text-gray-800 border-l-2 pl-2" style={{ borderColor: AMBER }}>
      {children}
    </p>
  )
}
