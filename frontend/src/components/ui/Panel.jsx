import React from 'react'

export default function Panel({ className = '', children }) {
  return <section className={`bg-white border border-gray-200 rounded-md ${className}`}>{children}</section>
}
