import React from 'react'

/**
 * A closed-by-default section: the summary line stays visible, the detail opens on click.
 * @param {{ summary: React.ReactNode, className?: string, summaryClassName?: string, children: React.ReactNode }} props
 */
export default function More({ summary, className = '', summaryClassName = 'text-xs text-gray-600', children }) {
  return (
    <details className={`group ${className}`}>
      <summary
        className={`cursor-pointer select-none list-none inline-flex items-center gap-1.5 hover:text-gray-900 [&::-webkit-details-marker]:hidden ${summaryClassName}`}
      >
        <span aria-hidden="true" className="inline-block text-[10px] transition-transform group-open:rotate-90">▶</span>
        {summary}
      </summary>
      <div className="mt-3">{children}</div>
    </details>
  )
}
