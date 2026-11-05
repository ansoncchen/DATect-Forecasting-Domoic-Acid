import React from 'react'
import { Link, useLocation } from 'react-router-dom'

const NAV_ITEMS = [
  { path: '/', label: 'Forecast' },
  { path: '/track-record', label: 'Track record' },
  { path: '/data', label: 'Data explorer' },
  { path: '/about', label: 'About' },
]

const Navigation = () => {
  const { pathname } = useLocation()

  return (
    <nav className="bg-white border-b border-gray-200">
      <div className="max-w-6xl mx-auto px-4 py-2 flex flex-wrap items-center justify-between gap-y-1 sm:h-14">
        <div>
          <div className="text-lg font-bold text-blue-600 leading-tight">DATect</div>
          <div className="text-xs text-gray-500">Domoic Acid Forecasting</div>
        </div>
        <div className="flex flex-wrap -mx-3 sm:mx-0 sm:gap-4 text-sm">
          {NAV_ITEMS.map(({ path, label }) => (
            <Link
              key={path}
              to={path}
              className={`px-3 py-1.5 rounded ${
                pathname === path ? 'bg-gray-100 text-gray-900 font-medium' : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              {label}
            </Link>
          ))}
        </div>
      </div>
    </nav>
  )
}

export default Navigation
