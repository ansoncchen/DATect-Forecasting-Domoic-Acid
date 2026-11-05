import React from 'react'
import { format } from 'date-fns'
import { useConfig } from '../../context/ConfigContext'
import { dataCurrencyNotice } from '../../lib/risk'

export default function DataCurrencyBanner() {
  const { config } = useConfig()
  if (!config) return null
  const notice = dataCurrencyNotice(config.newestMeasurement, format(new Date(), 'yyyy-MM-dd'))
  return notice ? <p className="text-xs text-gray-500">{notice}</p> : null
}
