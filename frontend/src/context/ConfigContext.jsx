import React, { createContext, useContext, useEffect, useState } from 'react'
import { getConfig } from '../services/datect'

const ConfigContext = createContext({ config: null, error: null })

export function ConfigProvider({ children }) {
  const [state, setState] = useState({ config: null, error: null })

  useEffect(() => {
    let cancelled = false
    getConfig()
      .then((config) => !cancelled && setState({ config, error: null }))
      .catch((e) => !cancelled && setState({ config: null, error: e?.message || String(e) }))
    return () => {
      cancelled = true
    }
  }, [])

  return <ConfigContext.Provider value={state}>{children}</ConfigContext.Provider>
}

// eslint-disable-next-line react-refresh/only-export-components
export function useConfig() {
  return useContext(ConfigContext)
}
