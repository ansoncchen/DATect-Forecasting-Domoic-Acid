
export const plotConfig = {
  responsive: true,
  displayModeBar: true,
  displaylogo: false,
  modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d', 'autoScale2d'],
  toImageButtonOptions: {
    format: 'png',
    filename: 'datect_plot',
    scale: 10
  }
}

export const getPlotFilename = (prefix = 'datect') => {
  const date = new Date().toISOString().split('T')[0]
  return `${prefix}_${date}`
}