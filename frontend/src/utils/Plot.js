// Plotly component with only the trace types this app draws (scatter, bar,
// heatmap, scattermapbox). The full plotly.js build also ships 3D/WebGL,
// geo and finance traces, roughly tripling the page's JavaScript.
import Plotly from 'plotly.js/lib/core'
import bar from 'plotly.js/lib/bar'
import heatmap from 'plotly.js/lib/heatmap'
import scattermapbox from 'plotly.js/lib/scattermapbox'
import createPlotlyComponent from 'react-plotly.js/factory'

Plotly.register([bar, heatmap, scattermapbox])

export default createPlotlyComponent(Plotly)
