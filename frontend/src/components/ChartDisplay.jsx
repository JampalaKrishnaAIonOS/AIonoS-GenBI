import React from 'react';
import Plot from 'react-plotly.js';

// Expects `chart` prop to be either a Plotly figure ({ data, layout })
// or a simple object like { type: 'scatter', x: [...], y: [...], layout: {...} }
const ChartDisplay = ({ chart }) => {
  if (!chart || (!chart.data && !chart.x)) return null;

  // If the backend already provided a full figure (chart.data may contain the full fig JSON)
  if (chart.data && chart.data.data && Array.isArray(chart.data.data)) {
    const fig = chart.data;
    return (
      <div style={{ width: '100%' }}>
        <Plot data={fig.data} layout={fig.layout || { autosize: true }} style={{ width: '100%' }} useResizeHandler />
      </div>
    );
  }

  if (chart.data && Array.isArray(chart.data)) {
    return (
      <div style={{ width: '100%' }}>
        <Plot data={chart.data} layout={chart.layout || { autosize: true }} style={{ width: '100%' }} useResizeHandler />
      </div>
    );
  }

  // Otherwise try to map a simple chart description
  if (chart.type && chart.x && chart.y) {
    const trace = { x: chart.x, y: chart.y, type: chart.type };
    return (
      <div style={{ width: '100%' }}>
        <Plot data={[trace]} layout={chart.layout || { autosize: true }} style={{ width: '100%' }} useResizeHandler />
      </div>
    );
  }

  // Unknown format: render JSON fallback
  return (
    <pre style={{ whiteSpace: 'pre-wrap', maxHeight: 300, overflow: 'auto', background: '#fafafa', padding: 8 }}>
      {JSON.stringify(chart, null, 2)}
    </pre>
  );
};

export default ChartDisplay;
