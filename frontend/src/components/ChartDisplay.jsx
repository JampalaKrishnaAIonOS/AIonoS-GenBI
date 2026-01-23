import React, { useRef, useState } from 'react';
import Plot from 'react-plotly.js';
import { Maximize2, DownloadCloud, FileText, X } from 'lucide-react';

const buildCSV = (figJson) => {
  // Combine traces into a wide CSV (x + each trace y)
  try {
    const traces = figJson.data || [];
    if (traces.length === 0) return '';

    // Determine x-axis values (use first trace x or index)
    const xVals = traces[0].x || traces[0].y.map((_, i) => i);
    const headers = ['x', ...traces.map(t => t.name || t.type || 'series')];
    const rows = xVals.map((x, i) => {
      const row = [x];
      for (const t of traces) {
        const y = (t.y && t.y[i] !== undefined) ? t.y[i] : '';
        row.push(y);
      }
      return row;
    });

    const csv = [headers.join(','), ...rows.map(r => r.map(v => JSON.stringify(v)).join(','))].join('\n');
    return csv;
  } catch (e) {
    return '';
  }
};

const downloadFile = (filename, content, mime = 'text/csv') => {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
};

const ChartDisplay = ({ chartData }) => {
  const [expanded, setExpanded] = useState(false);
  const plotRef = useRef(null);

  if (!chartData || !chartData.data) return (
    <div className="chart-container">No chart data</div>
  );

  const figJson = chartData.data; // { data: [...], layout: {...} }

  if (!figJson || !figJson.data) {
    return (
      <div className="chart-container">Invalid chart data
        <details>
          <summary>Raw chart JSON</summary>
          <pre style={{ maxHeight: 300, overflow: 'auto' }}>{JSON.stringify(chartData, null, 2)}</pre>
        </details>
      </div>
    );
  }

  const handleDownloadCSV = () => {
    const csv = buildCSV(figJson);
    if (!csv) return;
    const filename = (chartData.title || 'chart').replace(/[^a-z0-9_-]/gi, '_') + '.csv';
    downloadFile(filename, csv, 'text/csv');
  };

  const handleDownloadPNG = async () => {
    try {
      // Try to use Plotly's toImage via the plot DOM node
      const gd = plotRef.current && plotRef.current.el;
      if (gd && window.Plotly && window.Plotly.toImage) {
        const url = await window.Plotly.toImage(gd, { format: 'png', width: 1200, height: 600 });
        const a = document.createElement('a');
        a.href = url;
        a.download = (chartData.title || 'chart') + '.png';
        document.body.appendChild(a);
        a.click();
        a.remove();
      } else {
        // Fallback: render as SVG using plotly's relayout if possible
        console.warn('Plotly.toImage not available');
      }
    } catch (e) {
      console.error('PNG export failed', e);
    }
  };

  return (
    <div className="chart-container">
      <div className="chart-header">
        {chartData.title && <div className="chart-title">{chartData.title}</div>}
        <div className="chart-toolbar">
          <button className="chart-btn" title="Download CSV" onClick={handleDownloadCSV}><FileText size={16} /></button>
          <button className="chart-btn" title="Download PNG" onClick={handleDownloadPNG}><DownloadCloud size={16} /></button>
          <button className="chart-btn" title="Expand" onClick={() => setExpanded(true)}><Maximize2 size={16} /></button>
        </div>
      </div>

      <Plot
        ref={plotRef}
        data={figJson.data}
        layout={figJson.layout}
        config={{ responsive: true }}
        style={{ width: '100%', height: 360 }}
      />

      {expanded && (
        <div className="chart-modal">
          <div className="chart-modal-body">
            <div className="chart-modal-toolbar">
              <div className="chart-modal-title">{chartData.title}</div>
              <div>
                <button className="chart-btn" onClick={handleDownloadCSV} title="Download CSV"><FileText size={16} /></button>
                <button className="chart-btn" onClick={handleDownloadPNG} title="Download PNG"><DownloadCloud size={16} /></button>
                <button className="chart-btn" onClick={() => setExpanded(false)} title="Close"><X size={16} /></button>
              </div>
            </div>
            <div className="chart-modal-plot">
              <Plot
                data={figJson.data}
                layout={{ ...figJson.layout, autosize: true }}
                config={{ responsive: true }}
                style={{ width: '100%', height: '100%' }}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ChartDisplay;