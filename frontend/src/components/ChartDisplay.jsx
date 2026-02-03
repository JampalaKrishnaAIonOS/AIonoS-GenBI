import React, { useState } from 'react';
import Plot from 'react-plotly.js';
import { Maximize2, X, Download } from 'lucide-react';

const ChartDisplay = ({ chart }) => {
  const [isModalOpen, setIsModalOpen] = useState(false);

  if (!chart || !chart.data) return null;

  // Extract Plotly data and layout from the chart object
  const plotData = chart.data.data || [];
  const plotLayout = chart.data.layout || {};

  return (
    <>
      <div className="chart-container">
        <div className="chart-header">
          <div className="chart-title">
            {chart.title || 'Visualization'}
          </div>
          <div className="chart-toolbar">
            <button
              className="chart-btn"
              onClick={() => {
                const gd = document.getElementById(`plot-${chart.title || 'viz'}`);
                if (gd && window.Plotly) {
                  window.Plotly.downloadImage(gd, {
                    format: 'png',
                    width: 1200,
                    height: 800,
                    filename: (chart.title || 'visualization').replace(/\s+/g, '_').toLowerCase()
                  });
                }
              }}
              title="Download as PNG"
            >
              <Download size={16} />
            </button>
            <button
              className="chart-btn"
              onClick={() => setIsModalOpen(true)}
              title="Expand chart"
            >
              <Maximize2 size={16} />
            </button>
          </div>
        </div>
        <Plot
          divId={`plot-${chart.title || 'viz'}`}
          data={plotData}
          layout={{
            ...plotLayout,
            autosize: true,
            margin: {
              l: chart.chart_type === 'barh' ? 120 : 50, // More space for horizontal bar labels
              r: 30,
              t: 50,
              b: chart.chart_type === 'bar' ? 100 : 50 // More space for vertical bar rotating labels if needed
            },
            font: { family: 'Inter, sans-serif' },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
          }}
          style={{ width: '100%', height: '400px' }}
          config={{
            responsive: true,
            displayModeBar: 'hover',
            displaylogo: false,
            modeBarButtonsToRemove: ['lasso2d', 'select2d']
          }}
        />
        {chart.rows_plotted && (
          <div style={{
            fontSize: '12px',
            color: '#666',
            marginTop: '8px',
            textAlign: 'center'
          }}>
            Showing {chart.rows_plotted} data points
          </div>
        )}
      </div>

      {isModalOpen && (
        <div className="chart-modal" onClick={() => setIsModalOpen(false)}>
          <div className="chart-modal-body" onClick={(e) => e.stopPropagation()}>
            <div className="chart-modal-toolbar">
              <div className="chart-modal-title">{chart.title || 'Visualization'}</div>
              <button
                className="chart-btn"
                onClick={() => setIsModalOpen(false)}
              >
                <X size={20} />
              </button>
            </div>
            <div style={{ padding: '20px', height: 'calc(100% - 60px)' }}>
              <Plot
                data={plotData}
                layout={{
                  ...plotLayout,
                  autosize: true,
                  height: 600
                }}
                style={{ width: '100%', height: '100%' }}
                config={{
                  responsive: true,
                  displayModeBar: true,
                  displaylogo: false
                }}
              />
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default ChartDisplay;