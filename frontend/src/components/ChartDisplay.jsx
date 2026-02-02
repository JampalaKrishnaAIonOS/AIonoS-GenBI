import React, { useState } from 'react';
import Plot from 'react-plotly.js';
import { Maximize2, X } from 'lucide-react';

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
              onClick={() => setIsModalOpen(true)}
              title="Expand chart"
            >
              <Maximize2 size={16} />
            </button>
          </div>
        </div>
        <Plot
          data={plotData}
          layout={{
            ...plotLayout,
            autosize: true,
            margin: { l: 50, r: 50, t: 50, b: 50 }
          }}
          style={{ width: '100%', height: '400px' }}
          config={{
            responsive: true,
            displayModeBar: true,
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