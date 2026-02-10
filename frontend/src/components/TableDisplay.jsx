import React, { useMemo, useState } from 'react';

// Displays table data and offers a small plotting UI (choose X/Y and request plot)
const TableDisplay = ({ data, onRequestPlot }) => {
  const { columns = [], rows = [] } = data || {};
  const [xCol, setXCol] = useState(columns[0] || '');
  const [yCol, setYCol] = useState(columns[1] || '');
  const [filters, setFilters] = useState(() => ({}));
  const [page, setPage] = useState(1);
  const rowsPerPage = 10;

  const columnStats = useMemo(() => {
    const stats = {};
    for (const col of columns) {
      const values = new Set();
      let isNumeric = true;
      for (const r of rows) {
        const v = r[col];
        if (v !== null && v !== undefined && v !== '') {
          values.add(v);
          if (isNaN(Number(v))) isNumeric = false;
        }
      }
      stats[col] = {
        uniqueValues: Array.from(values).sort(),
        isNumeric,
        count: values.size
      };
    }
    return stats;
  }, [columns, rows]);

  const numericColumns = useMemo(() => {
    return Object.keys(columnStats).filter(col => columnStats[col].isNumeric);
  }, [columnStats]);

  // 🔍 LOG: Track TableDisplay lifecycle
  console.log('📊 TableDisplay Component:', {
    hasData: !!data,
    columns: columns.length,
    rows: rows.length,
    willRender: !!(data && data.rows && data.rows.length > 0),
    data: data
  });

  if (!data || !data.rows || data.rows.length === 0) return null;

  const handlePlotClick = () => {
    if (!onRequestPlot) return;
    // pick defaults if not set
    const x = xCol || numericColumns[0] || columns[0];
    const y = yCol || numericColumns[1] || numericColumns[0] || columns[1] || columns[0];
    onRequestPlot(data, x, y);
  };

  const handleFilterChange = (col, value) => {
    setPage(1);
    setFilters(prev => ({ ...prev, [col]: value }));
  };

  const applyFilters = (rowsToFilter) => {
    if (!rowsToFilter || rowsToFilter.length === 0) return [];
    return rowsToFilter.filter(r => {
      for (const [col, val] of Object.entries(filters)) {
        if (!val) continue;
        const cell = r[col];
        if (cell === null || cell === undefined) return false;
        const s = String(cell).toLowerCase();
        if (!s.includes(String(val).toLowerCase())) return false;
      }
      return true;
    });
  };

  const filteredRows = applyFilters(rows);
  const totalPages = Math.max(1, Math.ceil(filteredRows.length / rowsPerPage));
  const visibleRows = filteredRows.slice((page - 1) * rowsPerPage, page * rowsPerPage);

  return (
    <div className="table-display">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <strong>Data Preview</strong>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <label style={{ fontSize: 12 }}>X:</label>
          <select value={xCol} onChange={(e) => setXCol(e.target.value)}>
            {numericColumns.length ? numericColumns.map(c => <option key={c} value={c}>{c}</option>) : columns.map(c => <option key={c} value={c}>{c}</option>)}
          </select>

          <label style={{ fontSize: 12 }}>Y:</label>
          <select value={yCol} onChange={(e) => setYCol(e.target.value)}>
            {numericColumns.length ? numericColumns.map(c => <option key={c} value={c}>{c}</option>) : columns.map(c => <option key={c} value={c}>{c}</option>)}
          </select>

          <button onClick={handlePlotClick} style={{ padding: '6px 10px' }}>Plot</button>
        </div>
      </div>

      <div className="table-scroll" style={{ maxHeight: 360, overflow: 'auto', border: '1px solid #eee' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              {columns.map((c) => (
                <th key={c} style={{ textAlign: 'left', padding: '6px 8px', borderBottom: '1px solid #f0f0f0' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span>{c}</span>
                    {columnStats[c].count < 20 ? (
                      <select
                        value={filters[c] || ''}
                        onChange={(e) => handleFilterChange(c, e.target.value)}
                        style={{ fontSize: 11, padding: '2px' }}
                      >
                        <option value="">All</option>
                        {columnStats[c].uniqueValues.map(v => (
                          <option key={v} value={v}>{v}</option>
                        ))}
                      </select>
                    ) : (
                      <input
                        placeholder="filter"
                        value={filters[c] || ''}
                        onChange={(e) => handleFilterChange(c, e.target.value)}
                        style={{ fontSize: 11, padding: '2px 6px', maxWidth: '60px' }}
                      />
                    )}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {visibleRows.map((r, idx) => (
              <tr key={idx}>
                {columns.map((c) => (
                  <td key={c} style={{ padding: '6px 8px', borderBottom: '1px solid #fafafa' }}>{String(r[c] ?? '')}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 8 }}>
        <div style={{ fontSize: 13 }}>{filteredRows.length} rows</div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <button onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1}>Prev</button>
          <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
            {Array.from({ length: totalPages }).map((_, i) => (
              <button key={i} onClick={() => setPage(i + 1)} style={{ fontWeight: page === i + 1 ? 'bold' : 'normal' }}>{i + 1}</button>
            ))}
          </div>
          <button onClick={() => setPage(p => Math.min(totalPages, p + 1))} disabled={page === totalPages}>Next</button>
        </div>
      </div>
    </div>
  );
};

export default TableDisplay;
