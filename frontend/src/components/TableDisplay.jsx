import React, { useState, useMemo } from "react";

export default function TableDisplay({ data }) {
  // Adapter for existing prop structure
  const rows = data?.rows || [];
  const columns = data?.columns || data?.headers || [];

  // Store selected filter per column
  const [columnFilters, setColumnFilters] = useState({});

  // Get unique values for a column
  const getUniqueValues = (col) => {
    // Filter out nulls/undefined for cleaner dropdowns
    // Use String(v) for key to ensure uniqueness for objects/mixed types if any
    const uniqueValues = [...new Set(rows.map(r => r[col]))];
    return uniqueValues.filter(v => v !== null && v !== undefined);
  };

  // Apply filters
  const filteredRows = useMemo(() => {
    return rows.filter(row =>
      Object.entries(columnFilters).every(([col, value]) => {
        if (!value) return true;
        // Compare as strings to handle numeric types matching select string values
        return String(row[col]) === String(value);
      })
    );
  }, [rows, columnFilters]);

  // Safety check
  if (!rows.length || !columns.length) {
    return null;
  }

  return (
    <div className="table-responsive" style={{ overflowX: 'auto', margin: '1em 0' }}>
      <table cellPadding="6" style={{ width: '100%', borderCollapse: 'collapse', border: '1px solid #e5e7eb', fontSize: '14px' }}>
        <thead>
          <tr style={{ backgroundColor: '#f9fafb', borderBottom: '2px solid #e5e7eb' }}>
            {columns.map(col => (
              <th key={col} style={{ padding: '8px', textAlign: 'left', minWidth: '150px' }}>
                <div style={{ marginBottom: '4px', fontWeight: '600', color: '#374151' }}>{col}</div>
                {/* Column filter dropdown */}
                <select
                  value={columnFilters[col] || ""}
                  onChange={(e) =>
                    setColumnFilters(prev => ({
                      ...prev,
                      [col]: e.target.value || null
                    }))
                  }
                  style={{
                    width: '100%',
                    padding: '4px 8px',
                    borderRadius: '4px',
                    border: '1px solid #d1d5db',
                    backgroundColor: 'white',
                    fontSize: '12px'
                  }}
                >
                  <option value="">All</option>
                  {getUniqueValues(col).map(v => (
                    <option key={String(v)} value={String(v)}>
                      {String(v)}
                    </option>
                  ))}
                </select>
              </th>
            ))}
          </tr>
        </thead>

        <tbody>
          {filteredRows.map((row, i) => (
            <tr key={i} style={{ borderBottom: '1px solid #e5e7eb', backgroundColor: i % 2 === 0 ? 'white' : '#f9fafb' }}>
              {columns.map(col => (
                <td key={col} style={{ padding: '8px', color: '#1f2937' }}>{row[col]}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {filteredRows.length === 0 && (
        <div style={{ padding: '20px', textAlign: 'center', color: '#666' }}>
          No matching records found.
        </div>
      )}
    </div>
  );
}