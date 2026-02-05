import React, { useState, useMemo } from "react";

export default function TableDisplay({ data }) {
  // Adapter for existing prop structure
  const rows = data?.rows || [];
  const columns = data?.columns || data?.headers || [];

  // Store selected filter per column
  const [columnFilters, setColumnFilters] = useState({});
  const [currentPage, setCurrentPage] = useState(1);
  const rowsPerPage = 10;

  // Get unique values for a column
  const getUniqueValues = (col) => {
    // Filter out nulls/undefined for cleaner dropdowns
    // Use String(v) for key to ensure uniqueness for objects/mixed types if any
    const uniqueValues = [...new Set(rows.map(r => r[col]))];
    return uniqueValues.filter(v => v !== null && v !== undefined);
  };

  // Apply filters
  const filteredRows = useMemo(() => {
    // Reset to page 1 whenever filters change
    setCurrentPage(1);

    return rows.filter(row =>
      Object.entries(columnFilters).every(([col, value]) => {
        if (!value) return true;
        // Compare as strings to handle numeric types matching select string values
        return String(row[col]) === String(value);
      })
    );
  }, [rows, columnFilters]);

  // Pagination logic
  const totalPages = Math.ceil(filteredRows.length / rowsPerPage);
  const startIndex = (currentPage - 1) * rowsPerPage;
  const paginatedRows = filteredRows.slice(startIndex, startIndex + rowsPerPage);

  // Safety check
  if (!rows.length || !columns.length) {
    return null;
  }

  return (
    <div className="table-container" style={{ margin: '1em 0' }}>
      <div className="table-responsive" style={{ overflowX: 'auto', border: '1px solid #e5e7eb', borderRadius: '8px' }}>
        <table cellPadding="6" style={{ width: '100%', borderCollapse: 'collapse', fontSize: '14px' }}>
          <thead>
            <tr style={{ backgroundColor: '#f9fafb', borderBottom: '2px solid #e5e7eb' }}>
              {columns.map(col => (
                <th key={col} style={{ padding: '12px 8px', textAlign: 'left', minWidth: '150px' }}>
                  <div style={{ marginBottom: '8px', fontWeight: '600', color: '#111827', textTransform: 'capitalize' }}>
                    {col.replace(/_/g, ' ')}
                  </div>
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
                      padding: '6px 10px',
                      borderRadius: '6px',
                      border: '1px solid #d1d5db',
                      backgroundColor: 'white',
                      fontSize: '12px',
                      outline: 'none',
                      cursor: 'pointer'
                    }}
                  >
                    <option value="">All</option>
                    {getUniqueValues(col).sort().map(v => (
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
            {paginatedRows.map((row, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f3f4f6', backgroundColor: i % 2 === 0 ? 'white' : '#f9fafb' }}>
                {columns.map(col => (
                  <td key={col} style={{ padding: '10px 8px', color: '#374151' }}>
                    {typeof row[col] === 'number' ? row[col].toLocaleString() : row[col]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>

        {filteredRows.length === 0 && (
          <div style={{ padding: '30px', textAlign: 'center', color: '#6b7280' }}>
            No matching records found.
          </div>
        )}
      </div>

      {/* Pagination Controls */}
      {totalPages > 1 && (
        <div style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          padding: '16px 0',
          gap: '8px',
          flexWrap: 'wrap'
        }}>
          <button
            onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
            disabled={currentPage === 1}
            style={{
              padding: '6px 12px',
              borderRadius: '6px',
              border: '1px solid #d1d5db',
              backgroundColor: currentPage === 1 ? '#f3f4f6' : 'white',
              color: currentPage === 1 ? '#9ca3af' : '#374151',
              cursor: currentPage === 1 ? 'not-allowed' : 'pointer',
              fontSize: '13px'
            }}
          >
            Previous
          </button>

          {[...Array(totalPages)].map((_, i) => {
            const pageNum = i + 1;
            // Only show certain page numbers if there are too many
            if (totalPages > 7) {
              if (pageNum !== 1 && pageNum !== totalPages && (pageNum < currentPage - 1 || pageNum > currentPage + 1)) {
                if (pageNum === currentPage - 2 || pageNum === currentPage + 2) return <span key={pageNum}>...</span>;
                return null;
              }
            }

            return (
              <button
                key={pageNum}
                onClick={() => setCurrentPage(pageNum)}
                style={{
                  minWidth: '32px',
                  height: '32px',
                  borderRadius: '6px',
                  border: '1px solid',
                  borderColor: currentPage === pageNum ? '#2563eb' : '#d1d5db',
                  backgroundColor: currentPage === pageNum ? '#2563eb' : 'white',
                  color: currentPage === pageNum ? 'white' : '#374151',
                  cursor: 'pointer',
                  fontSize: '13px',
                  fontWeight: currentPage === pageNum ? '600' : '400'
                }}
              >
                {pageNum}
              </button>
            );
          })}

          <button
            onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
            disabled={currentPage === totalPages}
            style={{
              padding: '6px 12px',
              borderRadius: '6px',
              border: '1px solid #d1d5db',
              backgroundColor: currentPage === totalPages ? '#f3f4f6' : 'white',
              color: currentPage === totalPages ? '#9ca3af' : '#374151',
              cursor: currentPage === totalPages ? 'not-allowed' : 'pointer',
              fontSize: '13px'
            }}
          >
            Next
          </button>

          <div style={{ marginLeft: 'auto', fontSize: '13px', color: '#6b7280' }}>
            Showing {startIndex + 1}-{Math.min(startIndex + rowsPerPage, filteredRows.length)} of {filteredRows.length}
          </div>
        </div>
      )}
    </div>
  );
}
