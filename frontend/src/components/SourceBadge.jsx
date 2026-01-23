import React from 'react';
import { FileSpreadsheet, Table } from 'lucide-react';

const SourceBadge = ({ source }) => {
  if (!source) return null;

  return (
    <div className="source-badge">
      <FileSpreadsheet size={14} />
      <span className="source-file">{source.file_name}</span>
      <Table size={14} />
      <span className="source-sheet">[{source.sheet_name}]</span>
      <span className="source-rows">rows {source.rows_used}</span>
    </div>
  );
};

export default SourceBadge;