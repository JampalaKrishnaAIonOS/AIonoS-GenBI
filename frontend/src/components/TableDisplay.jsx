import React from 'react';
import { Table } from 'antd';

const TableDisplay = ({ data }) => {
  if (!data || !data.headers || !data.rows) return null;

  const columns = data.headers.map((header, idx) => ({
    title: header,
    dataIndex: idx,
    key: idx,
    render: (text) => {
      if (typeof text === 'number') {
        return text.toLocaleString();
      }
      return text;
    }
  }));

  const dataSource = data.rows.map((row, idx) => ({
    key: idx,
    ...row
  }));

  return (
    <div className="table-container">
      <Table
        columns={columns}
        dataSource={dataSource}
        pagination={{ pageSize: 10 }}
        size="small"
        scroll={{ x: true }}
      />
    </div>
  );
};

export default TableDisplay;