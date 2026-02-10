import React from 'react';

const StreamDebugger = ({ events = [] }) => {
  return (
    <div style={{ padding: 12, fontSize: 12, color: '#444', background: '#f7f7f9', borderTop: '1px solid #eee' }}>
      <strong>Raw stream events (most recent first):</strong>
      <div style={{ maxHeight: 300, overflow: 'auto', marginTop: 8 }}>
        {events.map((e, i) => (
          <pre key={i} style={{ whiteSpace: 'pre-wrap', margin: 0 }}>{JSON.stringify(e)}</pre>
        ))}
      </div>
    </div>
  );
};

export default StreamDebugger;
