import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import 'antd/dist/reset.css';
import './index.css';

import Login from './components/Login';
import ProtectedRoute from './components/ProtectedRoute';
import ChatInterface from './components/ChatInterface';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Navigate to="/genbi" replace />} />
        <Route path="/login" element={<Login />} />
        <Route path="/genbi" element={<ProtectedRoute><ChatInterface /></ProtectedRoute>} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;