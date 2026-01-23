import React from 'react';
import { Navigate } from 'react-router-dom';

export default function ProtectedRoute({ children }) {
  const authed = Boolean(localStorage.getItem('genbi_auth'));
  if (!authed) return <Navigate to="/login" replace />;
  return children;
}
