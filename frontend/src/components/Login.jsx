import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

export default function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const navigate = useNavigate();

  function handleSubmit(e) {
    e.preventDefault();
    setError('');

    // Check credentials as per request
    if (email === 'Admin@aionos.ai' && password === 'Admin@123') {
      localStorage.setItem('genbi_auth', 'true');
      navigate('/genbi');
    } else {
      setError('Invalid credentials. Please check your email and password.');
    }
  }

  return (
    <div className="login-root">
      {/* 50/50 split - Left Side */}
      <div className="login-left">
        <div className="login-left-inner">
          <h1 className="brand-main">GenBI</h1>
          <p className="brand-sub">AIonOS</p>
        </div>
      </div>

      {/* 50/50 split - Right Side */}
      <div className="login-right">
        <form className="login-form" onSubmit={handleSubmit}>
          <h2>Sign in</h2>

          <div className="form-group">
            <label className="form-label">Email Address</label>
            <input
              className="form-input"
              type="email"
              value={email}
              onChange={e => setEmail(e.target.value)}
              placeholder="Admin@aionos.ai"
              required
            />
          </div>

          <div className="form-group">
            <label className="form-label">Password</label>
            <input
              className="form-input"
              type="password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              placeholder="••••••••"
              required
            />
          </div>

          {error && <div className="form-error">{error}</div>}

          <button className="form-submit" type="submit">LOGIN</button>

          <p className="note">
            New user? Contact your administrator to request access.
          </p>
        </form>
      </div>
    </div>
  );
}
