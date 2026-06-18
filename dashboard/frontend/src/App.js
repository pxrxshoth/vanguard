import React from 'react';
import Dashboard from './components/Dashboard';

function App() {
  return (
    <div className="App" style={{ backgroundColor: '#0f172a', minHeight: '100vh', color: 'white', fontFamily: 'Inter, sans-serif' }}>
      <header style={{ padding: '1.5rem', borderBottom: '1px solid #1e293b' }}>
        <h1 style={{ margin: 0, fontSize: '1.5rem', fontWeight: 'bold' }}>
          VANGUARD <span style={{ color: '#3b82f6', fontWeight: 'normal' }}>// Industrial Intelligence Platform</span>
        </h1>
      </header>
      <main style={{ padding: '2rem' }}>
        <Dashboard />
      </main>
    </div>
  );
}

export default App;
