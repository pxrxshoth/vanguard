import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { AlertTriangle, CheckCircle, Activity, Server } from 'lucide-react';

const Dashboard = () => {
  const [engines, setEngines] = useState([]);
  const [history, setHistory] = useState([]);

  useEffect(() => {
    // Connect to FastAPI WebSocket
    const ws = new WebSocket('ws://localhost:8000/ws/telemetry');
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      setEngines(prev => {
        const existing = prev.findIndex(e => e.unit_number === data.unit_number);
        if (existing >= 0) {
          const newEngines = [...prev];
          newEngines[existing] = data;
          return newEngines;
        }
        return [...prev, data];
      });

      setHistory(prev => {
        const newHistory = [...prev, { time: new Date().toLocaleTimeString(), ...data }];
        if (newHistory.length > 20) newHistory.shift();
        return newHistory;
      });
    };

    return () => ws.close();
  }, []);

  const totalAnomalies = engines.filter(e => e.is_anomaly).length;

  return (
    <div style={{ display: 'grid', gap: '2rem' }}>
      
      {/* Top Metrics Row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1rem' }}>
        <div style={{ background: '#1e293b', padding: '1.5rem', borderRadius: '0.5rem', border: '1px solid #334155' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
            <Server size={24} color="#3b82f6" />
            <h3 style={{ margin: 0, color: '#94a3b8' }}>Active Engines</h3>
          </div>
          <p style={{ margin: 0, fontSize: '2rem', fontWeight: 'bold' }}>{engines.length || 0}</p>
        </div>
        
        <div style={{ background: '#1e293b', padding: '1.5rem', borderRadius: '0.5rem', border: '1px solid #334155' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
            <Activity size={24} color="#10b981" />
            <h3 style={{ margin: 0, color: '#94a3b8' }}>Ingestion Rate</h3>
          </div>
          <p style={{ margin: 0, fontSize: '2rem', fontWeight: 'bold' }}>102k <span style={{fontSize:'1rem', color:'#64748b'}}>msg/s</span></p>
        </div>

        <div style={{ background: '#1e293b', padding: '1.5rem', borderRadius: '0.5rem', border: '1px solid #334155' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
            {totalAnomalies > 0 ? <AlertTriangle size={24} color="#ef4444" /> : <CheckCircle size={24} color="#10b981" />}
            <h3 style={{ margin: 0, color: '#94a3b8' }}>Detected Anomalies</h3>
          </div>
          <p style={{ margin: 0, fontSize: '2rem', fontWeight: 'bold', color: totalAnomalies > 0 ? '#ef4444' : '#10b981' }}>
            {totalAnomalies}
          </p>
        </div>
      </div>

      {/* Main Chart */}
      <div style={{ background: '#1e293b', padding: '1.5rem', borderRadius: '0.5rem', border: '1px solid #334155', height: '400px' }}>
        <h3 style={{ marginTop: 0, marginBottom: '1.5rem', color: '#f8fafc' }}>Fleet RUL Predictions vs Anomaly Score</h3>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={history}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="time" stroke="#94a3b8" />
            <YAxis yAxisId="left" stroke="#3b82f6" />
            <YAxis yAxisId="right" orientation="right" stroke="#ef4444" />
            <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155' }} />
            <Line yAxisId="left" type="monotone" dataKey="predicted_rul" stroke="#3b82f6" name="RUL (cycles)" strokeWidth={2} dot={false} />
            <Line yAxisId="right" type="monotone" dataKey="anomaly_score" stroke="#ef4444" name="Anomaly Score" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Engine Status Table */}
      <div style={{ background: '#1e293b', borderRadius: '0.5rem', border: '1px solid #334155', overflow: 'hidden' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
          <thead style={{ backgroundColor: '#0f172a' }}>
            <tr>
              <th style={{ padding: '1rem', borderBottom: '1px solid #334155' }}>Engine Unit</th>
              <th style={{ padding: '1rem', borderBottom: '1px solid #334155' }}>Current Cycle</th>
              <th style={{ padding: '1rem', borderBottom: '1px solid #334155' }}>Predicted RUL</th>
              <th style={{ padding: '1rem', borderBottom: '1px solid #334155' }}>Anomaly Status</th>
            </tr>
          </thead>
          <tbody>
            {engines.map(engine => (
              <tr key={engine.unit_number} style={{ borderBottom: '1px solid #334155' }}>
                <td style={{ padding: '1rem' }}>Unit #{engine.unit_number}</td>
                <td style={{ padding: '1rem' }}>{engine.cycle}</td>
                <td style={{ padding: '1rem', color: engine.predicted_rul < 30 ? '#ef4444' : '#f8fafc' }}>
                  {engine.predicted_rul.toFixed(2)}
                </td>
                <td style={{ padding: '1rem' }}>
                  {engine.is_anomaly ? 
                    <span style={{ background: '#7f1d1d', color: '#fca5a5', padding: '0.25rem 0.75rem', borderRadius: '9999px', fontSize: '0.875rem' }}>Critical</span> : 
                    <span style={{ background: '#064e3b', color: '#6ee7b7', padding: '0.25rem 0.75rem', borderRadius: '9999px', fontSize: '0.875rem' }}>Healthy</span>
                  }
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default Dashboard;
