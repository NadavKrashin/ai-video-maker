// AI Video Maker admin panel — a browser UI over the pipeline's admin API
// (`pipeline.py serve`), which also serves this app's build at /.
import React, { useCallback, useRef, useState } from 'react';
import { api, getToken, hasSettings, saveSettings } from './api.js';
import { Btn, C, S } from './ui.jsx';
import OrdersTab from './OrdersTab.jsx';
import ProjectsTab from './ProjectsTab.jsx';
import ProjectDetail from './ProjectDetail.jsx';

function Settings({ onConnected }) {
  const [token, setToken] = useState(getToken());
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  const connect = async () => {
    setBusy(true); setError('');
    saveSettings(token);
    try {
      await api.health();
      await api.projects(); // exercises auth, not just reachability
      onConnected();
    } catch (e) {
      setError(`Could not connect: ${e.message}`);
    } finally { setBusy(false); }
  };

  return (
    <div style={{ ...S.card, maxWidth: 520, margin: '10vh auto' }}>
      <h2 style={{ marginTop: 0 }}>Connect to the pipeline</h2>
      <p style={{ color: C.muted, fontSize: 13 }}>
        The <code>ADMIN_API_TOKEN</code> value from the pipeline's
        <code> .env</code> — it authenticates every request from this panel.
      </p>
      <label style={S.label}>Admin token</label>
      <input style={S.input} value={token} onChange={(e) => setToken(e.target.value)}
        type="password" placeholder="ADMIN_API_TOKEN from .env"
        onKeyDown={(e) => e.key === 'Enter' && connect()} />
      {error && <p style={S.err}>{error}</p>}
      <div style={{ marginTop: 14 }}>
        <Btn busy={busy} onClick={connect}>Connect</Btn>
      </div>
    </div>
  );
}

export default function App() {
  const [connected, setConnected] = useState(hasSettings());
  const [tab, setTab] = useState('orders');
  const [project, setProject] = useState('');
  const [toast, setToast] = useState('');
  const toastTimer = useRef(null);

  const notify = useCallback((msg) => {
    setToast(msg);
    clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(''), 5000);
  }, []);

  if (!connected) {
    return <div style={S.page}><Settings onConnected={() => setConnected(true)} /></div>;
  }

  const openProject = (name) => { setProject(name); setTab('project'); };

  return (
    <div style={S.page}>
      <div style={S.wrap}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 14, marginBottom: 16, flexWrap: 'wrap' }}>
          <h1 style={S.h1}>AI Video Maker</h1>
          <nav style={{ display: 'flex', gap: 8 }}>
            <Btn ghost style={tab === 'orders' ? { borderColor: C.accent, color: C.accentSoft } : {}}
              onClick={() => { setTab('orders'); setProject(''); }}>Orders</Btn>
            <Btn ghost style={tab === 'projects' || tab === 'project' ? { borderColor: C.accent, color: C.accentSoft } : {}}
              onClick={() => { setTab('projects'); setProject(''); }}>Projects</Btn>
          </nav>
          <span style={{ flex: 1 }} />
          <Btn ghost onClick={() => setConnected(false)}>Settings</Btn>
        </div>
        {tab === 'orders' && <OrdersTab onOpenProject={openProject} notify={notify} />}
        {tab === 'projects' && <ProjectsTab onOpenProject={openProject} notify={notify} />}
        {tab === 'project' && project && (
          <ProjectDetail name={project} onBack={() => setTab('projects')} notify={notify} />
        )}
      </div>
      {toast && (
        <div style={{
          position: 'fixed', bottom: 18, left: '50%', transform: 'translateX(-50%)',
          background: C.panel, border: `1px solid ${C.border}`, color: C.ink,
          padding: '10px 18px', borderRadius: 8, fontSize: 14, maxWidth: '90vw'
        }}>{toast}</div>
      )}
    </div>
  );
}
