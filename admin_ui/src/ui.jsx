// Shared look & tiny building blocks for the panel.
import React from 'react';

export const C = {
  bg: '#12141A', panel: '#1B1E27', card: '#232734', border: '#333949',
  ink: '#E8EAF2', muted: '#9AA1B5', accent: '#D96A38', accentSoft: '#E8A13C',
  ok: '#4CAF7D', err: '#E05B5B', run: '#5B9DE0'
};

export const S = {
  page: { minHeight: '100vh', background: C.bg, color: C.ink, fontFamily: 'system-ui, sans-serif' },
  wrap: { maxWidth: 1100, margin: '0 auto', padding: '20px 16px 80px' },
  h1: { fontSize: 22, margin: '0 0 4px', color: C.accentSoft },
  card: { background: C.card, border: `1px solid ${C.border}`, borderRadius: 10, padding: 14, marginBottom: 12 },
  input: { background: C.panel, color: C.ink, border: `1px solid ${C.border}`, borderRadius: 6, padding: '8px 10px', fontSize: 14, width: '100%', boxSizing: 'border-box' },
  btn: { background: C.accent, color: '#fff', border: 'none', borderRadius: 6, padding: '8px 14px', fontSize: 14, cursor: 'pointer', fontWeight: 600 },
  btnGhost: { background: 'transparent', color: C.ink, border: `1px solid ${C.border}`, borderRadius: 6, padding: '7px 12px', fontSize: 13, cursor: 'pointer' },
  label: { fontSize: 12, color: C.muted, display: 'block', marginBottom: 4 },
  chip: (color) => ({ display: 'inline-block', padding: '2px 10px', borderRadius: 999, fontSize: 12, fontWeight: 700, background: `${color}22`, color, border: `1px solid ${color}55` }),
  err: { color: C.err, fontSize: 13, whiteSpace: 'pre-wrap' }
};

export const stepChip = (next) =>
  next === 'storyboard' ? ['needs storyboard', C.accentSoft]
    : next === 'render' ? ['needs render', C.run]
      : next === 'combine' ? ['needs combine', C.run]
        : ['complete', C.ok];

export function Btn({ ghost, busy, children, ...rest }) {
  return (
    <button {...rest} disabled={busy || rest.disabled}
      style={{ ...(ghost ? S.btnGhost : S.btn), opacity: busy || rest.disabled ? 0.55 : 1, ...(rest.style || {}) }}>
      {busy ? '…' : children}
    </button>
  );
}

export function Field({ label, children, style }) {
  return (
    <div style={style}>
      <label style={S.label}>{label}</label>
      {children}
    </div>
  );
}

export function Toggle({ label, checked, indeterminate, onChange, title }) {
  // Tri-state when `indeterminate` is allowed: null = use config default.
  return (
    <label title={title}
      style={{ display: 'inline-flex', gap: 6, alignItems: 'center', fontSize: 13, color: C.ink, cursor: 'pointer' }}>
      <input type="checkbox" checked={Boolean(checked)}
        ref={(el) => { if (el) el.indeterminate = Boolean(indeterminate); }}
        onChange={(e) => onChange(e.target.checked)} />
      {label}
    </label>
  );
}
