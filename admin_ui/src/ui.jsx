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

// What an action costs, shown in every confirmation dialog: money honesty
// is the whole point of the modal.
export const COST = {
  free: { text: 'Free — no API credits', color: C.ok },
  openai: { text: 'Spends OpenAI credits', color: C.accentSoft },
  fal: { text: 'Spends fal.ai credits', color: C.accentSoft },
  both: { text: 'Spends OpenAI + fal.ai credits', color: C.err }
};

export function Modal({ title, onClose, children }) {
  return (
    <div onClick={onClose}
      style={{
        position: 'fixed', inset: 0, background: '#000A', zIndex: 50,
        display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 16
      }}>
      <div onClick={(e) => e.stopPropagation()}
        style={{
          background: C.card, border: `1px solid ${C.border}`, borderRadius: 12,
          padding: 18, width: 480, maxWidth: '100%', maxHeight: '85vh', overflowY: 'auto'
        }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
          <h3 style={{ margin: 0, fontSize: 16 }}>{title}</h3>
          <button onClick={onClose} aria-label="close"
            style={{ background: 'none', border: 'none', color: C.muted, cursor: 'pointer', fontSize: 16 }}>
            ✕
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}

// The verification step before anything that changes files or spends money:
// says exactly what will happen and what it costs, then asks.
export function ConfirmModal({ confirm, busy, onConfirm, onCancel }) {
  if (!confirm) return null;
  const cost = COST[confirm.cost || 'free'];
  return (
    <Modal title={confirm.title} onClose={onCancel}>
      <ul style={{ margin: '0 0 12px', paddingLeft: 18, color: C.ink, fontSize: 14, lineHeight: 1.55 }}>
        {confirm.lines.map((line, i) => <li key={i}>{line}</li>)}
      </ul>
      <div style={{ marginBottom: 14 }}>
        <span style={S.chip(cost.color)}>{cost.text}</span>
      </div>
      <div style={{ display: 'flex', gap: 10, justifyContent: 'flex-end' }}>
        <Btn ghost onClick={onCancel}>Cancel</Btn>
        <Btn busy={busy} style={confirm.danger ? { background: C.err } : {}} onClick={onConfirm}>
          {confirm.label || 'Confirm'}
        </Btn>
      </div>
    </Modal>
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
