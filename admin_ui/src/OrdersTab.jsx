// Orders tab: paid web orders (Firestore when configured, else Cloudinary
// folders). Each row shows the customer/package metadata the frontend saved
// and the order's pipeline state; one click ingests it as a project.
//
// Ingest outcomes are first-class: while a job runs the tab live-polls, and
// when it settles the row keeps saying what happened — green "ingested" or a
// red "ingest failed" with the error and its job log — instead of silently
// resetting the button.
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { api } from './api.js';
import { Btn, C, Modal, S } from './ui.jsx';

const statusChip = (o) => {
  if (o.project) return ['ingested', C.ok];
  if (o.ingesting) return ['ingesting…', C.run];
  if (o.ingest_state === 'failed') return ['ingest failed', C.err];
  if (o.ingest_state === 'cancelled') return ['ingest cancelled', C.muted];
  if (o.status && o.status !== 'new') return [o.status, C.run];
  return ['new', C.accentSoft];
};

export default function OrdersTab({ onOpenProject, notify }) {
  const [orders, setOrders] = useState(null);
  const [busyRow, setBusyRow] = useState('');
  const [checking, setChecking] = useState(false);
  const [logJob, setLogJob] = useState(null);
  const ingestingRef = useRef(new Set());

  const refresh = useCallback(async () => {
    try {
      const fresh = (await api.orders()).orders;
      // Toast the outcome of any ingest that settled since the last look.
      const still = new Set();
      for (const o of fresh) {
        if (o.ingesting) still.add(o.folder);
        else if (ingestingRef.current.has(o.folder)) {
          if (o.project) notify(`Order ingested as project "${o.project}"`);
          else if (o.ingest_state === 'failed') notify(`Ingest failed: ${o.ingest_error}`);
        }
      }
      ingestingRef.current = still;
      setOrders(fresh);
    } catch (e) { notify(`Orders failed: ${e.message}`); }
  }, [notify]);
  useEffect(() => { refresh(); }, [refresh]);

  // Live-poll while any ingest is running so the outcome shows up by itself.
  useEffect(() => {
    if (!(orders || []).some((o) => o.ingesting)) return undefined;
    const timer = setInterval(refresh, 3000);
    return () => clearInterval(timer);
  }, [orders, refresh]);

  const ingest = async (order) => {
    setBusyRow(order.folder);
    try {
      const res = await api.ingestOrder(order.folder, true);
      notify(`Ingesting as project "${res.project}" (storyboard follows)`);
      await refresh();
    } catch (e) { notify(`Ingest failed: ${e.message}`); }
    finally { setBusyRow(''); }
  };

  const checkNow = async () => {
    setChecking(true);
    try {
      const res = await api.pollWatcher();
      notify(res.enqueued.length
        ? `Watcher queued: ${res.enqueued.join(', ')}`
        : 'No new complete orders.');
      await refresh();
    } catch (e) { notify(`Watcher poll failed: ${e.message}`); }
    finally { setChecking(false); }
  };

  const showLog = async (jobId) => {
    try { setLogJob(await api.job(jobId)); }
    catch (e) { notify(`Log failed: ${e.message}`); }
  };

  if (orders === null) return <p style={{ color: C.muted }}>Loading orders…</p>;
  return (
    <div>
      {logJob && (
        <Modal title={`${logJob.command} — ${logJob.state}`} onClose={() => setLogJob(null)}>
          {logJob.error && <p style={S.err}>{logJob.error}</p>}
          <pre style={{ fontSize: 11, color: C.muted, overflowX: 'auto', maxHeight: 320 }}>
            {(logJob.log || []).join('\n') || '(no log lines)'}
          </pre>
        </Modal>
      )}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <h2 style={{ margin: 0, fontSize: 17 }}>Orders</h2>
        <Btn ghost busy={checking} onClick={checkNow}>Check for new orders now</Btn>
      </div>
      {orders.length === 0 && <p style={{ color: C.muted }}>No orders found.</p>}
      {orders.map((o) => {
        const [text, color] = statusChip(o);
        const failed = !o.project && o.ingest_state === 'failed';
        const details = [
          o.order_id,
          o.package_id && `package: ${o.package_id}`,
          o.music_mood && `mood: ${o.music_mood}`,
          o.phone, o.email,
          o.uploaded_at && `uploaded ${o.uploaded_at}`
        ].filter(Boolean).join(' · ');
        return (
          <div key={o.folder || o.order_id} style={{ ...S.card, display: 'flex', gap: 12, alignItems: 'center' }}>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontWeight: 700 }}>{o.customer || o.folder || o.order_id}</div>
              <div style={{ color: C.muted, fontSize: 12 }}>{details}</div>
              {o.blessing && (
                <div style={{ color: C.muted, fontSize: 12, marginTop: 2, fontStyle: 'italic' }}
                  title="The customer's blessing / dedication text">
                  “{o.blessing}”
                </div>
              )}
              {failed && (
                <div style={{ ...S.err, marginTop: 4 }} title={o.ingest_error}>
                  {o.ingest_error}
                </div>
              )}
            </div>
            <span style={S.chip(color)}>{text}</span>
            {failed && o.ingest_job && (
              <Btn ghost onClick={() => showLog(o.ingest_job)}>log</Btn>
            )}
            {o.project ? (
              <Btn ghost onClick={() => onOpenProject(o.project)}>open “{o.project}”</Btn>
            ) : !o.ingesting && (
              <Btn busy={Boolean(o.folder) && busyRow === o.folder} disabled={!o.folder}
                title={o.folder ? '' : 'No photo folder recorded for this order yet'}
                onClick={() => ingest(o)}>
                {failed ? 'Retry ingest' : 'Ingest + storyboard'}
              </Btn>
            )}
          </div>
        );
      })}
    </div>
  );
}
