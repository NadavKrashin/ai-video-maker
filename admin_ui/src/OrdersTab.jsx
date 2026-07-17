// Orders tab: paid web orders (Firestore when configured, else Cloudinary
// folders). Each row shows the customer/package metadata the frontend saved
// and the order's pipeline state; one click ingests it as a project.
import React, { useCallback, useEffect, useState } from 'react';
import { api } from './api.js';
import { Btn, C, S } from './ui.jsx';

const statusChip = (o) => {
  if (o.project) return ['ingested', C.ok];
  if (o.ingesting) return ['ingesting…', C.run];
  if (o.status && o.status !== 'new') return [o.status, C.run];
  return ['new', C.accentSoft];
};

export default function OrdersTab({ onOpenProject, notify }) {
  const [orders, setOrders] = useState(null);
  const [busyRow, setBusyRow] = useState('');
  const [checking, setChecking] = useState(false);

  const refresh = useCallback(async () => {
    try { setOrders((await api.orders()).orders); }
    catch (e) { notify(`Orders failed: ${e.message}`); }
  }, [notify]);
  useEffect(() => { refresh(); }, [refresh]);

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

  if (orders === null) return <p style={{ color: C.muted }}>Loading orders…</p>;
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <h2 style={{ margin: 0, fontSize: 17 }}>Orders</h2>
        <Btn ghost busy={checking} onClick={checkNow}>Check for new orders now</Btn>
      </div>
      {orders.length === 0 && <p style={{ color: C.muted }}>No orders found.</p>}
      {orders.map((o) => {
        const [text, color] = statusChip(o);
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
            </div>
            <span style={S.chip(color)}>{text}</span>
            {o.project ? (
              <Btn ghost onClick={() => onOpenProject(o.project)}>open “{o.project}”</Btn>
            ) : !o.ingesting && (
              <Btn busy={busyRow === o.folder} disabled={!o.folder}
                title={o.folder ? '' : 'No photo folder recorded for this order yet'}
                onClick={() => ingest(o)}>Ingest + storyboard</Btn>
            )}
          </div>
        );
      })}
    </div>
  );
}
