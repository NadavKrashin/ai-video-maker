// Orders tab: paid web orders (Firestore when configured, else Cloudinary
// folders). Each row shows the customer/package metadata the frontend saved
// and the order's REAL pipeline position — not just "ingested or not" but
// how far its project has come (storyboard review, clips rendered, final
// ready), joined live from the project snapshot. Sortable by status.
//
// Ingest outcomes are first-class: while a job runs the tab live-polls, and
// when it settles the row keeps saying what happened — green "ingested" or a
// red "ingest failed" with the error and its job log — instead of silently
// resetting the button.
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { api } from './api.js';
import { Btn, C, Modal, S } from './ui.jsx';

// The one place that decides what an order's chip says. rank orders the
// "By status" sort: most actionable first, finished movies last.
function stage(o) {
  if (!o.project && o.ingest_state === 'failed') {
    return { text: 'ingest failed', color: C.err, rank: 0 };
  }
  if (o.ingesting) return { text: 'ingesting…', color: C.run, rank: 2 };
  if (o.active_job) {
    return { text: `${o.active_job.command} running…`, color: C.run, rank: 2 };
  }
  const p = o.progress;
  if (p) {
    if (p.final) return { text: 'final ready', color: C.ok, rank: 8 };
    if (p.clips_total > 0 && p.clips_rendered === p.clips_total) {
      return { text: 'needs combine', color: C.run, rank: 5 };
    }
    if (p.clips_rendered > 0) {
      return { text: `clips ${p.clips_rendered}/${p.clips_total}`, color: C.run, rank: 4 };
    }
    if (p.clips_total > 0) return { text: 'review storyboard', color: C.accentSoft, rank: 3 };
    return { text: 'needs storyboard', color: C.accentSoft, rank: 3 };
  }
  if (o.project) return { text: 'ingested', color: C.ok, rank: 3 };
  if (o.status && o.status !== 'new') return { text: o.status, color: C.run, rank: 9 };
  return { text: 'new', color: C.accentSoft, rank: 1 };
}

const progressLine = (p) => p && [
  `${p.photos} photos`,
  p.clips_total > 0
    ? `${p.clips_rendered}/${p.clips_total} clips`
    : 'no storyboard yet',
  p.clips_stale > 0 && `${p.clips_stale} outdated`,
  p.placeholders > 0 && `${p.placeholders} generic prompt(s)`,
  p.final && 'final ready'
].filter(Boolean).join(' · ');

export default function OrdersTab({ onOpenProject, notify }) {
  const [orders, setOrders] = useState(null);
  const [busyRow, setBusyRow] = useState('');
  const [checking, setChecking] = useState(false);
  const [logJob, setLogJob] = useState(null);
  const [sort, setSort] = useState('date');
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

  // Live-poll while anything is running so outcomes show up by themselves.
  useEffect(() => {
    if (!(orders || []).some((o) => o.ingesting || o.active_job)) return undefined;
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

  const shown = sort === 'status'
    ? [...orders].sort((a, b) => stage(a).rank - stage(b).rank)
    : orders; // the server already returns newest first

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
      <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 8 }}>
        <h2 style={{ margin: 0, fontSize: 17, flex: 1 }}>Orders</h2>
        <select style={{ ...S.input, width: 170 }} value={sort}
          title="Sort the orders"
          onChange={(e) => setSort(e.target.value)}>
          <option value="date">Newest first</option>
          <option value="status">By status (todo first)</option>
        </select>
        <Btn ghost busy={checking} onClick={checkNow}>Check for new orders now</Btn>
      </div>
      {shown.length === 0 && <p style={{ color: C.muted }}>No orders found.</p>}
      {shown.map((o) => {
        const { text, color } = stage(o);
        const failed = !o.project && o.ingest_state === 'failed';
        const details = [
          o.order_id,
          o.package_id && `package: ${o.package_id}`,
          o.music_mood && `mood: ${o.music_mood}`,
          o.phone, o.email,
          o.uploaded_at && `uploaded ${o.uploaded_at}`
        ].filter(Boolean).join(' · ');
        const progress = progressLine(o.progress);
        return (
          <div key={o.folder || o.order_id} style={{ ...S.card, display: 'flex', gap: 12, alignItems: 'center' }}>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontWeight: 700 }}>{o.customer || o.folder || o.order_id}</div>
              <div style={{ color: C.muted, fontSize: 12 }}>{details}</div>
              {progress && (
                <div style={{ color: C.ink, fontSize: 12, marginTop: 2 }}>{progress}</div>
              )}
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
