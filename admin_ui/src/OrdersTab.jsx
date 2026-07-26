// Orders tab: paid web orders (Firestore when configured, else Cloudinary
// folders). Each row shows the customer/package metadata the frontend saved
// and the order's REAL pipeline position, joined live from the project
// snapshot. Searchable, filterable by status, sortable.
//
// Ingest outcomes are first-class: while a job runs the tab live-polls, and
// when it settles the row keeps saying what happened — green "ingested" or a
// red "ingest failed" with the error and its job log — instead of silently
// resetting the button.
import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Badge, Button, Card, Group, Modal, ScrollArea, Select, Stack, Text, TextInput, Title
} from '@mantine/core';
import { api } from './api.js';
import { notify } from './ui.jsx';

// The one place that decides what an order's chip says. rank orders the
// "By status" sort (most actionable first, finished movies last); group is
// what the status filter buckets on.
function stage(o) {
  if (!o.project && o.ingest_state === 'failed') {
    return { text: 'ingest failed', color: 'red', rank: 0, group: 'failed' };
  }
  if (o.ingesting) return { text: 'ingesting…', color: 'blue', rank: 2, group: 'running' };
  if (o.active_job) {
    return { text: `${o.active_job.command} running…`, color: 'blue', rank: 2, group: 'running' };
  }
  const p = o.progress;
  if (p) {
    if (p.final) return { text: 'final ready', color: 'green', rank: 8, group: 'final' };
    if (p.clips_total > 0 && p.clips_rendered === p.clips_total) {
      return { text: 'needs combine', color: 'blue', rank: 5, group: 'todo' };
    }
    if (p.clips_rendered > 0) {
      return { text: `clips ${p.clips_rendered}/${p.clips_total}`, color: 'blue', rank: 4, group: 'todo' };
    }
    if (p.clips_total > 0) {
      return { text: 'review storyboard', color: 'yellow', rank: 3, group: 'todo' };
    }
    return { text: 'needs storyboard', color: 'yellow', rank: 3, group: 'todo' };
  }
  if (o.project) return { text: 'ingested', color: 'green', rank: 3, group: 'todo' };
  if (o.status && o.status !== 'new') return { text: o.status, color: 'blue', rank: 9, group: 'other' };
  return { text: 'new', color: 'yellow', rank: 1, group: 'new' };
}

// 'failed' is both its own bucket and part of "needs action".
const inGroup = (o, group) =>
  group === 'all'
  || stage(o).group === group
  || (group === 'todo' && stage(o).group === 'failed');

const matchesQuery = (o, q) =>
  !q || [
    o.customer, o.order_id, o.folder, o.project, o.email, o.phone,
    o.package_id, o.music_mood, o.blessing
  ].some((v) => (v || '').toLowerCase().includes(q));

const progressLine = (p) => p && [
  `${p.photos} photos`,
  p.clips_total > 0
    ? `${p.clips_rendered}/${p.clips_total} clips`
    : 'no storyboard yet',
  p.clips_stale > 0 && `${p.clips_stale} outdated`,
  p.placeholders > 0 && `${p.placeholders} generic prompt(s)`,
  p.final && 'final ready'
].filter(Boolean).join(' · ');

export default function OrdersTab({ onOpenProject }) {
  const [orders, setOrders] = useState(null);
  const [busyRow, setBusyRow] = useState('');
  const [checking, setChecking] = useState(false);
  const [logJob, setLogJob] = useState(null);
  const [sort, setSort] = useState('date');
  const [filter, setFilter] = useState('all');
  const [query, setQuery] = useState('');
  const ingestingRef = useRef(new Set());

  const refresh = useCallback(async () => {
    try {
      const fresh = (await api.orders()).orders;
      // Toast the outcome of any ingest that settled since the last look.
      const still = new Set();
      for (const o of fresh) {
        if (o.ingesting) still.add(o.folder);
        else if (ingestingRef.current.has(o.folder)) {
          if (o.project) notify(`Order ingested as project "${o.project}"`, 'green');
          else if (o.ingest_state === 'failed') notify(`Ingest failed: ${o.ingest_error}`, 'red');
        }
      }
      ingestingRef.current = still;
      setOrders(fresh);
    } catch (e) { notify(`Orders failed: ${e.message}`, 'red'); }
  }, []);
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
    } catch (e) { notify(`Ingest failed: ${e.message}`, 'red'); }
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
    } catch (e) { notify(`Watcher poll failed: ${e.message}`, 'red'); }
    finally { setChecking(false); }
  };

  const showLog = async (jobId) => {
    try { setLogJob(await api.job(jobId)); }
    catch (e) { notify(`Log failed: ${e.message}`, 'red'); }
  };

  if (orders === null) return <Text c="dimmed">Loading orders…</Text>;

  const count = (group) => orders.filter((o) => inGroup(o, group)).length;
  const q = query.trim().toLowerCase();
  const shown = orders
    .filter((o) => inGroup(o, filter) && matchesQuery(o, q));
  if (sort === 'status') shown.sort((a, b) => stage(a).rank - stage(b).rank);
  // else: keep the server's newest-first order

  return (
    <Stack gap="sm">
      <Modal opened={Boolean(logJob)} onClose={() => setLogJob(null)} centered size="xl"
        title={logJob ? `${logJob.command} — ${logJob.state}` : ''}>
        {logJob?.error && <Text c="red" size="sm" mb="sm">{logJob.error}</Text>}
        <ScrollArea.Autosize mah={360}>
          <Text component="pre" size="xs" c="dimmed" style={{ margin: 0 }}>
            {(logJob?.log || []).join('\n') || '(no log lines)'}
          </Text>
        </ScrollArea.Autosize>
      </Modal>

      <Group>
        <Title order={4} style={{ flex: 1 }}>Orders</Title>
        <Button variant="default" size="xs" loading={checking} onClick={checkNow}>
          Check for new orders now
        </Button>
      </Group>
      <Group>
        <TextInput style={{ flex: 1 }} miw={200} value={query}
          placeholder="Search name, order id, email, phone, package…"
          onChange={(e) => setQuery(e.target.value)} />
        <Select w={190} value={filter} onChange={(v) => setFilter(v || 'all')}
          allowDeselect={false} title="Show only orders in this state"
          data={[
            { value: 'all', label: `All statuses (${count('all')})` },
            { value: 'new', label: `New (${count('new')})` },
            { value: 'todo', label: `Needs action (${count('todo')})` },
            { value: 'running', label: `Running (${count('running')})` },
            { value: 'failed', label: `Failed (${count('failed')})` },
            { value: 'final', label: `Final ready (${count('final')})` }
          ]} />
        <Select w={190} value={sort} onChange={(v) => setSort(v || 'date')}
          allowDeselect={false}
          data={[
            { value: 'date', label: 'Newest first' },
            { value: 'status', label: 'By status (todo first)' }
          ]} />
      </Group>

      {shown.length === 0 && (
        <Text c="dimmed" size="sm">
          {orders.length === 0 ? 'No orders found.'
            : 'No orders match the current search/filter.'}
        </Text>
      )}
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
          <Card key={o.folder || o.order_id} withBorder padding="md">
            <Group align="center" wrap="nowrap">
              <Stack gap={2} style={{ flex: 1, minWidth: 0 }}>
                <Text fw={600}>{o.customer || o.folder || o.order_id}</Text>
                <Text size="xs" c="dimmed">{details}</Text>
                {progress && <Text size="xs">{progress}</Text>}
                {o.blessing && (
                  <Text size="xs" c="dimmed" fs="italic"
                    title="The customer's blessing / dedication text">
                    “{o.blessing}”
                  </Text>
                )}
                {failed && (
                  <Text size="xs" c="red" title={o.ingest_error}>{o.ingest_error}</Text>
                )}
              </Stack>
              <Badge variant="light" color={color}>{text}</Badge>
              {failed && o.ingest_job && (
                <Button variant="subtle" size="xs" onClick={() => showLog(o.ingest_job)}>
                  log
                </Button>
              )}
              {o.project ? (
                <Button variant="default" size="xs" onClick={() => onOpenProject(o.project)}>
                  open “{o.project}”
                </Button>
              ) : !o.ingesting && (
                <Button size="xs" disabled={!o.folder}
                  loading={Boolean(o.folder) && busyRow === o.folder}
                  title={o.folder ? '' : 'No photo folder recorded for this order yet'}
                  onClick={() => ingest(o)}>
                  {failed ? 'Retry ingest' : 'Ingest + storyboard'}
                </Button>
              )}
            </Group>
          </Card>
        );
      })}
    </Stack>
  );
}
