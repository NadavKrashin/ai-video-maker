// Projects tab: every local project workspace — searchable, filterable by
// pipeline status, sortable — plus creating a new one from scratch (the UI
// twin of `pipeline.py init`).
import React, { useEffect, useState } from 'react';
import {
  Badge, Button, Card, Group, Progress, Select, Stack, Text, TextInput
} from '@mantine/core';
import { api } from './api.js';
import { notify, stepChip } from './ui.jsx';

// Status key + sort rank per project. Mirrors stepChip, with 'error' extra;
// rank puts actionable work first, finished movies last.
function projectStage(p) {
  if (p.error) return { key: 'error', rank: 0, ...{ label: 'error', color: 'red' } };
  const chip = stepChip(p.next_step);
  const rank = { storyboard: 1, render: 2, combine: 3 }[p.next_step] ?? 4;
  return { key: p.next_step || 'complete', rank, ...chip };
}

const matchesQuery = (p, q) =>
  !q || [p.project, p.order?.customer, p.order?.order_id]
    .some((v) => (v || '').toLowerCase().includes(q));

export default function ProjectsTab({ onOpenProject }) {
  const [projects, setProjects] = useState(null);
  const [newName, setNewName] = useState('');
  const [creating, setCreating] = useState(false);
  const [query, setQuery] = useState('');
  const [filter, setFilter] = useState('all');
  const [sort, setSort] = useState('name');

  useEffect(() => {
    api.projects().then((d) => setProjects(d.projects))
      .catch((e) => notify(`Projects failed: ${e.message}`, 'red'));
  }, []);

  const create = async () => {
    const name = newName.trim();
    if (!name) return;
    setCreating(true);
    try {
      await api.createProject(name);
      notify(`Project "${name}" created — add photos, then run Storyboard.`, 'green');
      onOpenProject(name);
    } catch (e) { notify(`Create failed: ${e.message}`, 'red'); }
    finally { setCreating(false); }
  };

  if (projects === null) return <Text c="dimmed">Loading projects…</Text>;

  const count = (key) => key === 'all'
    ? projects.length
    : projects.filter((p) => projectStage(p).key === key).length;
  const q = query.trim().toLowerCase();
  const shown = projects.filter(
    (p) => (filter === 'all' || projectStage(p).key === filter) && matchesQuery(p, q)
  );
  if (sort === 'status') shown.sort((a, b) => projectStage(a).rank - projectStage(b).rank);
  // else: keep the server's name order

  return (
    <Stack gap="sm">
      <Card withBorder padding="md">
        <Group align="flex-end">
          <TextInput style={{ flex: 1 }} label="New project" placeholder="my-movie"
            description="letters, digits, - and _"
            value={newName} onChange={(e) => setNewName(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && create()} />
          <Button loading={creating} disabled={!newName.trim()} onClick={create}>
            Create
          </Button>
        </Group>
      </Card>

      <Group>
        <TextInput style={{ flex: 1 }} miw={200} value={query}
          placeholder="Search project or customer…"
          onChange={(e) => setQuery(e.target.value)} />
        <Select w={210} value={filter} onChange={(v) => setFilter(v || 'all')}
          allowDeselect={false}
          data={[
            { value: 'all', label: `All statuses (${count('all')})` },
            { value: 'storyboard', label: `Needs storyboard (${count('storyboard')})` },
            { value: 'render', label: `Needs render (${count('render')})` },
            { value: 'combine', label: `Needs combine (${count('combine')})` },
            { value: 'complete', label: `Complete (${count('complete')})` },
            { value: 'error', label: `Error (${count('error')})` }
          ]} />
        <Select w={190} value={sort} onChange={(v) => setSort(v || 'name')}
          allowDeselect={false}
          data={[
            { value: 'name', label: 'By name' },
            { value: 'status', label: 'By status (todo first)' }
          ]} />
      </Group>

      {shown.length === 0 && (
        <Text c="dimmed" size="sm">
          {projects.length === 0
            ? 'No projects yet — create one above or ingest an order.'
            : 'No projects match the current search/filter.'}
        </Text>
      )}
      {shown.map((p) => {
        const { label, color } = projectStage(p);
        const total = (p.clips || []).length;
        const rendered = (p.clips || []).filter((c) => c.rendered).length;
        return (
          <Card key={p.project} withBorder padding="md"
            style={{ cursor: 'pointer' }} onClick={() => onOpenProject(p.project)}>
            <Group align="center" wrap="nowrap">
              <Stack gap={2} style={{ flex: 1, minWidth: 0 }}>
                <Text fw={600}>
                  {p.project}
                  {p.order?.customer && (
                    <Text span c="dimmed" fw={400}> · {p.order.customer}</Text>
                  )}
                </Text>
                <Text size="xs" c="dimmed">
                  {p.error || `${(p.input_images || []).length} photos · ${rendered}/${total} clips` +
                    (p.final_video ? ' · final ready' : '')}
                </Text>
              </Stack>
              {total > 0 && !p.error && (
                <Progress value={(rendered / total) * 100} w={110} size="sm"
                  color={rendered === total ? 'green' : 'blue'}
                  title={`${rendered}/${total} clips rendered`} />
              )}
              <Badge variant="light" color={color}>{label}</Badge>
            </Group>
          </Card>
        );
      })}
    </Stack>
  );
}
