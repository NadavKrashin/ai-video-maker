// Spending tab: what each project cost, and what the studio has spent in
// total. Every number here is an ESTIMATE priced from the `pricing` block in
// config.json (see costs.py) — no provider billing API is involved — so the
// page says that plainly rather than implying these are invoiced amounts.
import React, { useEffect, useState } from 'react';
import {
  Alert, Badge, Card, Group, Progress, Select, Stack, Table, Text, TextInput,
  Title, Tooltip
} from '@mantine/core';
import { api } from './api.js';
import { notify, usd } from './ui.jsx';

// Colour per cost kind, reused by the bar and the legend so a segment can be
// matched to its label at a glance.
const KIND_COLOR = {
  clip: 'orange',
  image: 'blue',
  text: 'grape',
  sfx: 'teal'
};

// A project's headline figure: what was recorded, unless the ledger clearly
// predates the work (then the from-disk estimate is the honest number).
const headline = (p) => (p.estimated ? p.estimated_usd : p.total_usd);

function StatCard({ label, value, hint, color }) {
  return (
    <Card withBorder padding="md" style={{ flex: '1 1 180px', minWidth: 160 }}>
      <Text size="xs" c="dimmed" tt="uppercase" fw={600}>{label}</Text>
      <Text fw={700} fz={28} c={color} mt={2}>{value}</Text>
      {hint && <Text size="xs" c="dimmed" mt={2}>{hint}</Text>}
    </Card>
  );
}

// Where the money went, as one stacked bar over all projects.
function KindBreakdown({ byKind, labels, total }) {
  const kinds = Object.entries(byKind || {}).filter(([, b]) => b.usd > 0);
  if (!kinds.length) {
    return (
      <Text size="sm" c="dimmed">
        Nothing recorded yet — the ledger fills up as steps run from now on.
      </Text>
    );
  }
  return (
    <>
      <Progress.Root size="xl" mb="sm">
        {kinds.map(([kind, bucket]) => (
          <Tooltip key={kind} label={`${labels[kind] || kind}: ${usd(bucket.usd)}`}>
            <Progress.Section value={(bucket.usd / total) * 100}
              color={KIND_COLOR[kind] || 'gray'} />
          </Tooltip>
        ))}
      </Progress.Root>
      <Group gap="lg">
        {kinds.map(([kind, bucket]) => (
          <Group key={kind} gap={6}>
            <div style={{
              width: 10, height: 10, borderRadius: 2,
              background: `var(--mantine-color-${KIND_COLOR[kind] || 'gray'}-5)`
            }} />
            <Text size="sm">{labels[kind] || kind}</Text>
            <Text size="sm" fw={600}>{usd(bucket.usd)}</Text>
            <Text size="xs" c="dimmed">{bucket.count} call(s)</Text>
          </Group>
        ))}
      </Group>
    </>
  );
}

export default function SpendingTab({ onOpenProject }) {
  const [data, setData] = useState(null);
  const [query, setQuery] = useState('');
  const [sort, setSort] = useState('spend');

  useEffect(() => {
    api.costs().then(setData)
      .catch((e) => notify(`Spending failed: ${e.message}`, 'red'));
  }, []);

  if (!data) return <Text c="dimmed">Loading spending…</Text>;

  const labels = data.labels || {};
  const totals = data.totals || {};
  const q = query.trim().toLowerCase();
  const projects = (data.projects || []).filter(
    (p) => !q || [p.project, p.order?.customer, p.order?.order_id]
      .some((v) => (v || '').toLowerCase().includes(q))
  );
  if (sort === 'spend') projects.sort((a, b) => headline(b) - headline(a));
  else if (sort === 'clips') {
    projects.sort((a, b) => (b.clips_rendered || 0) - (a.clips_rendered || 0));
  } else projects.sort((a, b) => a.project.localeCompare(b.project));

  // The studio-wide headline uses each project's own honest figure, so a mix
  // of tracked and pre-tracking projects still adds up to something true.
  const grand = (data.projects || []).reduce((sum, p) => sum + headline(p), 0);
  const perClip = totals.by_kind?.clip?.count
    ? totals.by_kind.clip.usd / totals.by_kind.clip.count : 0;

  return (
    <Stack gap="sm">
      <Group gap="sm" align="stretch" wrap="wrap">
        <StatCard label="Total spend" value={usd(grand)} color="orange.4"
          hint={`across ${(data.projects || []).length} project(s)`} />
        <StatCard label="Recorded" value={usd(totals.total_usd)}
          hint="tracked call by call" />
        <StatCard label="Valued from files" value={usd(totals.estimated_usd)}
          hint="what the artifacts on disk are worth" />
        <StatCard label="Average per clip" value={perClip ? usd(perClip) : '—'}
          hint={totals.by_kind?.clip?.count
            ? `${totals.by_kind.clip.count} clip render(s) recorded`
            : 'no clip renders recorded yet'} />
      </Group>

      <Card withBorder padding="md">
        <Text fw={600} mb="xs">Where the money goes</Text>
        <KindBreakdown byKind={totals.by_kind} labels={labels}
          total={totals.total_usd} />
      </Card>

      <Alert color="gray" variant="light">
        These are <b>estimates</b>, not invoices. Every call is priced from the{' '}
        <code>pricing</code> block in config.json — currently{' '}
        {usd(data.pricing?.clip_usd_per_second * 5)} for a 5s clip,{' '}
        {usd(data.pricing?.clip_usd_per_second * 10)} for 10s, and{' '}
        {usd(data.pricing?.openai_image_usd)} per styled image. Edit those
        numbers when a provider changes its rates and every total here follows.
      </Alert>

      <Group>
        <TextInput style={{ flex: 1 }} miw={200} value={query}
          placeholder="Search project or customer…"
          onChange={(e) => setQuery(e.target.value)} />
        <Select w={200} value={sort} allowDeselect={false}
          onChange={(v) => setSort(v || 'spend')}
          data={[
            { value: 'spend', label: 'Most expensive first' },
            { value: 'clips', label: 'Most clips first' },
            { value: 'name', label: 'By name' }
          ]} />
      </Group>

      <Card withBorder padding={0}>
        <Table highlightOnHover verticalSpacing="sm" horizontalSpacing="md">
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Project</Table.Th>
              <Table.Th ta="right">Spend</Table.Th>
              <Table.Th ta="right">Recorded</Table.Th>
              <Table.Th ta="right">Clips</Table.Th>
              <Table.Th ta="right">Images</Table.Th>
              <Table.Th>Breakdown</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {projects.map((p) => (
              <Table.Tr key={p.project} style={{ cursor: 'pointer' }}
                onClick={() => onOpenProject(p.project)}>
                <Table.Td>
                  <Text fw={600} size="sm">{p.project}</Text>
                  {p.order?.customer && (
                    <Text size="xs" c="dimmed">{p.order.customer}</Text>
                  )}
                  {p.error && <Text size="xs" c="red">{p.error}</Text>}
                </Table.Td>
                <Table.Td ta="right">
                  <Text fw={700} size="sm">{usd(headline(p))}</Text>
                  {p.estimated && (
                    <Tooltip multiline w={260} label={
                      'This project has no full ledger — its calls happened '
                      + 'before cost tracking, so the figure is priced from '
                      + 'the styled frames and clips on disk.'}>
                      <Badge size="xs" variant="light" color="gray">
                        from files
                      </Badge>
                    </Tooltip>
                  )}
                </Table.Td>
                <Table.Td ta="right">
                  <Text size="sm" c="dimmed">{usd(p.total_usd)}</Text>
                </Table.Td>
                <Table.Td ta="right">
                  <Text size="sm">{p.clips_rendered ?? 0}</Text>
                  {p.clip_seconds > 0 && (
                    <Text size="xs" c="dimmed">{p.clip_seconds}s</Text>
                  )}
                </Table.Td>
                <Table.Td ta="right">
                  <Text size="sm">{p.images ?? 0}</Text>
                </Table.Td>
                <Table.Td>
                  <Group gap={4} wrap="nowrap">
                    {Object.entries(p.by_kind || {})
                      .filter(([, b]) => b.usd > 0)
                      .map(([kind, b]) => (
                        <Badge key={kind} size="xs" variant="light"
                          color={KIND_COLOR[kind] || 'gray'}>
                          {kind} {usd(b.usd)}
                        </Badge>
                      ))}
                  </Group>
                </Table.Td>
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
        {projects.length === 0 && (
          <Text c="dimmed" size="sm" p="md">
            {(data.projects || []).length === 0
              ? 'No projects yet.'
              : 'No projects match the search.'}
          </Text>
        )}
      </Card>
    </Stack>
  );
}
