// Learning tab: what the planner has learned from watching real clips.
//
// The rules listed here are appended to EVERY later planning call, so this
// page is effectively the planner's standing instructions — which is why a
// rule can be switched off (kept for the record) as easily as it was added,
// and why the note that produced each one is shown next to it. Feedback
// itself is given per clip, in a project (see ProjectDetail).
import React, { useCallback, useEffect, useState } from 'react';
import {
  Alert, Badge, Button, Card, Group, Select, Stack, Switch, Text, Textarea,
  TextInput, Title
} from '@mantine/core';
import { api } from './api.js';
import { ConfirmModal, notify } from './ui.jsx';

const SCOPE_LABEL = {
  motion: 'clip planning',
  style: 'image styling'
};

const localTime = (iso) => {
  if (!iso) return '';
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso.slice(0, 19) : d.toLocaleString();
};

// One rule: editable in place, switchable, deletable. Editing is free and
// local (it only rewrites a text file), so it saves inline — the same
// treatment the storyboard and letter editors get.
function LessonRow({ lesson, onSave, onToggle, onForget, busy }) {
  const [text, setText] = useState(lesson.text);
  const [dirty, setDirty] = useState(false);
  useEffect(() => { if (!dirty) setText(lesson.text); }, [lesson.text, dirty]);
  const source = lesson.source || {};
  return (
    <Card withBorder padding="sm" style={{ opacity: lesson.active ? 1 : 0.55 }}>
      <Group gap="xs" mb={6} wrap="nowrap">
        <Badge size="sm" variant="light"
          color={lesson.scope === 'style' ? 'blue' : 'orange'}>
          {SCOPE_LABEL[lesson.scope] || lesson.scope}
        </Badge>
        {lesson.origin === 'manual'
          ? <Badge size="sm" variant="light" color="gray">written by hand</Badge>
          : <Text size="xs" c="dimmed">
              learned from {source.project || 'a clip'}
              {source.transition ? ` · ${source.transition}` : ''}
              {source.verdict === 'good' ? ' · what worked' : ''}
            </Text>}
        <Text size="xs" c="dimmed" style={{ flex: 1 }} ta="right">
          {localTime(lesson.created_at)}
        </Text>
        <Switch size="xs" checked={lesson.active} disabled={busy}
          onChange={(e) => onToggle(lesson, e.currentTarget.checked)}
          label={lesson.active ? 'in use' : 'off'} />
      </Group>
      <Textarea autosize minRows={1} value={text} disabled={busy}
        onChange={(e) => { setText(e.currentTarget.value); setDirty(true); }} />
      <Group gap="xs" mt={6}>
        {dirty && (
          <Button size="compact-xs" loading={busy}
            onClick={async () => { await onSave(lesson, text); setDirty(false); }}>
            Save
          </Button>
        )}
        {dirty && (
          <Button size="compact-xs" variant="subtle"
            onClick={() => { setText(lesson.text); setDirty(false); }}>
            Revert
          </Button>
        )}
        <div style={{ flex: 1 }} />
        <Button size="compact-xs" variant="subtle" color="red" disabled={busy}
          onClick={() => onForget(lesson)}
          title="Delete this rule for good. Switching it off keeps the record.">
          Forget
        </Button>
      </Group>
    </Card>
  );
}

export default function LearningTab({ onOpenProject }) {
  const [data, setData] = useState(null);
  const [notes, setNotes] = useState([]);
  const [newText, setNewText] = useState('');
  const [newScope, setNewScope] = useState('motion');
  const [busy, setBusy] = useState(false);
  const [confirm, setConfirm] = useState(null);
  const [filter, setFilter] = useState('all');

  const load = useCallback(async () => {
    const [lessons, feedback] = await Promise.all([
      api.lessons(), api.feedback(50)
    ]);
    setData(lessons);
    setNotes(feedback.feedback || []);
  }, []);

  useEffect(() => {
    load().catch((e) => notify(`Lessons failed: ${e.message}`, 'red'));
  }, [load]);

  if (!data) return <Text c="dimmed">Loading what the planner has learned…</Text>;

  const lessons = (data.lessons || []).filter(
    (l) => filter === 'all' || l.scope === filter
  );
  const active = (data.lessons || []).filter((l) => l.active);
  // Only the newest `max_in_prompt` rules ride along with a call, so a long
  // list quietly stops being fully applied — say so rather than let it be a
  // surprise.
  const overflow = Math.max(0, active.length - (data.max_in_prompt || 0));

  const guard = async (label, fn) => {
    setBusy(true);
    try { await fn(); await load(); }
    catch (e) { notify(`${label} failed: ${e.message}`, 'red'); }
    finally { setBusy(false); }
  };

  const addLesson = () => guard('Adding the rule', async () => {
    await api.addLesson(newText.trim(), newScope);
    setNewText('');
    notify('Rule added — it applies to every plan from now on.', 'green');
  });

  const saveLesson = (lesson, text) =>
    guard('Saving the rule', () => api.updateLesson(lesson.id, { text }));

  const toggleLesson = (lesson, activeNow) =>
    guard('Updating the rule', () =>
      api.updateLesson(lesson.id, { active: activeNow }));

  const forgetLesson = (lesson) => setConfirm({
    title: 'Forget this rule?',
    lines: [
      `"${lesson.text}"`,
      'It is deleted from the lessons file and stops being sent to the planner.',
      'If you only want to stop using it, switch it off instead — that keeps the record of what was tried.'
    ],
    cost: 'free', danger: true, label: 'Forget it',
    action: () => guard('Forgetting the rule', async () => {
      await api.deleteLesson(lesson.id);
      notify('Rule forgotten.');
    })
  });

  return (
    <Stack gap="sm">
      <ConfirmModal confirm={confirm} onCancel={() => setConfirm(null)}
        onConfirm={() => { const c = confirm; setConfirm(null); c.action(); }} />

      <Card withBorder padding="md">
        <Group gap="sm" mb={4}>
          <Title order={4}>What the planner has learned</Title>
          <Badge variant="light" color="orange">{active.length} in use</Badge>
          {(data.lessons || []).length > active.length && (
            <Badge variant="light" color="gray">
              {(data.lessons || []).length - active.length} off
            </Badge>
          )}
        </Group>
        <Text size="sm" c="dimmed">
          Every rule here is added to the instructions the AI planner gets when
          it writes motion prompts (or, for styling rules, when it draws a
          frame). They come from the feedback you give a clip inside a project
          — watch it, say what went wrong, and the note is turned into a rule
          that applies to every movie from then on. Rules you disagree with can
          be edited, switched off, or forgotten.
        </Text>
        {!data.enabled && (
          <Alert color="yellow" variant="light" mt="sm">
            Learning is switched off in config (<code>learning_enabled</code>),
            so none of these rules are currently sent to the model.
          </Alert>
        )}
        {overflow > 0 && (
          <Alert color="yellow" variant="light" mt="sm">
            Only the newest {data.max_in_prompt} rules are sent with each call
            (<code>max_lessons_in_prompt</code>), so {overflow} older one(s) are
            not being applied. Switch off the ones that no longer matter, or
            raise the limit in config.
          </Alert>
        )}
      </Card>

      <Card withBorder padding="md">
        <Text fw={600} mb="xs">Write a rule yourself</Text>
        <Group align="flex-end">
          <Textarea style={{ flex: 1 }} autosize minRows={1}
            placeholder="e.g. When two people trade sides between the frames, make one of them visibly cross in front of the other."
            description="Short, general and imperative — it is sent with every future plan"
            value={newText} onChange={(e) => setNewText(e.currentTarget.value)} />
          <Select w={170} label="Applies to" allowDeselect={false}
            value={newScope} onChange={(v) => setNewScope(v || 'motion')}
            data={[
              { value: 'motion', label: 'Clip planning' },
              { value: 'style', label: 'Image styling' }
            ]} />
          <Button disabled={!newText.trim() || busy} loading={busy}
            onClick={addLesson}>Add rule</Button>
        </Group>
        <Text size="xs" c="dimmed" mt={6}>
          Free — writing a rule by hand spends nothing.
        </Text>
      </Card>

      <Group>
        <Select w={220} value={filter} allowDeselect={false}
          onChange={(v) => setFilter(v || 'all')}
          data={[
            { value: 'all', label: `All rules (${(data.lessons || []).length})` },
            { value: 'motion', label: 'Clip planning' },
            { value: 'style', label: 'Image styling' }
          ]} />
      </Group>

      {lessons.length === 0 ? (
        <Text c="dimmed" size="sm">
          Nothing learned yet. Open a project, play a rendered clip, and use its
          “Feedback” button to say what went wrong — that is where rules come
          from.
        </Text>
      ) : (
        // Newest first: the most recent correction is the one you are most
        // likely to be looking for.
        lessons.slice().reverse().map((lesson) => (
          <LessonRow key={lesson.id} lesson={lesson} busy={busy}
            onSave={saveLesson} onToggle={toggleLesson} onForget={forgetLesson} />
        ))
      )}

      {notes.length > 0 && (
        <Card withBorder padding="md" mt="md">
          <Text fw={600} mb={4}>Recent feedback</Text>
          <Text size="xs" c="dimmed" mb="sm">
            The notes behind the rules — kept exactly as written, with the
            motion prompt that produced the clip.
          </Text>
          <Stack gap="sm">
            {notes.slice(0, 20).map((n) => (
              <div key={n.id}>
                <Group gap="xs">
                  <Badge size="sm" variant="light"
                    color={n.verdict === 'good' ? 'green' : 'red'}>
                    {n.verdict === 'good' ? 'worked' : 'went wrong'}
                  </Badge>
                  <Text size="sm" fw={600} style={{ cursor: 'pointer' }}
                    onClick={() => onOpenProject?.(n.project)}>
                    {n.project}
                  </Text>
                  {n.transition_id && (
                    <Text size="xs" c="dimmed">{n.transition_id}</Text>
                  )}
                  <Text size="xs" c="dimmed" style={{ flex: 1 }} ta="right">
                    {localTime(n.created_at)}
                    {n.lesson_ids?.length
                      ? ` · taught ${n.lesson_ids.length} rule(s)`
                      : ' · no rule'}
                  </Text>
                </Group>
                <Text size="sm">{n.note}</Text>
                {n.motion_prompt && (
                  <Text size="xs" c="dimmed" lineClamp={2}
                    title={n.motion_prompt}>
                    prompt: {n.motion_prompt}
                  </Text>
                )}
              </div>
            ))}
          </Stack>
        </Card>
      )}
    </Stack>
  );
}
