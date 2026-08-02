// One project's workspace, laid out as the pipeline it is:
//   1 Storyboard → 2 Render → 3 Audio → 4 Combine
// Each step is a tile showing its status; clicking it opens that step's
// options + an explanation. Every action that changes files or spends money
// goes through a confirmation modal that says exactly what will happen.
import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Alert, Badge, Button, Card, Checkbox, Divider, Group, Image, Modal, Paper,
  ScrollArea, SegmentedControl, Select, Stack, Text, Textarea, TextInput,
  Title, UnstyledButton
} from '@mantine/core';
import { api, fileUrl } from './api.js';
import { COST, ConfirmModal, notify, stepChip, usd } from './ui.jsx';

// A transition's start_frame/end_frame is the styled image path in practice
// ("styled_images/img1.png"), but may also be a frame id — handle both.
function frameName(framesById, frameRef) {
  if (frameRef && frameRef.includes('/')) return frameRef.split('/').pop();
  const f = framesById[frameRef];
  return f ? f.output_path.split('/').pop() : '';
}

const CLIP_PRICE = '≈ $0.35 (5s) – $0.70 (10s) per clip';

// State timestamps are stored in UTC. Rendering the raw string put an
// in-flight render three hours in the past on an IDT machine, which read as
// "abandoned long ago" — always show these in the viewer's own time.
const localTime = (iso) => {
  if (!iso) return 'unknown';
  const d = new Date(iso);
  return Number.isNaN(d.getTime())
    ? iso.replace('T', ' ').slice(0, 19)
    : d.toLocaleString();
};

// Every clip the CURRENT storyboard says is not up to date: never rendered,
// or rendered before an edit/re-plan/re-style changed its plan ("outdated").
// The backend marks the second kind when the panel saves storyboard edits, so
// this survives a page reload rather than living in one browser tab.
const clipsNeedingRender = (snap) =>
  (snap?.clips || []).filter((c) => !c.rendered || c.stale);

const jobColor = (state) =>
  state === 'done' ? 'green'
    : state === 'failed' ? 'red'
      // Ran to the end, but some items failed — never green.
      : state === 'partial' ? 'orange'
        : state === 'cancelled' ? 'gray'
          : state === 'cancelling' ? 'yellow'
            : 'blue';

function JobRow({ job, onShowLog, onCancel, cancelBusy }) {
  const cancellable = job.state === 'queued' || job.state === 'running';
  return (
    <Group gap="sm" py={6} wrap="nowrap"
      style={{ borderBottom: '1px solid var(--mantine-color-dark-4)' }}>
      <Badge variant="light" color={jobColor(job.state)}>
        {job.failures > 0 ? `${job.state} · ${job.failures} failed` : job.state}
      </Badge>
      <Text size="sm" fw={600}>{job.command}</Text>
      <Text size="xs" c="dimmed" style={{ flex: 1 }}>
        {(job.started_at || job.created_at || '').replace('T', ' ').slice(0, 19)}
      </Text>
      {job.error && (
        <Text size="xs" c="red" maw={380} truncate title={job.error}>{job.error}</Text>
      )}
      {(cancellable || job.state === 'cancelling') && (
        <Button variant="subtle" color="red" size="compact-xs"
          loading={cancelBusy} disabled={!cancellable}
          title="Queued jobs stop now; running jobs finish the current item first. Re-running later resumes."
          onClick={() => onCancel(job.id)}>
          {job.state === 'cancelling' ? 'cancelling…' : 'cancel'}
        </Button>
      )}
      <Button variant="subtle" size="compact-xs" onClick={() => onShowLog(job.id)}>
        log
      </Button>
    </Group>
  );
}

// Telling the planner what a rendered clip got wrong. This is the only place
// in the app where the AI's work is judged AFTER seeing the result, so it is
// deliberately a full form and not a thumbs icon: the note is what teaches
// the rule, and a vague note teaches nothing.
//
// It doubles as its own confirmation (rather than stacking a second modal on
// top): it says exactly what will happen and carries the same cost badge
// every other action's confirmation does.
function FeedbackModal({
  opened, clipId, motionPrompt, duration, result, onClose, onSubmit, onApply, busy
}) {
  const [verdict, setVerdict] = useState('bad');
  const [note, setNote] = useState('');
  const [learn, setLearn] = useState(true);
  const [watch, setWatch] = useState(true);
  useEffect(() => {
    // A fresh form per clip — a note left over from the previous clip would
    // be attributed to this one.
    if (opened) { setVerdict('bad'); setNote(''); setLearn(true); setWatch(true); }
  }, [opened, clipId]);
  const cost = (learn || watch) ? COST.openai : COST.free;
  // With a clip named, the reviewer can report on its own — so a note is
  // only strictly required when nothing is going to watch it.
  const canSend = Boolean(note.trim()) || (watch && clipId);

  const title = clipId ? `Feedback on ${clipId}` : 'Feedback on this movie';
  if (result) {
    return (
      <Modal opened={opened} onClose={onClose} centered size="lg"
        title={`${title} — what came back`}>
        <FeedbackResult result={result} duration={duration}
          onApply={onApply} onClose={onClose} />
      </Modal>
    );
  }
  return (
    <Modal opened={opened} onClose={onClose} centered size="lg" title={title}>
      <Text size="sm" c="dimmed" mb="sm">
        Say what this clip got wrong (or right) in your own words. It is saved
        with the exact motion prompt that produced it, and turned into a short
        general rule that the planner follows on every clip it writes from now
        on — in this project and every future one.
      </Text>
      {motionPrompt && (
        <Paper withBorder p="xs" mb="sm" bg="dark.6">
          <Text size="xs" c="dimmed">The prompt this clip was rendered from:</Text>
          <Text size="xs">{motionPrompt}</Text>
        </Paper>
      )}
      <SegmentedControl fullWidth value={verdict} onChange={setVerdict}
        data={[
          { label: '👎 Something went wrong', value: 'bad' },
          { label: '👍 This one worked', value: 'good' }
        ]} />
      <Textarea mt="sm" autosize minRows={3} maxRows={10} value={note}
        onChange={(e) => setNote(e.currentTarget.value)}
        label={watch && clipId ? 'What happened? (optional)' : 'What happened?'}
        description={verdict === 'bad'
          ? 'Be concrete: what did the video model do that it should not have?'
          : 'What is worth keeping — what did the prompt get right?'}
        placeholder={verdict === 'bad'
          ? 'e.g. the boy slid across the lawn without ever taking a step, and the dad turned into a different man halfway'
          : 'e.g. keeping it to one action for the whole 5 seconds made the movement look natural'} />
      {clipId && (
        <Checkbox mt="md" checked={watch}
          onChange={(e) => setWatch(e.currentTarget.checked)}
          label="Let the AI watch this clip too"
          description={'It looks at stills taken across the rendered clip, next '
            + 'to its two key frames, says what actually happened, and proposes '
            + 'a corrected prompt and length you can accept. Leave the note '
            + 'empty to just ask it what it thinks.'} />
      )}
      <Checkbox mt="sm" checked={learn} onChange={(e) => setLearn(e.currentTarget.checked)}
        label="Turn this into a rule the planner follows from now on"
        description="Written from your note and what the AI saw. Off: nothing is learned." />
      <Badge variant="light" color={cost.color} mt="md">{cost.text}</Badge>
      <Text size="xs" c="dimmed" mt={6}>
        Nothing is re-rendered by this. Any suggested prompt is shown for you
        to accept or ignore.
      </Text>
      <Group justify="flex-end" mt="lg">
        <Button variant="default" onClick={onClose}>Cancel</Button>
        <Button loading={busy} disabled={!canSend}
          onClick={() => onSubmit({ note: note.trim(), verdict, learn, review: watch })}>
          {watch && clipId ? 'Send & watch the clip'
            : learn ? 'Send & learn from it' : 'Save the note'}
        </Button>
      </Group>
    </Modal>
  );
}

// Phase two: what the reviewer saw, and the fix it proposes for THIS clip.
// Applying only edits the storyboard in the browser — the normal Save →
// "generate everything that needs it" path then marks the clip outdated and
// asks before spending anything on a new render.
function FeedbackResult({ result, duration, onApply, onClose }) {
  const review = result.review || {};
  const lessons = result.lessons || [];
  const changed = review.changes_clip;
  return (
    <>
      {result.review_error && (
        <Alert color="yellow" variant="light" mb="sm" title="The clip wasn't watched">
          {result.review_error} Your note is saved either way.
        </Alert>
      )}
      {review.observed && (
        <Paper withBorder p="sm" mb="sm">
          <Group gap="xs" mb={4}>
            <Text size="sm" fw={600}>What the AI saw</Text>
            <Badge size="sm" variant="light"
              color={review.matches_prompt ? 'green' : 'orange'}>
              {review.matches_prompt ? 'matches the prompt' : 'does not match the prompt'}
            </Badge>
          </Group>
          <Text size="sm">{review.observed}</Text>
          {(review.problems || []).length > 0 && (
            <Stack gap={2} mt="xs">
              {review.problems.map((p, i) => (
                <Text key={i} size="sm" c="orange.4">• {p}</Text>
              ))}
            </Stack>
          )}
        </Paper>
      )}
      {review.suggested_motion_prompt && (
        <Paper withBorder p="sm" mb="sm">
          <Text size="sm" fw={600} mb={4}>
            {changed ? 'Suggested instead' : 'It would keep the prompt as it is'}
          </Text>
          <Text size="sm">{review.suggested_motion_prompt}</Text>
          <Group gap="xs" mt={6}>
            <Badge size="sm" variant="light"
              color={review.suggested_duration === duration ? 'gray' : 'orange'}>
              {review.suggested_duration}s
              {review.suggested_duration === duration
                ? ' (unchanged)' : ` (was ${duration}s)`}
            </Badge>
            {review.why && <Text size="xs" c="dimmed">{review.why}</Text>}
          </Group>
        </Paper>
      )}
      {result.learn_error ? (
        <Alert color="yellow" variant="light" mb="sm">
          No rule was written ({result.learn_error}) — the note is saved.
        </Alert>
      ) : lessons.length > 0 ? (
        <Paper withBorder p="sm" mb="sm">
          <Text size="sm" fw={600} mb={4}>Learned for every future clip</Text>
          <Stack gap={4}>
            {lessons.map((l) => (
              <Text key={l.id} size="sm">• {l.text}</Text>
            ))}
          </Stack>
        </Paper>
      ) : null}
      <Group justify="flex-end" mt="lg">
        <Button variant="default" onClick={onClose}>Close</Button>
        {changed && (
          <Button onClick={() => onApply(review)}
            title="Puts this prompt and length into the transition. You still save and regenerate yourself — nothing is spent here.">
            Use this prompt for the clip
          </Button>
        )}
      </Group>
    </>
  );
}

// "Who is in this photo": the one place a human states identity as fact.
//
// The planner otherwise decides from the pixels whether the child in this
// frame is the child from the last one, and a wrong guess is what makes the
// video model morph two people into each other. Clicking a face pins a cast
// member to a spot; the x position gives the left-to-right order the video
// model actually works in, which is also what arrangement-swap detection
// needs. Free and local — it is saved with the storyboard.
function FrameTagger({ project, frame, cast, mediaV, onChange }) {
  const [active, setActive] = useState(cast[0]?.id || '');
  const people = frame.people || [];
  const byId = Object.fromEntries(cast.map((c) => [c.id, c]));
  const imgName = frame.output_path.split('/').pop();

  const place = (event) => {
    if (!active) return;
    const box = event.currentTarget.getBoundingClientRect();
    const x = Math.min(1, Math.max(0, (event.clientX - box.left) / box.width));
    const y = Math.min(1, Math.max(0, (event.clientY - box.top) / box.height));
    // One marker per person: clicking again moves them rather than cloning.
    onChange([...people.filter((p) => p.id !== active), { id: active, x, y }]);
  };

  return (
    <Card withBorder padding="sm">
      <Group gap="xs" mb={6} wrap="nowrap">
        <Text size="sm" fw={600}>{imgName}</Text>
        <Text size="xs" c="dimmed" style={{ flex: 1 }}>
          {people.length
            ? [...people].sort((a, b) => a.x - b.x)
                .map((p) => byId[p.id]?.epithet || p.id).join(' · ')
            : 'nobody tagged yet'}
        </Text>
        {people.length > 0 && (
          <Button size="compact-xs" variant="subtle" color="red"
            onClick={() => onChange([])}>clear</Button>
        )}
      </Group>
      <Group gap={4} mb={6}>
        {cast.map((c) => (
          <Button key={c.id} size="compact-xs"
            variant={active === c.id ? 'filled' : 'default'}
            onClick={() => setActive(c.id)}
            title={`Click the photo to place ${c.epithet}`}>
            {c.epithet}
            {people.some((p) => p.id === c.id) ? ' ✓' : ''}
          </Button>
        ))}
      </Group>
      <div style={{ position: 'relative', cursor: active ? 'crosshair' : 'default' }}
        onClick={place}>
        <Image src={fileUrl(project, 'styled', imgName, mediaV)} radius="sm"
          alt={imgName} />
        {people.map((p) => (
          <div key={p.id}
            title={`${byId[p.id]?.epithet || p.id} — click to remove`}
            onClick={(e) => {
              e.stopPropagation();
              onChange(people.filter((x) => x.id !== p.id));
            }}
            style={{
              position: 'absolute',
              left: `${p.x * 100}%`, top: `${p.y * 100}%`,
              transform: 'translate(-50%, -50%)',
              background: 'var(--mantine-color-orange-6)',
              color: '#fff', borderRadius: 4, padding: '1px 6px',
              fontSize: 11, whiteSpace: 'nowrap', cursor: 'pointer',
              border: '2px solid rgba(0,0,0,0.45)'
            }}>
            {byId[p.id]?.epithet || p.id}
          </div>
        ))}
      </div>
    </Card>
  );
}

// What this project has cost. Estimated, always: the figures are priced from
// config.pricing, never read off a provider invoice — so the card says so
// rather than presenting them as billed amounts.
function SpendCard({ project, cost }) {
  const [entries, setEntries] = useState(null);
  const [busy, setBusy] = useState(false);
  if (!cost) return null;
  const headline = cost.estimated ? cost.estimated_usd : cost.total_usd;
  const kinds = Object.entries(cost.by_kind || {}).filter(([, b]) => b.usd > 0);

  const showCalls = async () => {
    setBusy(true);
    try {
      const data = await api.projectCosts(project);
      setEntries(data.entries.slice().reverse());
    } catch (e) { notify(`Could not load the ledger: ${e.message}`, 'red'); }
    finally { setBusy(false); }
  };

  return (
    <Card withBorder padding="md">
      <Group gap="sm" align="baseline">
        <Text fw={600}>Spent on this project</Text>
        <Text fw={700} fz={24} c="orange.4">≈ {usd(headline)}</Text>
        {kinds.map(([kind, bucket]) => (
          <Badge key={kind} variant="light" color="gray">
            {kind} {usd(bucket.usd)}
          </Badge>
        ))}
        <div style={{ flex: 1 }} />
        {cost.entries > 0 && (
          <Button size="compact-xs" variant="subtle" loading={busy}
            onClick={() => (entries ? setEntries(null) : showCalls())}>
            {entries ? 'hide calls' : `${cost.entries} calls`}
          </Button>
        )}
      </Group>
      <Text size="xs" c="dimmed" mt={4}>
        {cost.estimated
          ? `Priced from the ${cost.images} styled image(s) and ${cost.clips_rendered} `
            + `clip(s) on disk (${cost.clip_seconds}s of video) — this project's calls `
            + `ran before spending was tracked, so only ${usd(cost.total_usd)} is recorded.`
          : `${cost.clips_rendered} clip(s), ${cost.clip_seconds}s of video, `
            + `${cost.images} styled image(s).`}
        {' '}An estimate from the prices in config.json, not a provider invoice.
      </Text>
      {entries && (
        <ScrollArea.Autosize mah={220} mt="sm">
          <Stack gap={2}>
            {entries.map((e, i) => (
              <Group key={`${e.at}-${i}`} gap="xs" wrap="nowrap">
                <Text size="xs" c="dimmed" w={140}>{localTime(e.at)}</Text>
                <Badge size="xs" variant="light">{e.kind}</Badge>
                <Text size="xs" style={{ flex: 1 }} truncate>{e.detail}</Text>
                <Text size="xs" fw={600}>{usd(e.usd)}</Text>
              </Group>
            ))}
          </Stack>
        </ScrollArea.Autosize>
      )}
    </Card>
  );
}

function TransitionCard({ project, tr, framesById, clip, edited, placeholder, verdict, planBehind, onEdit, onRegenerate, onReplan, onRedoAudio, onFeedback, busy, replanBusy, audioBusy, mediaV }) {
  const startImg = frameName(framesById, tr.start_frame);
  const endImg = frameName(framesById, tr.end_frame);
  const clipFile = tr.output_path.split('/').pop();
  return (
    <Card withBorder padding="md">
      <Group gap="xs" mb="sm">
        <Text fw={700}>{tr.id}</Text>
        <Badge variant="light" color={clip?.rendered ? 'green' : 'yellow'}>
          {clip?.rendered ? (clip.sfx ? 'rendered · sfx' : 'rendered · silent') : 'not rendered'}
        </Badge>
        {clip?.stale && (
          <Badge variant="light" color="orange"
            title="The storyboard changed after this clip was rendered (re-planned motion). The clip is kept as-is — click regenerate when you want the new plan applied (spends credits).">
            outdated
          </Badge>
        )}
        <Text size="xs" c="dimmed">{tr.duration}s</Text>
        {edited && <Badge variant="light" color="yellow">edited</Badge>}
        {placeholder && !edited && (
          <Badge variant="light" color="red"
            title="Planning failed for this pair; it still has the generic fallback prompt. Re-running Storyboard re-plans it.">
            generic prompt
          </Badge>
        )}
        {planBehind && (
          <Badge variant="light" color="yellow"
            title="This prompt was written before the current photo tags or cast names. Re-plan it to use them — the clip itself is untouched until you regenerate it.">
            prompt predates your tags
          </Badge>
        )}
        {verdict && (
          <Badge variant="light" color={verdict === 'good' ? 'green' : 'orange'}
            title="You have given feedback on this clip — see the Learning tab for what it taught.">
            {verdict === 'good' ? 'you liked this' : 'you flagged this'}
          </Badge>
        )}
      </Group>
      <Group align="center" gap="md">
        <Group gap={6} align="center" wrap="nowrap">
          {startImg && <Image src={fileUrl(project, 'styled', startImg, mediaV)} alt={tr.start_frame}
            w={128} radius="sm" loading="lazy" />}
          <Text c="dimmed">→</Text>
          {endImg && <Image src={fileUrl(project, 'styled', endImg, mediaV)} alt={tr.end_frame}
            w={128} radius="sm" loading="lazy" />}
        </Group>
        {clip?.rendered && (
          // preload="none": don't fetch clip bytes until the user hits play.
          // A project with ~20 clips otherwise fires 20+ range requests on
          // open, which floods the single-worker origin through the tunnel
          // and freezes the panel (looked like the tunnel "going down").
          <video controls preload="none"
            style={{ width: 260, borderRadius: 8, background: '#000' }}
            src={fileUrl(project, 'clips', clipFile, mediaV)} />
        )}
      </Group>
      <Textarea label="Motion prompt" mt="sm" autosize minRows={2}
        value={tr.motion_prompt}
        onChange={(e) => onEdit({ ...tr, motion_prompt: e.target.value })} />
      <Group mt="sm" align="flex-end">
        <Select label="Duration" w={90} allowDeselect={false}
          value={String(tr.duration)}
          onChange={(v) => onEdit({ ...tr, duration: Number(v) })}
          data={[{ value: '5', label: '5s' }, { value: '10', label: '10s' }]} />
        <TextInput label="Sound prompt (optional)" style={{ flex: 1 }} miw={220}
          value={tr.sound_prompt || ''}
          onChange={(e) => onEdit({ ...tr, sound_prompt: e.target.value })} />
        <Button variant="default" size="xs" loading={replanBusy}
          onClick={() => onReplan(tr.id)}
          title="Ask the AI planner to write a fresh motion prompt for this pair (small OpenAI call; the clip is not re-rendered)">
          Re-plan prompt
        </Button>
        {clip?.rendered && (
          <Button variant="default" size="xs" loading={audioBusy}
            onClick={() => onRedoAudio(tr.id)}
            title="Redo just this clip's SFX (use after editing its sound prompt)">
            Redo audio
          </Button>
        )}
        {clip?.rendered && (
          // Only for a clip that EXISTS: feedback is a judgement of a result,
          // and there is nothing to judge before the render.
          <Button variant="light" size="xs" onClick={() => onFeedback(tr)}
            title="Watched it? Tell the planner what it got wrong — it learns a rule from your note and applies it to every future clip.">
            Feedback
          </Button>
        )}
        <Button variant="default" size="xs" loading={busy}
          onClick={() => onRegenerate(tr.id)}
          title="Re-render just this clip (save edits first)">
          {clip?.rendered ? 'Regenerate clip' : 'Render this clip'}
        </Button>
      </Group>
    </Card>
  );
}

// ----------------------- per-step option panels ----------------------------- #
// Each panel = a plain-words explanation of the step, its CLI options, and a
// primary button that opens the confirmation modal (nothing runs directly).

function PanelIntro({ children }) {
  return <Text size="sm" c="dimmed" mb="md">{children}</Text>;
}

// The music bed: always a track the user uploads, never generated. It is
// CHOSEN during combine, so it appears in both the Audio and Combine steps —
// looking for it only under "Audio" made sense when a model wrote the music,
// but now it is just a file the final movie needs.
function MusicBed({ info, onUpload, onRemove, onFetchUrl, busy, note }) {
  const inputRef = useRef(null);
  const [url, setUrl] = useState('');
  const submitUrl = () => {
    if (!url.trim()) return;
    onFetchUrl(url.trim(), () => setUrl(''));
  };
  return (
    <>
      {/* The accept list is deliberately wide. A bare accept="audio/*" makes
          the iOS Files picker grey out tracks it can't type — including an
          mp3 saved without a .mp3 on the end, which is exactly what a real
          phone upload hit. The server checks the CONTENT, so letting the
          picker offer the file is safe. */}
      <input ref={inputRef} type="file" style={{ display: 'none' }}
        accept={'audio/*,video/mp4,application/octet-stream,'
          + '.mp3,.m4a,.wav,.aac,.ogg,.oga,.opus,.flac,.aif,.aiff,.wma'}
        onChange={(e) => { onUpload(e.target.files?.[0]); e.target.value = ''; }} />
      <Group mt="md" gap="sm" align="center">
        <Text size="sm" fw={600}>Background music:</Text>
        {info.customMusic ? (
          <>
            <Badge variant="light" color="green">track set</Badge>
            <Button variant="default" size="xs" loading={busy}
              onClick={() => inputRef.current?.click()}>Replace file</Button>
            <Button variant="subtle" color="red" size="xs" onClick={onRemove}>
              Remove
            </Button>
          </>
        ) : (
          <>
            <Badge variant="light" color="gray">none — the movie will have no music</Badge>
            <Button variant="default" size="xs" loading={busy}
              onClick={() => inputRef.current?.click()}>Upload a file…</Button>
          </>
        )}
      </Group>
      <Group mt="xs" gap="sm" align="flex-end">
        <TextInput style={{ flex: 1 }} miw={260} size="xs"
          label="…or fetch from a URL"
          description="A direct audio link, or a video/track page to pull the audio from"
          placeholder="https://…"
          value={url} onChange={(e) => setUrl(e.currentTarget.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') submitUrl(); }} />
        <Button variant="default" size="xs" mb={2} loading={busy}
          disabled={!url.trim()} onClick={submitUrl}>Fetch</Button>
      </Group>
      <Text size="xs" c="dimmed" mt={6}>
        These movies are sold to customers, so use music you have the right to
        use — a royalty-free library, a CC-licensed track, or something you
        licensed. Downloading a copyrighted song does not make it usable.
      </Text>
      <Text size="xs" c="dimmed" mt={4}>{note}</Text>
    </>
  );
}

// The closing letter: plain text scrolled over the ending. It used to be a
// file only a shell on the server could write (the panel could switch it on
// but not author it), which behind the tunnel made it unusable remotely.
// The box detects direction per paragraph while typing (see the input
// styles); the renderer does its own bidi pass again at combine time.
function LetterEditor({ text, onSave, busy }) {
  const [value, setValue] = useState(text || '');
  const [dirty, setDirty] = useState(false);
  const serverText = useRef(text || '');
  useEffect(() => {
    // Adopt the server's copy only while the box is untouched: the detail
    // view is re-fetched every 3s during a job, and a poll must never
    // overwrite a letter someone is in the middle of writing.
    if (serverText.current !== (text || '')) {
      serverText.current = text || '';
      if (!dirty) setValue(text || '');
    }
  }, [text, dirty]);
  const save = async () => {
    // A failed save must leave the box dirty (and the text in it) — losing
    // someone's letter to a network blip is not an acceptable outcome.
    try { await onSave(value); } catch { return; }
    serverText.current = value;
    setDirty(false);
  };
  const chars = value.trim().length;
  return (
    <>
      <Group mt="md" gap="sm" align="center">
        <Text size="sm" fw={600}>Closing letter:</Text>
        {chars
          ? <Badge variant="light" color="green">{chars} characters</Badge>
          : <Badge variant="light" color="gray">
              none — the “Closing letter” toggle will have nothing to scroll
            </Badge>}
        {dirty && <Badge variant="light" color="yellow">unsaved</Badge>}
      </Group>
      {/* `unicode-bidi: plaintext` is dir="auto" per PARAGRAPH, and it has to
          be set on the input itself: Mantine puts a plain dir attribute on
          its wrapper, where auto resolves from the (empty) wrapper text and
          leaves a Hebrew letter laid out left-to-right. */}
      <Textarea mt="xs" autosize minRows={4} maxRows={14}
        styles={{ input: { unicodeBidi: 'plaintext', textAlign: 'start' } }}
        placeholder={'A few lines to close the movie with — Hebrew is fine.\n\n'
          + 'Leave a blank line between paragraphs.'}
        value={value} disabled={busy}
        onChange={(e) => { setValue(e.currentTarget.value); setDirty(true); }} />
      <Group mt="xs" gap="sm">
        <Button size="xs" variant="default" loading={busy} disabled={!dirty}
          onClick={save}>Save letter</Button>
        <Text size="xs" c="dimmed">
          Saved to the project as letter.txt. The letter is drawn during
          Combine, so changing it needs another Combine — never a re-render.
        </Text>
      </Group>
    </>
  );
}

function StoryboardPanel({ ask, locked, info, onReplanAll }) {
  const [idea, setIdea] = useState('');
  const [frameCount, setFrameCount] = useState('');
  const [stylePrompt, setStylePrompt] = useState('');
  const [duration, setDuration] = useState('');
  const [analyze, setAnalyze] = useState(true);
  const [force, setForce] = useState(false);
  const fromIdea = Boolean(idea.trim());
  const start = () => {
    const o = {};
    if (fromIdea) o.idea = idea.trim();
    if (fromIdea && frameCount !== '') o.frame_count = Number(frameCount);
    if (stylePrompt.trim()) o.style_prompt = stylePrompt.trim();
    if (duration) o.duration = Number(duration);
    if (!analyze) o.analyze_frames = false;
    if (force) o.force = true;
    ask({
      title: 'Run storyboard?',
      lines: fromIdea ? [
        'Invents the whole storyboard from your idea: the AI writes the frames and generates an image for each one.',
        'Then plans a motion prompt for every pair of frames.'
      ] : [
        force
          ? `Re-styles ALL ${info.photoCount} photo(s) from scratch (force).`
          : info.unstyledCount > 0
            ? `Styles ${info.unstyledCount} new/changed photo(s) into the cartoon look (${info.photoCount - info.unstyledCount} already styled are reused).`
            : 'All photos are already styled — they are reused as-is.',
        'Plans motion prompts only for new/changed pairs — existing prompts and your hand edits are kept.',
        'Then takes a first guess at who is in each untagged photo, so step 2 starts from something to correct rather than a blank slate (one extra vision call; photos you have already tagged are untouched).',
        'Nothing is deleted: if a re-plan affects an already-rendered clip, that clip is only marked "outdated".'
      ],
      cost: 'openai',
      label: 'Run storyboard',
      command: 'storyboard', options: o
    });
  };
  return (
    <div>
      <PanelIntro>
        Turns the photos into styled cartoon frames and writes a motion plan
        for each consecutive pair. It re-runs only what changed, so running it
        again after edits is safe. The result stops here for your review —
        nothing renders until you say so.
      </PanelIntro>
      <Textarea
        label="Idea"
        description="Leave empty to storyboard from the project's photos; fill to invent frames from text"
        placeholder="(from photos)" autosize minRows={2}
        value={idea} onChange={(e) => setIdea(e.target.value)} />
      <Group mt="sm" align="flex-end">
        {fromIdea && (
          <TextInput label="Frames" description="0 = model decides" w={120}
            type="number" placeholder="config"
            value={frameCount} onChange={(e) => setFrameCount(e.target.value)} />
        )}
        <Select label="Force every clip to" w={150} value={duration}
          onChange={(v) => setDuration(v || '')}
          data={[
            { value: '', label: 'planner mixes' },
            { value: '5', label: '5s' },
            { value: '10', label: '10s' }
          ]} />
        <TextInput label="Style prompt override (optional)" style={{ flex: 1 }} miw={220}
          placeholder="(config style_prompt)"
          value={stylePrompt} onChange={(e) => setStylePrompt(e.target.value)} />
      </Group>
      <Group mt="md">
        <Checkbox label="Analyze frames (per-clip motion plans)"
          checked={analyze} onChange={(e) => setAnalyze(e.target.checked)}
          title="Off: skip the vision analysis and give every clip the one generic motion prompt" />
        <Checkbox label="Force re-style all photos"
          checked={force} onChange={(e) => setForce(e.target.checked)}
          title="Re-style photos that are already styled (spends OpenAI credits on every photo again)" />
        <div style={{ flex: 1 }} />
        <Button variant="default" disabled={locked || !info.total}
          onClick={onReplanAll}
          title="Re-write every motion prompt from the current cast and photo tags. Hand-written prompts are replaced.">
          Re-plan all prompts…
        </Button>
        <Button disabled={locked} onClick={start}>Run storyboard…</Button>
      </Group>
    </div>
  );
}

// Step 2: the review stop. Planning has just named the cast and taken a
// first guess at who is in each photo; this is where a human fixes both and
// then pushes those corrections into the prompts. Nothing here renders.
function PeoplePanel({
  locked, info, onPropose, onReplanAll, onReplanOutdated, onOpenTagger
}) {
  const untagged = info.totalFrames - info.taggedFrames;
  return (
    <div>
      <PanelIntro>
        The video model only ever sees two photos and a sentence, so who it
        thinks is who comes entirely from the words the planner chose. This
        step is where you make those words true: check the cast names, say who
        is in each photo, then re-plan so the prompts use it. Two people
        morphing into each other almost always starts here.
      </PanelIntro>
      {info.outdatedPlans > 0 && (
        <Alert color="yellow" variant="light" mb="md"
          title={`${info.outdatedPlans} prompt(s) don't know about your latest tags`}>
          They were written before you tagged those photos (or renamed someone
          in the cast), so they still describe people the old way. Re-planning
          just these rewrites them and leaves every other prompt alone.
          <Group mt="sm">
            <Button size="xs" disabled={locked} onClick={onReplanOutdated}>
              Re-plan these {info.outdatedPlans}…
            </Button>
          </Group>
        </Alert>
      )}
      {info.fragileEpithets > 0 && (
        <Alert color="orange" variant="light" mb="md"
          title={`${info.fragileEpithets} cast name(s) describe clothing`}>
          They only work in the photo they came from. Rewrite them below as
          something the person keeps — “the smaller boy with curly hair”, “the
          taller boy” — before re-planning.
        </Alert>
      )}
      <Group align="flex-end">
        <Text size="sm" style={{ flex: 1 }}>
          {info.totalFrames === 0
            ? 'No frames yet — run Storyboard first.'
            : untagged === 0
              ? `All ${info.totalFrames} photo(s) have someone tagged.`
              : `${info.taggedFrames} of ${info.totalFrames} photo(s) tagged — ${untagged} still unanswered.`}
        </Text>
        <Button variant="default" size="xs" disabled={locked || !info.hasCast}
          onClick={onPropose}
          title="Ask the AI who is in each untagged photo. A draft you then correct — already-tagged photos are left alone.">
          Let the AI propose…
        </Button>
        <Button variant="default" size="xs" onClick={onOpenTagger}>
          Open the tagger
        </Button>
        <Button size="xs" disabled={locked || !info.total} onClick={onReplanAll}
          title="Rewrite every motion prompt using the current cast and tags. Hand-written prompts are replaced; rendered clips whose plan changes are marked outdated.">
          Re-plan all with these…
        </Button>
      </Group>
      <Text size="xs" c="dimmed" mt="xs">
        Tagging and cast edits are free and change nothing on their own — the
        re-plan is what carries them into the prompts, and it never re-renders
        a clip by itself.
      </Text>
    </div>
  );
}

function RenderPanel({ ask, locked, info, onGenerateAll }) {
  const [motionPrompt, setMotionPrompt] = useState('');
  const [duration, setDuration] = useState('');
  const [dryRun, setDryRun] = useState(false);
  const start = () => {
    const o = {};
    if (motionPrompt.trim()) o.motion_prompt = motionPrompt.trim();
    if (duration) o.duration = Number(duration);
    if (dryRun) o.dry_run = true;
    ask(dryRun ? {
      title: 'Dry-run render?',
      lines: ['Shows what would be rendered in the job log. No clips are generated, nothing changes.'],
      cost: 'free', label: 'Dry run', command: 'render', options: o
    } : {
      title: `Render ${info.missing} missing clip(s)?`,
      lines: info.missing > 0 ? [
        `Generates the ${info.missing} clip(s) that don't exist yet with the video model (${info.rendered} already rendered are untouched).`,
        CLIP_PRICE,
        'Existing clips are never replaced by this — regenerating a specific clip is a per-clip action.'
      ] : [
        'Every clip already exists — nothing will be rendered (use a clip\'s own "Regenerate" button to redo one).'
      ],
      cost: info.missing > 0 ? 'fal' : 'free',
      label: 'Render', command: 'render', options: o
    });
  };
  return (
    <div>
      <PanelIntro>
        Generates a video clip for every transition in the storyboard that
        doesn't have one yet. Review the storyboard first — each clip is paid
        the moment it renders.
      </PanelIntro>
      {info.outdated > 0 && (
        <Alert color="orange" variant="light" mb="md"
          title={`${info.outdated} rendered clip(s) are outdated`}>
          Their motion prompt, duration or frames changed after they were
          rendered. Rendering missing clips leaves them alone — use
          “Generate everything that needs it” to redo them in the same job.
        </Alert>
      )}
      <Group align="flex-end">
        <TextInput label="Motion prompt override for this run (optional)"
          style={{ flex: 1 }} miw={240} placeholder="(per-clip storyboard prompts)"
          value={motionPrompt} onChange={(e) => setMotionPrompt(e.target.value)} />
        <Select label="Force duration" w={130} value={duration}
          onChange={(v) => setDuration(v || '')}
          data={[
            { value: '', label: 'storyboard' },
            { value: '5', label: '5s' },
            { value: '10', label: '10s' }
          ]} />
        <Checkbox label="Dry run" checked={dryRun}
          onChange={(e) => setDryRun(e.target.checked)} mb={6}
          title="Show what would be rendered without spending credits" />
        <Button variant="default" disabled={locked} onClick={start}>
          {`Render ${info.missing || 'missing'} clip(s)…`}
        </Button>
        <Button disabled={locked || info.needsRender === 0} onClick={onGenerateAll}
          title="One job for every clip that is missing OR outdated — the batch you normally want after editing motion prompts">
          {`Generate everything that needs it (${info.needsRender})…`}
        </Button>
      </Group>
    </div>
  );
}

function AudioPanel({ ask, locked, info, onUploadMusic, onRemoveMusic, onFetchMusicUrl, musicBusy }) {
  const start = () => {
    ask({
      title: 'Run audio?',
      lines: [
        info.silentRendered > 0
          ? `Adds synced sound effects to ${info.silentRendered} silent clip(s) (clips that already have SFX are skipped).`
          : 'All rendered clips already have SFX — they are skipped.',
        info.customMusic
          ? 'Uses your uploaded music track for the background bed, then rebuilds the final video with everything mixed.'
          : 'No music track has been uploaded, so the movie is rebuilt with sound effects only — upload one below if you want a music bed.',
        'Audio jobs are much cheaper than clip renders.'
      ],
      cost: 'fal', label: 'Run audio', command: 'audio'
    });
  };
  return (
    <div>
      <PanelIntro>
        Two layers of sound: per-clip effects synced to the action, and one
        background music track mixed over the whole movie. Optional — skip it
        for a silent film.
      </PanelIntro>
      <Group align="flex-end">
        <Button disabled={locked} onClick={start}>Run audio…</Button>
      </Group>
      <MusicBed info={info} onUpload={onUploadMusic} onRemove={onRemoveMusic}
        onFetchUrl={onFetchMusicUrl} busy={musicBusy}
        note={'Music is never generated — the bed is whatever track you upload '
          + 'here (mp3, wav, m4a…). It is used as-is for the whole movie and '
          + 'mixed in when the movie is combined.'} />
    </div>
  );
}

// Tri-state checkbox cycling config → on → off → config.
function TriToggle({ value, onChange, label, title, note = '' }) {
  const suffix = value === null ? ' (config)' : value ? ' (on)' : ' (off)';
  return (
    <Checkbox label={label + suffix + note} title={title}
      checked={value === true} indeterminate={value === null}
      onChange={() => onChange(value === null ? true : value === true ? false : null)} />
  );
}

function CombinePanel({
  ask, locked, info, onUploadMusic, onRemoveMusic, onFetchMusicUrl, musicBusy,
  letterText, onSaveLetter, letterBusy
}) {
  const [intro, setIntro] = useState(null);
  const [credits, setCredits] = useState(null);
  const [letter, setLetter] = useState(null);
  const start = (finalize) => {
    // The delivery preset takes everything this project HAS: the letter is
    // opt-in per project (most orders have none), so it joins only when
    // there is text to scroll rather than switching on a silent no-op.
    const o = finalize
      ? {
        intro_clip: true, credits_photos: true,
        closing_letter: info.letterChars > 0, force: true
      }
      : { force: true };
    if (!finalize) {
      if (intro !== null) o.intro_clip = intro;
      if (credits !== null) o.credits_photos = credits;
      if (letter !== null) o.closing_letter = letter;
    }
    const extras = [
      o.intro_clip && 'intro', o.credits_photos && 'photo credits',
      o.closing_letter && 'closing letter'
    ].filter(Boolean);
    ask({
      title: finalize ? 'Finalize the movie?' : 'Combine the clips?',
      lines: [
        `Concatenates the ${info.rendered} rendered clip(s) into output/final_video.mp4` +
          (info.finalExists ? ', replacing the existing final video.' : '.'),
        extras.length ? `Extras: ${extras.join(' + ')}.` : 'No extras (intro/credits/letter follow the config).',
        // Turning the letter on without one written is a silent no-op in the
        // pipeline (it warns to the log and combines without it) — say so
        // here instead, where the decision is being made.
        ...(o.closing_letter && !info.letterChars
          ? ['No letter is written for this project yet, so the movie will be built WITHOUT one — write it below first.']
          : []),
        'Pure ffmpeg on your machine — nothing is sent to any API.'
      ],
      cost: 'free',
      danger: info.finalExists,
      label: finalize ? 'Finalize' : 'Combine',
      command: 'combine', options: o
    });
  };
  return (
    <div>
      <PanelIntro>
        Stitches the rendered clips (in storyboard order) into the final
        1920×1080 movie, mixing in your uploaded music track. Free and
        repeatable — rerun it any time a clip or the music changes.
      </PanelIntro>
      <Group>
        <TriToggle value={intro} onChange={setIntro} label="Intro clip"
          title="Prepend the shared intro.mp4" />
        <TriToggle value={credits} onChange={setCredits} label="Credits photos"
          title="End-credits montage of the original photos" />
        <TriToggle value={letter} onChange={setLetter} label="Closing letter"
          note={info.letterChars ? '' : ' — none written yet'}
          title="Scroll the letter written below over the ending" />
        <div style={{ flex: 1 }} />
        <Button variant="default" disabled={locked} onClick={() => start(true)}
          title="The delivery preset: intro + photo credits + the letter (when there is one) + rebuild">
          Finalize…</Button>
        <Button disabled={locked} onClick={() => start(false)}>Combine…</Button>
      </Group>
      <LetterEditor text={letterText} onSave={onSaveLetter} busy={letterBusy} />
      <MusicBed info={info} onUpload={onUploadMusic} onRemove={onRemoveMusic}
        onFetchUrl={onFetchMusicUrl} busy={musicBusy}
        note={'The music bed is laid under the finished movie by THIS step, so '
          + 'changing the track only needs another Combine — no re-render.'} />
    </div>
  );
}

// Step 5: deliver the finished movie back into the customer's own Cloudinary
// order folder. The name it will be saved under is fetched from the server
// (which asks Cloudinary what is already there) and shown for approval BEFORE
// anything is uploaded — see onPublish in ProjectDetail.
function PublishPanel({ project, locked, info, onPublish, publishing }) {
  const published = info.published || {};
  const latest = published.latest;
  const versions = published.versions || [];
  return (
    <div>
      <PanelIntro>
        Uploads the finished movie into this order's own Cloudinary folder,
        next to the customer's photos. Nothing already there is ever replaced
        or deleted: each publish is saved as the next version — final_v1,
        final_v2, final_v3… A copy of every delivered version is also kept
        here, under the project's output/published/ folder.
      </PanelIntro>
      {!published.publishable && (
        <Alert color="gray" variant="light" mb="sm">
          This project didn't come from a web order (no order.json), so there
          is no Cloudinary order folder to publish into.
        </Alert>
      )}
      {latest && (
        <Text size="sm" mb="xs">
          Last published: <b>v{latest.version}</b>
          {latest.published_at ? ` on ${localTime(latest.published_at)}` : ''}
          {published.changed_since
            && ' · the final video has been rebuilt since, so publishing now adds a new version.'}
        </Text>
      )}
      {/* Every delivery, newest first. The local copy matters because
          output/final_video.mp4 is overwritten by the next Combine — this is
          the only way back to what an earlier customer actually received. */}
      {versions.length > 0 && (
        <Stack gap={4} mb="sm">
          {versions.map((v) => (
            <Group key={v.version} gap="xs" wrap="nowrap">
              <Badge variant="light" size="sm">v{v.version}</Badge>
              <Text size="xs" c="dimmed" style={{ flex: 1 }} truncate
                title={v.public_id}>
                {v.published_at ? localTime(v.published_at) : ''} · {v.public_id}
              </Text>
              {v.url && (
                <Button component="a" size="compact-xs" variant="subtle"
                  href={v.url} target="_blank" rel="noreferrer">
                  Cloudinary
                </Button>
              )}
              {v.local_exists ? (
                <Button component="a" size="compact-xs" variant="light"
                  href={fileUrl(project, 'published', v.local_file.split('/').pop())}
                  download={`${project}_v${v.version}.mp4`}>
                  Download copy
                </Button>
              ) : (
                <Text size="xs" c="dimmed" title="Only the Cloudinary copy exists">
                  no local copy
                </Text>
              )}
            </Group>
          ))}
        </Stack>
      )}
      <Group align="flex-end">
        <Text size="xs" c="dimmed" style={{ flex: 1 }}>
          {info.finalExists
            ? 'You will see the exact file name before anything is uploaded.'
            : 'Build the final movie in step 4 first — there is nothing to publish yet.'}
        </Text>
        <Button disabled={locked || !info.finalExists || !published.publishable}
          loading={publishing} onClick={onPublish}>
          Publish to Cloudinary…
        </Button>
      </Group>
    </div>
  );
}

function RunAllPanel({ ask, locked, info }) {
  const [noCombine, setNoCombine] = useState(false);
  const start = () => {
    const o = noCombine ? { no_combine: true } : {};
    ask({
      title: 'Run the whole pipeline?',
      lines: [
        'Storyboard (if missing) → render ALL missing clips → audio (if enabled) → final video, as one unattended job.',
        `Right now that means up to ${info.missing || info.total || 'all'} clip render(s). ${CLIP_PRICE}.`,
        'The usual review stops are skipped — confirmation gates auto-proceed on the server. Use the steps above when you want to check the storyboard first.'
      ],
      cost: 'both', danger: true,
      label: 'Run everything', command: 'run', options: o
    });
  };
  return (
    <div>
      <PanelIntro>
        The hands-off mode: every remaining step in one job, no review stops.
        Best for a project whose storyboard you've already checked — or when
        you trust the plan blindly.
      </PanelIntro>
      <Group>
        <Checkbox label="Stop after clips (no final video)" checked={noCombine}
          onChange={(e) => setNoCombine(e.target.checked)} />
        <div style={{ flex: 1 }} />
        <Button color="red" disabled={locked} onClick={start}>Run everything…</Button>
      </Group>
    </div>
  );
}

// ------------------------------ main view ---------------------------------- #

// Sentinel ids in the `dirty` set for the storyboard-level fields — the
// global motion prompt and the cast list (transition ids never collide).
const GLOBAL_EDIT = '__global_motion__';
const CAST_EDIT = '__characters__';
const TAGS_EDIT = '__frame_people__';

// The review stop between planning and rendering is its own step, because
// that is where the work actually is: the planner names the cast and takes a
// first guess at who is in each photo, a human corrects both, and only a
// re-plan carries those corrections into the prompts.
const STEPS = [
  { id: 'storyboard', n: 1, name: 'Storyboard', caption: 'Style photos & plan each clip' },
  { id: 'people', n: 2, name: 'People', caption: 'Check who is who, then re-plan' },
  { id: 'render', n: 3, name: 'Render', caption: 'Generate the video clips' },
  { id: 'audio', n: 4, name: 'Audio', caption: 'Sound effects + music' },
  { id: 'combine', n: 5, name: 'Combine', caption: 'Build the final movie' },
  { id: 'publish', n: 6, name: 'Publish', caption: 'Deliver to the order folder' }
];

function StepTile({ selected, highlight, dashed, onClick, children }) {
  return (
    <UnstyledButton onClick={onClick}
      style={{
        flex: dashed ? '0 1 160px' : '1 1 160px', minWidth: 150,
        borderRadius: 'var(--mantine-radius-md)', padding: '10px 12px',
        background: dashed ? 'transparent' : 'var(--mantine-color-dark-6)',
        border: `1px ${dashed ? 'dashed' : 'solid'} ${
          selected ? 'var(--mantine-color-orange-5)'
            : highlight ? 'var(--mantine-color-orange-8)'
              : 'var(--mantine-color-dark-4)'}`
      }}>
      {children}
    </UnstyledButton>
  );
}

export default function ProjectDetail({ name, onBack }) {
  const [snap, setSnap] = useState(null);
  const [storyboard, setStoryboard] = useState(null); // parsed, editable copy
  const [dirty, setDirty] = useState(new Set());
  const [busyAction, setBusyAction] = useState('');
  const [logJob, setLogJob] = useState(null);
  const [showPhotos, setShowPhotos] = useState(false);
  const [showTagger, setShowTagger] = useState(false);
  const [openPanel, setOpenPanel] = useState('');
  const [uploading, setUploading] = useState(false);
  const [musicBusy, setMusicBusy] = useState(false);
  const [letterBusy, setLetterBusy] = useState(false);
  const [confirm, setConfirm] = useState(null);
  // Which transition the feedback form is open for (null = closed), and what
  // came back for it (the AI's review + any rule learned).
  const [feedbackFor, setFeedbackFor] = useState(null);
  const [feedbackBusy, setFeedbackBusy] = useState(false);
  const [feedbackResult, setFeedbackResult] = useState(null);
  // Cache-buster for styled frames and clips: they are replaced in place, so
  // the browser would otherwise keep showing the pre-render version. Bumped
  // on every poll while a job runs, and once more when it settles.
  const [mediaV, setMediaV] = useState(0);
  const pollRef = useRef(null);
  const fileInputRef = useRef(null);

  const load = useCallback(async () => {
    const data = await api.project(name);
    setSnap(data);
    if (data.storyboard_json) {
      try {
        setStoryboard((prev) => {
          const fresh = JSON.parse(data.storyboard_json);
          // Keep unsaved edits across background refreshes.
          if (!prev) return fresh;
          const editedById = Object.fromEntries(
            prev.transitions.filter((t) => dirtyRef.current.has(t.id)).map((t) => [t.id, t])
          );
          fresh.transitions = fresh.transitions.map((t) => editedById[t.id] || t);
          if (dirtyRef.current.has(GLOBAL_EDIT)) {
            fresh.global_motion_prompt = prev.global_motion_prompt;
          }
          if (dirtyRef.current.has(CAST_EDIT)) {
            fresh.characters = prev.characters;
          }
          if (dirtyRef.current.has(TAGS_EDIT)) {
            // Who-is-who tags are hand work; a background refresh must never
            // wipe markers someone is in the middle of placing.
            const taggedByPath = Object.fromEntries(
              (prev.frames || []).map((f) => [f.output_path, f.people])
            );
            fresh.frames = (fresh.frames || []).map(
              (f) => ({ ...f, people: taggedByPath[f.output_path] ?? f.people })
            );
          }
          return fresh;
        });
      } catch { setStoryboard(null); }
    } else setStoryboard(null);
    return data;
  }, [name]);

  const dirtyRef = useRef(dirty);
  dirtyRef.current = dirty;

  useEffect(() => { load().catch((e) => notify(`Load failed: ${e.message}`, 'red')); }, [load]);

  // Poll while a job is queued/running, refresh once when it settles.
  //
  // The media version is bumped ONLY when a job settles — never per tick.
  // Busting the cache every 3s re-downloaded every styled image and clip
  // preview on each poll (17 images/tick was measured, ~6000 requests in an
  // afternoon), which floods the single-worker origin through the tunnel and
  // makes the whole panel feel frozen. It also isn't needed to watch frames
  // arrive: a newly styled frame is a URL the browser has never fetched, so
  // it loads anyway. Cache-busting only matters for files REPLACED in place,
  // and once at the end covers that.
  const activeJobId = (snap?.jobs || []).find(
    (j) => ['queued', 'running', 'cancelling'].includes(j.state)
  )?.id || '';
  useEffect(() => {
    if (!activeJobId) {
      setMediaV((v) => v + 1);
      return undefined;
    }
    setShowPhotos(true); // watching the frames arrive is the point of waiting
    pollRef.current = setInterval(() => load().catch(() => {}), 3000);
    return () => clearInterval(pollRef.current);
  }, [activeJobId, load]);

  if (!snap) return <Text c="dimmed">Loading {name}…</Text>;

  const framesById = Object.fromEntries((storyboard?.frames || []).map((f) => [f.id, f]));
  const totalFrames = (storyboard?.frames || []).length;
  const taggedFrames = (storyboard?.frames || []).filter(
    (f) => (f.people || []).length
  ).length;
  const clipsById = Object.fromEntries((snap.clips || []).map((c) => [c.id, c]));
  // Transitions the backend planner never succeeded on (still carrying the
  // config fallback prompt) — flagged so nobody renders a whole order generic.
  const placeholderIds = new Set(snap.storyboard?.placeholder_transitions || []);
  // Prompts written before the tags/cast they should have used. Tagging and
  // renaming change nothing on their own, so this is the only way to see
  // which prompts are behind without remembering what you touched.
  const outdatedPlans = new Set(snap.storyboard?.outdated_plans || []);
  const chip = stepChip(snap.next_step);
  const activeJob = (snap.jobs || []).find((j) => ['running', 'queued', 'cancelling'].includes(j.state));
  const locked = Boolean(activeJob);
  // A running storyboard/run job is what produces styled frames.
  const styling = ['storyboard', 'run'].includes(activeJob?.command);

  // Facts every confirmation modal builds on.
  const inputImages = snap.input_images || [];
  const styledImages = snap.styled_images || [];
  const total = (snap.clips || []).length;
  const rendered = (snap.clips || []).filter((c) => c.rendered).length;
  const info = {
    photoCount: inputImages.length,
    unstyledCount: inputImages.filter(
      (img) => !styledImages.includes(img.replace(/\.[^.]+$/, '.png'))
    ).length,
    total,
    rendered,
    missing: total - rendered,
    outdated: (snap.clips || []).filter((c) => c.rendered && c.stale).length,
    needsRender: clipsNeedingRender(snap).length,
    silentRendered: (snap.clips || []).filter((c) => c.rendered && !c.sfx).length,
    finalExists: Boolean(snap.final_video),
    customMusic: Boolean(snap.custom_music),
    // 0 when no letter is written: the Closing letter toggle is a no-op then.
    letterChars: snap.letter?.chars || 0,
    // Delivery state from published.json (no network): {count, latest,
    // changed_since, publishable}.
    published: snap.published || {},
    // The identity review step (step 2).
    totalFrames,
    taggedFrames,
    outdatedPlans: (snap.storyboard?.outdated_plans || []).length,
    hasCast: (storyboard?.characters || []).length > 0,
    fragileEpithets: (snap.storyboard?.fragile_epithets || []).length
  };

  const stepStatus = (id) => {
    const published = info.published || {};
    const done = {
      storyboard: Boolean(storyboard),
      // Every photo answered for. Tagging is never mandatory, so this step
      // is 'optional' rather than 'todo' until then — an untagged movie is
      // a perfectly normal one, just planned from the model's own guesses.
      people: totalFrames > 0 && taggedFrames === totalFrames,
      render: total > 0 && rendered === total,
      audio: rendered > 0 && info.silentRendered === 0,
      combine: info.finalExists,
      // Delivered, and the movie hasn't been rebuilt since.
      publish: published.count > 0 && !published.changed_since
    }[id];
    if (done) return 'done';
    if (snap.next_step === id) return 'next';
    // A hand-made project has no order folder, so publishing never applies.
    if (id === 'publish' && !published.publishable) return 'optional';
    return ['audio', 'people'].includes(id) ? 'optional' : 'todo';
  };

  const run = async (command, options = {}, label = command) => {
    setBusyAction(label);
    try {
      await api.runAction(name, command, options);
      notify(`${label} started`);
      setOpenPanel('');
      await load();
    } catch (e) { notify(`${label} failed: ${e.message}`, 'red'); }
    finally { setBusyAction(''); }
  };

  // The verification step: every mutating action lands here first.
  const ask = (cfg) => setConfirm(cfg);
  const confirmed = async () => {
    const { command, options = {}, label, action } = confirm;
    setConfirm(null);
    if (action) return action();
    return run(command, options, label || command);
  };

  // One job for every clip that is missing or outdated — the batch you want
  // after editing a handful of motion prompts, instead of clicking through
  // the clips one at a time. Listing `clips` FORCES those clips, so the
  // outdated ones are re-rendered rather than skipped as "done".
  const generateAll = (data) => {
    const todo = clipsNeedingRender(data);
    if (!todo.length) {
      notify('Every clip is rendered and up to date — nothing to generate.', 'green');
      return;
    }
    const missing = todo.filter((c) => !c.rendered).map((c) => c.id);
    const outdated = todo.filter((c) => c.rendered).map((c) => c.id);
    ask({
      title: `Generate ${todo.length} clip(s) in one job?`,
      lines: [
        ...(missing.length
          ? [`${missing.length} never rendered: ${missing.join(', ')}`] : []),
        ...(outdated.length
          ? [`${outdated.length} outdated — REPLACED with the current motion prompt: ${outdated.join(', ')}`]
          : []),
        `${CLIP_PRICE} — up to ${todo.length} clip(s) this run.`,
        'Every other rendered clip is left untouched.',
        ...(outdated.length
          ? ['Regenerated clips lose their SFX, so run Audio and Combine again afterwards.']
          : [])
      ],
      cost: 'fal',
      danger: outdated.length > 0,
      label: `Generate ${todo.length}`,
      action: () => run('render', { clips: todo.map((c) => c.id) },
        `render ${todo.length} clip(s)`)
    });
  };

  const saveEdits = async () => {
    setBusyAction('save');
    try {
      const res = await api.saveStoryboard(name, storyboard);
      setDirty(new Set());
      const marked = (res.outdated || []).length;
      notify(marked
        ? `Storyboard saved — ${marked} rendered clip(s) marked outdated`
        : 'Storyboard saved', 'green');
      // Editing a clip that had a paid render still waiting on the provider
      // throws that render away (its fingerprint no longer matches), so the
      // loss is called out rather than absorbed silently.
      const orphaned = res.orphaned_renders || [];
      if (orphaned.length) {
        notify(
          `${orphaned.length} already-paid render(s) can no longer be collected: `
          + `${orphaned.join(', ')} — your edit changed their plan, so rendering `
          + 'them now buys fresh clips.', 'red');
      }
      const data = await load();
      // Saving edits is normally step one of "now re-render them", so offer
      // the whole batch straight away. It is still a confirmation modal:
      // nothing is spent until it is accepted.
      generateAll(data);
    } catch (e) { notify(`Save failed: ${e.message}`, 'red'); }
    finally { setBusyAction(''); }
  };

  const editTransition = (tr) => {
    setStoryboard((sb) => ({
      ...sb, transitions: sb.transitions.map((t) => (t.id === tr.id ? tr : t))
    }));
    setDirty((d) => new Set(d).add(tr.id));
  };

  const editGlobalMotion = (value) => {
    setStoryboard((sb) => ({ ...sb, global_motion_prompt: value }));
    setDirty((d) => new Set(d).add(GLOBAL_EDIT));
  };

  // Identity tags, like cast epithets, are a PLANNING input: they change what
  // the planner is told next time, not the prompt a clip was already rendered
  // from — so tagging never marks a clip outdated. Re-plan a clip to apply it.
  const editFramePeople = (outputPath, people) => {
    setStoryboard((sb) => ({
      ...sb,
      frames: (sb.frames || []).map(
        (f) => (f.output_path === outputPath ? { ...f, people } : f)
      ),
    }));
    setDirty((d) => new Set(d).add(TAGS_EDIT));
  };

  const askForTagSuggestions = () => {
    const untagged = (storyboard?.frames || []).filter(
      (f) => !(f.people || []).length
    ).length;
    ask({
      title: 'Let the AI propose who is in each photo?',
      lines: [
        `Looks at ${untagged} untagged frame(s) and proposes which cast member is in each, and where.`,
        'Frames you have already tagged are left exactly as they are — your corrections are never overwritten.',
        'One OpenAI vision call. Nothing is re-planned or re-rendered: tags apply to plans made from here on.',
        'It is a draft — check it afterwards, especially wherever two people look alike.'
      ],
      cost: 'openai',
      label: 'Propose tags',
      command: 'tag', options: {}
    });
  };

  // Cast epithets feed FUTURE planning calls (they are baked into motion
  // prompts when a pair is planned), so editing one never marks a rendered
  // clip outdated — re-plan a clip to pick up a changed epithet.
  const editCharacter = (id, epithet) => {
    setStoryboard((sb) => ({
      ...sb,
      characters: (sb.characters || []).map(
        (c) => (c.id === id ? { ...c, epithet } : c)
      )
    }));
    setDirty((d) => new Set(d).add(CAST_EDIT));
  };

  const needsSave = () => {
    if (dirty.size) { notify('Save your storyboard edits first.', 'yellow'); return true; }
    return false;
  };

  const regenerate = (clipId) => {
    if (needsSave()) return;
    const exists = clipsById[clipId]?.rendered;
    ask({
      title: exists ? `Regenerate clip ${clipId}?` : `Render clip ${clipId}?`,
      lines: [
        exists
          ? `Re-renders this one clip with its current motion prompt and REPLACES clips/${clipId}.mp4.`
          : 'Renders this one clip with its current motion prompt.',
        CLIP_PRICE,
        exists ? 'Its "outdated" mark (if any) is cleared; its SFX will need redoing afterwards.' : 'No other clip is touched.'
      ],
      cost: 'fal', danger: exists,
      label: exists ? 'Regenerate' : 'Render',
      action: () => run('render', { clips: [clipId] }, `render ${clipId}`)
    });
  };

  // Fetch a render that was already submitted and billed. It runs the same
  // render action, but the pipeline resumes the persisted request_id instead
  // of submitting again — so this costs nothing as long as the plan is
  // unchanged, and the modal must not claim otherwise.
  const collectRender = (pending) => {
    if (needsSave()) return;
    ask({
      title: `Collect the finished render for ${pending.id}?`,
      lines: [
        'This clip was already submitted to the video provider and paid for; its result is waiting to be downloaded.',
        'Rendering it now fetches that result instead of buying a new clip — as long as its frames, prompt and duration are still unchanged.',
        'Editing the storyboard first would invalidate it and force a fresh (paid) render.'
      ],
      cost: 'free',
      label: 'Collect',
      action: () => run('render', { clips: [pending.id] }, `render ${pending.id}`)
    });
  };

  const replanPrompt = (clipId) => {
    if (needsSave()) return;
    ask({
      title: `Re-plan the prompt for ${clipId}?`,
      lines: [
        'Asks the AI planner to look at this pair of frames again and write a fresh motion prompt (one small vision call).',
        'Only this transition changes — every other prompt and edit is kept.',
        'The clip itself is NOT re-rendered; if it already exists it gets marked "outdated" so you can regenerate it when the new prompt looks right.'
      ],
      cost: 'openai',
      label: 'Re-plan prompt',
      action: () => run('storyboard', { replan_clips: [clipId] }, `re-plan ${clipId}`)
    });
  };

  // The batch you want after tagging (or after rewriting cast epithets):
  // both are plan-time inputs, so nothing already planned uses them until
  // its pair is planned again. One job re-plans every pair — a single vision
  // call over the whole movie, like the original storyboard run.
  const replanOutdated = () => {
    if (needsSave()) return;
    const ids = snap.storyboard?.outdated_plans || [];
    if (!ids.length) return;
    ask({
      title: `Re-plan the ${ids.length} prompt(s) that are behind?`,
      lines: [
        `These were written before your current tags or cast names: ${ids.join(', ')}.`,
        'Every other prompt is left exactly as it is — including any you wrote yourself.',
        'One vision call. Clips whose plan changes are marked "outdated"; nothing is re-rendered or deleted here.'
      ],
      cost: 'openai',
      label: `Re-plan ${ids.length}`,
      action: () => run('storyboard', { replan_clips: ids },
        `re-plan ${ids.length} prompt(s)`)
    });
  };

  const replanAll = () => {
    if (needsSave()) return;
    const ids = (storyboard?.transitions || []).map((t) => t.id);
    if (!ids.length) return;
    const rendered = (snap.clips || []).filter((c) => c.rendered).length;
    ask({
      title: `Re-plan all ${ids.length} motion prompt(s)?`,
      lines: [
        `Asks the planner to look at every pair again and write fresh motion prompts, using the current cast and whoever you have tagged in each photo.`,
        'Any motion prompt you hand-wrote is REPLACED — re-plan is not a merge.',
        rendered > 0
          ? `${rendered} already-rendered clip(s) whose plan changes get marked "outdated"; none are re-rendered here, and nothing is deleted.`
          : 'No clips are rendered yet, so nothing is invalidated.',
        'One vision call over the whole movie — the same cost as the original storyboard run.'
      ],
      cost: 'openai',
      danger: true,
      label: `Re-plan ${ids.length}`,
      action: () => run('storyboard', { replan_clips: ids }, 'replan all')
    });
  };

  const redoAudio = (clipId) => {
    if (needsSave()) return;
    ask({
      title: `Redo audio for ${clipId}?`,
      lines: [
        'Regenerates this clip\'s sound effects from its sound prompt and muxes them in.',
        'Audio jobs are much cheaper than clip renders. The final video needs a Combine afterwards to pick the new sound up.'
      ],
      cost: 'fal',
      label: 'Redo audio',
      action: () => run('audio', { clips: [clipId] }, `audio ${clipId}`)
    });
  };

  // Re-style one frame from scratch (spends image credits). Runs the
  // storyboard command with restyle_frames; reconcile then marks the
  // adjacent clips outdated (never deletes them).
  const regenerateImage = (inputName) => {
    if (needsSave()) return;
    const styledName = inputName.replace(/\.[^.]+$/, '.png');
    ask({
      title: `Regenerate the styled image for ${inputName}?`,
      lines: [
        `Re-styles ${inputName} into a fresh cartoon frame, replacing styled_images/${styledName}.`,
        'Spends OpenAI image credits (one image, plus a small vision call to re-plan the motion prompts on each side of this frame).',
        'Any clips next to this frame are marked "outdated" so you can regenerate them against the new image — they are never deleted automatically.'
      ],
      cost: 'openai', danger: true,
      label: 'Regenerate image',
      action: () => run('storyboard', { restyle_frames: [styledName] }, `restyle ${styledName}`)
    });
  };

  // Publishing is the one action whose target name is decided by what is
  // ALREADY in Cloudinary, so the modal can't be written from the local
  // snapshot: ask the server first (read-only), show the real file name for
  // approval, then pin that exact name into the job (`publish_as`) so the
  // upload can only ever land under the name that was approved.
  const startPublish = async () => {
    setBusyAction('publish-preview');
    try {
      const plan = await api.publishPreview(name);
      if (!plan.final_video) {
        notify('There is no final video to publish — run Combine first.', 'yellow');
        return;
      }
      const mb = (plan.bytes / (1024 * 1024)).toFixed(1);
      const versions = (plan.published || []).map((p) => `v${p.version}`).join(', ');
      ask({
        title: 'Publish the movie to the order folder?',
        lines: [
          `Uploads output/final_video.mp4 (${mb} MB) into the Cloudinary order folder ${plan.order_folder}.`,
          `It will be saved as:  ${plan.filename}   (full name: ${plan.public_id}.mp4)`,
          versions
            ? `Already published there: ${versions} — those files are kept exactly as they are. Nothing is replaced or deleted.`
            : 'Nothing has been published for this order yet, so this is version 1.',
          ...(plan.latest && !plan.changed_since_last
            ? [`The final video has not changed since v${plan.latest.version} — this publishes the same movie again as a new version.`]
            : []),
          ...(plan.keeps_local_copy
            ? [`A copy of exactly what is delivered is also kept here, as output/published/${plan.filename}.`]
            : []),
          'Free of API credits; it uses your Cloudinary storage.'
        ],
        cost: 'free',
        label: `Publish ${plan.filename}`,
        action: () => run('publish', { publish_as: plan.public_id }, 'publish')
      });
    } catch (e) { notify(`Could not prepare the publish: ${e.message}`, 'red'); }
    finally { setBusyAction(''); }
  };

  // Feedback runs inline (it is one short call, and the serial job runner may
  // be halfway through a render). The form stays open on failure so a note
  // nobody wants to retype isn't lost to a network blip.
  const sendFeedback = async ({ note, verdict, learn, review }) => {
    setFeedbackBusy(true);
    try {
      const res = await api.sendFeedback(name, {
        clip: feedbackFor?.id || '', note, verdict, learn, review,
      });
      if (res.review?.observed || res.review_error) {
        // The AI has something to show: keep the modal open on its verdict
        // and the prompt it proposes, instead of a toast that scrolls away.
        setFeedbackResult(res);
      } else {
        setFeedbackFor(null);
        if (res.learn_error) {
          notify(
            `Note saved, but the rule could not be written (${res.learn_error}). `
            + 'You can add it by hand in the Learning tab.', 'yellow');
        } else if (res.lessons?.length) {
          notify(
            `Thanks — the planner learned: “${res.lessons[0].text}”`, 'green');
        } else if (learn) {
          notify(
            'Note saved. There was nothing general enough in it to turn into a '
            + 'rule — say what the video model DID wrong for that.', 'gray');
        } else {
          // Learning was switched off on purpose: don't imply the note was
          // too vague to teach anything.
          notify('Note saved — no rule was written from it.', 'gray');
        }
      }
      await load();
    } catch (e) { notify(`Feedback failed: ${e.message}`, 'red'); }
    finally { setFeedbackBusy(false); }
  };

  const closeFeedback = () => { setFeedbackFor(null); setFeedbackResult(null); };
  // Always open on a blank form: without clearing the previous result, the
  // next clip's modal would open showing the last clip's review.
  const openFeedback = (tr) => { setFeedbackResult(null); setFeedbackFor(tr); };

  // Accepting the AI's fix is an ordinary storyboard edit — the same one you
  // would type yourself. It becomes an unsaved change; saving marks the clip
  // outdated and offers the (confirmed, paid) re-render. Nothing is spent
  // here, and nothing is applied without this click.
  const applySuggestion = (review) => {
    const target = feedbackFor;
    closeFeedback();
    if (!target) return;
    const current = (storyboard?.transitions || []).find((t) => t.id === target.id);
    if (!current) return;
    editTransition({
      ...current,
      motion_prompt: review.suggested_motion_prompt,
      duration: review.suggested_duration || current.duration,
    });
    notify(
      'Applied to the clip. Save your edits, then regenerate it to see the '
      + 'new version.', 'green');
  };

  // Fetching can take a few seconds (a page URL is extracted + transcoded
  // server-side), so it shares the same busy flag as the file upload.
  const fetchMusicUrl = async (url, onDone) => {
    setMusicBusy(true);
    try {
      await api.fetchMusicUrl(name, url);
      notify('Music track fetched', 'green');
      onDone?.();
      await load();
    } catch (e) { notify(`Fetch failed: ${e.message}`, 'red'); }
    finally { setMusicBusy(false); }
  };

  // Saving the letter only writes a text file — no job, no credits — so it
  // runs inline like the storyboard save rather than through a confirmation.
  const saveLetter = async (text) => {
    setLetterBusy(true);
    try {
      const res = await api.saveLetter(name, text);
      notify(res.letter.chars
        ? `Letter saved (${res.letter.chars} characters)`
        : 'Letter cleared — the movie will end without one', 'green');
      await load();
    } catch (e) {
      notify(`Saving the letter failed: ${e.message}`, 'red');
      throw e;
    } finally { setLetterBusy(false); }
  };

  const uploadMusic = async (file) => {
    if (!file) return;
    setMusicBusy(true);
    try {
      await api.uploadMusic(name, file);
      notify('Music track uploaded', 'green');
      await load();
    } catch (e) { notify(`Music upload failed: ${e.message}`, 'red'); }
    finally { setMusicBusy(false); }
  };

  const removeMusic = () => {
    ask({
      title: 'Remove the music track?',
      lines: [
        'Deletes the uploaded music track from this project.',
        'The movie will then have no background music at all until you upload another track.'
      ],
      cost: 'free', danger: true,
      label: 'Remove music',
      action: async () => {
        try {
          await api.deleteMusic(name);
          notify('Music track removed');
          await load();
        } catch (e) { notify(`Remove failed: ${e.message}`, 'red'); }
      }
    });
  };

  const upload = async (files) => {
    if (!files?.length) return;
    setUploading(true);
    try {
      const res = await api.uploadPhotos(name, Array.from(files));
      notify(`Uploaded ${res.saved.length} photo(s)`, 'green');
      setShowPhotos(true);
      await load();
    } catch (e) { notify(`Upload failed: ${e.message}`, 'red'); }
    finally { setUploading(false); if (fileInputRef.current) fileInputRef.current.value = ''; }
  };

  const deletePhoto = (filename) => {
    ask({
      title: `Delete photo ${filename}?`,
      lines: [
        'Removes this photo from the movie: the input file is deleted from input_images/.',
        'Its styled frame and any rendered clips stay on disk; the next Storyboard run re-plans the pairs around the gap.'
      ],
      cost: 'free', danger: true,
      label: 'Delete',
      action: async () => {
        try {
          await api.deletePhoto(name, filename);
          notify(`Deleted ${filename}`);
          await load();
        } catch (e) { notify(`Delete failed: ${e.message}`, 'red'); }
      }
    });
  };

  const showLog = async (jobId) => {
    try { setLogJob(await api.job(jobId)); }
    catch (e) { notify(`Log failed: ${e.message}`, 'red'); }
  };

  const cancelJob = async (jobId) => {
    setBusyAction(`cancel ${jobId}`);
    try {
      const res = await api.cancelJob(jobId);
      notify(res.job.state === 'cancelling'
        ? 'Cancelling — the clip being generated will finish, then the job stops.'
        : 'Job cancelled.');
      await load();
    } catch (e) { notify(`Cancel failed: ${e.message}`, 'red'); }
    finally { setBusyAction(''); }
  };

  const statusBadge = (status) =>
    status === 'done' ? <Badge variant="light" color="green" size="sm">✓ done</Badge>
      : status === 'next' ? <Badge variant="light" color="orange" size="sm">next step</Badge>
        : status === 'optional' ? <Text size="xs" c="dimmed">optional</Text>
          : null;

  const panels = {
    storyboard: <StoryboardPanel ask={ask} locked={locked} info={info}
      onReplanAll={replanAll} />,
    people: <PeoplePanel locked={locked} info={info}
      onPropose={askForTagSuggestions} onReplanAll={replanAll}
      onReplanOutdated={replanOutdated}
      onOpenTagger={() => { setShowTagger(true); setOpenPanel(''); }} />,
    render: <RenderPanel ask={ask} locked={locked} info={info}
      onGenerateAll={() => { if (!needsSave()) generateAll(snap); }} />,
    audio: <AudioPanel ask={ask} locked={locked} info={info}
      onUploadMusic={uploadMusic} onRemoveMusic={removeMusic}
      onFetchMusicUrl={fetchMusicUrl} musicBusy={musicBusy} />,
    combine: <CombinePanel ask={ask} locked={locked} info={info}
      onUploadMusic={uploadMusic} onRemoveMusic={removeMusic}
      onFetchMusicUrl={fetchMusicUrl} musicBusy={musicBusy}
      letterText={snap.letter_text || ''} onSaveLetter={saveLetter}
      letterBusy={letterBusy} />,
    publish: <PublishPanel project={name} locked={locked} info={info}
      onPublish={startPublish}
      publishing={busyAction === 'publish-preview' || busyAction === 'publish'} />,
    runall: <RunAllPanel ask={ask} locked={locked} info={info} />
  };

  return (
    <Stack gap="sm">
      <ConfirmModal confirm={confirm} onCancel={() => setConfirm(null)} onConfirm={confirmed} />

      <FeedbackModal opened={Boolean(feedbackFor)} clipId={feedbackFor?.id}
        motionPrompt={feedbackFor?.motion_prompt}
        duration={feedbackFor?.duration} busy={feedbackBusy}
        result={feedbackResult} onApply={applySuggestion}
        onClose={closeFeedback} onSubmit={sendFeedback} />

      <Modal opened={Boolean(logJob)} onClose={() => setLogJob(null)} centered size="xl"
        title={logJob ? `${logJob.command} — ${logJob.state}` : ''}>
        {logJob?.error && <Text c="red" size="sm" mb="sm">{logJob.error}</Text>}
        <ScrollArea.Autosize mah={360}>
          <Text component="pre" size="xs" c="dimmed" style={{ margin: 0 }}>
            {(logJob?.log || []).join('\n') || '(no log lines)'}
          </Text>
        </ScrollArea.Autosize>
      </Modal>

      <Group gap="sm">
        <Button variant="subtle" size="xs" onClick={onBack}>← back</Button>
        <Title order={3}>{name}</Title>
        <Badge variant="light" color={chip.color}>{chip.label}</Badge>
        {snap.order?.customer && (
          <Text size="sm" c="dimmed">{snap.order.customer} · {snap.order.order_id}</Text>
        )}
        {activeJob && (
          <Badge variant="light" color={activeJob.state === 'cancelling' ? 'yellow' : 'blue'}>
            {activeJob.command} {activeJob.state}…
          </Badge>
        )}
      </Group>

      <Card withBorder padding="sm">
        <Group gap="xs" align="stretch">
          {STEPS.map((step) => {
            const status = stepStatus(step.id);
            const selected = openPanel === step.id;
            return (
              <StepTile key={step.id} selected={selected} highlight={status === 'next'}
                onClick={() => setOpenPanel(selected ? '' : step.id)}>
                <Group justify="space-between" gap={6} wrap="nowrap">
                  <Text size="sm" fw={700}>
                    <Text span c="dimmed" fw={400}>{step.n} · </Text>{step.name}
                  </Text>
                  {statusBadge(status)}
                </Group>
                <Text size="xs" c="dimmed" mt={2}>{step.caption}</Text>
              </StepTile>
            );
          })}
          <StepTile dashed selected={openPanel === 'runall'}
            onClick={() => setOpenPanel(openPanel === 'runall' ? '' : 'runall')}>
            <Text size="sm" fw={700} c="dimmed">⚡ Run everything</Text>
            <Text size="xs" c="dimmed" mt={2}>All steps, no stops</Text>
          </StepTile>
        </Group>
        {openPanel && (
          <>
            <Divider my="sm" />
            {panels[openPanel]}
          </>
        )}
      </Card>

      <SpendCard project={name} cost={snap.cost} />

      {/* The last run's failures. The pipeline survives individual failures
          on purpose, so a command can finish having produced nothing — this
          is what stops that from reading as success. */}
      {(snap.failed_jobs || []).length > 0 && (
        <Alert color="red" variant="light"
          title={`Last run had ${snap.failed_jobs.length} failure(s) — nothing was produced for these`}>
          <Stack gap={6}>
            {snap.failed_jobs.slice(0, 6).map((f, i) => (
              <div key={`${f.id}-${i}`}>
                <Text size="sm" fw={600}>
                  {f.kind ? `${f.kind}: ` : ''}{f.id.replace(/^clip:/, '')}
                </Text>
                <Text size="xs" c="dimmed">{f.error}</Text>
              </div>
            ))}
            {snap.failed_jobs.length > 6 && (
              <Text size="xs" c="dimmed">
                …and {snap.failed_jobs.length - 6} more.
              </Text>
            )}
            <Text size="xs" c="dimmed">
              These clips were NOT regenerated — the files on disk are still the
              previous versions, which is why they remain marked outdated. This
              clears when a run finishes without failures.
            </Text>
          </Stack>
        </Alert>
      )}

      {/* Renders submitted (and billed) but not yet downloaded. Split by
          in_flight: a clip the RUNNING job is polling for collects itself
          and is just progress, while one left behind by an interrupted run
          is money sitting on the provider that only you can rescue. Showing
          both as "waiting" made every healthy render look like an alarm. */}
      {(snap.pending_renders || []).length > 0 && (() => {
        // The server decides this (it owns the job runner), but fall back to
        // the active job when talking to a backend that predates the field:
        // the panel is served from disk and can be newer than the process,
        // so it must never depend on a field the API might not send yet.
        const renderRunning = ['render', 'run'].includes(activeJob?.command);
        const isInFlight = (r) =>
          r.in_flight === undefined ? renderRunning : r.in_flight;
        const inFlight = snap.pending_renders.filter(isInFlight);
        const stranded = snap.pending_renders.filter((r) => !isInFlight(r));
        return (
          <Alert
            color={stranded.some((r) => !r.recoverable) ? 'red'
              : stranded.length ? 'blue' : 'gray'}
            variant="light"
            title={stranded.length
              ? `${stranded.length} paid render(s) waiting to be collected`
              : `${inFlight.length} clip(s) rendering on the provider`}>
            <Stack gap={6}>
              {inFlight.map((r) => (
                <Group key={r.clip} gap="xs" wrap="nowrap">
                  <Text size="sm" fw={600}>{r.id}</Text>
                  <Text size="xs" c="dimmed" style={{ flex: 1 }}>
                    rendering now (submitted {localTime(r.submitted_at)}) — the
                    running job downloads it automatically when it finishes.
                  </Text>
                </Group>
              ))}
              {stranded.map((r) => (
                <Group key={r.clip} gap="xs" wrap="nowrap">
                  <Text size="sm" fw={600}>{r.id}</Text>
                  <Text size="xs" c="dimmed" style={{ flex: 1 }}>
                    {r.recoverable
                      ? `submitted ${localTime(r.submitted_at)} — already paid for; collecting fetches it instead of buying a new one`
                      : 'its frames/prompt/duration changed since it was submitted, so this paid render can no longer be collected — rendering buys a fresh clip'}
                    {r.recoverable && locked && ' (another job is running — wait for it to finish)'}
                  </Text>
                  {r.recoverable && (
                    <Button size="compact-xs" variant="light" disabled={locked}
                      loading={busyAction === `render ${r.id}`}
                      onClick={() => collectRender(r)}>
                      Collect
                    </Button>
                  )}
                </Group>
              ))}
            </Stack>
          </Alert>
        );
      })()}

      {snap.final_video && (
        <Card withBorder padding="md">
          <Text fw={600} mb="sm">Final video</Text>
          <Group align="center">
            <video controls preload="metadata"
              style={{ width: 420, maxWidth: '100%', borderRadius: 8, background: '#000' }}
              src={fileUrl(name, 'output', 'final_video.mp4', mediaV)} />
            <Button component="a" variant="light"
              href={fileUrl(name, 'output', 'final_video.mp4', mediaV)} download={`${name}.mp4`}>
              Download
            </Button>
            {info.published.publishable && (
              <Button variant="default" disabled={locked}
                loading={busyAction === 'publish-preview' || busyAction === 'publish'}
                onClick={startPublish}
                title="Upload this movie into the customer's Cloudinary order folder as the next version">
                Publish to Cloudinary…
              </Button>
            )}
          </Group>
          {info.published.latest && (
            <Text size="xs" c="dimmed" mt="xs">
              Published as v{info.published.latest.version}
              {info.published.latest.published_at
                ? ` on ${localTime(info.published.latest.published_at)}` : ''}
              {info.published.changed_since
                ? ' — this movie has been rebuilt since, so it is newer than what the customer folder holds.'
                : ' — the order folder holds exactly this movie.'}
            </Text>
          )}
        </Card>
      )}

      <Card withBorder padding="md">
        <Group gap="sm">
          <Text fw={600}>Photos ({inputImages.length})</Text>
          {/* Styled frames land one file at a time and the panel polls every
              3s, so the grid below fills in DURING the storyboard run instead
              of staying empty until it finishes. */}
          {styling && (
            <Badge variant="light" color="blue">
              styling {styledImages.length} / {inputImages.length}
            </Badge>
          )}
          <Text size="xs" c="dimmed" style={{ flex: 1 }}>
            {styling
              ? 'Frames appear here as they come back from the styler.'
              : 'Movie order follows the filenames (sorted). Styled versions shown when available.'}
          </Text>
          <input ref={fileInputRef} type="file" accept="image/*" multiple
            style={{ display: 'none' }} onChange={(e) => upload(e.target.files)} />
          <Button variant="default" size="xs" loading={uploading}
            onClick={() => fileInputRef.current?.click()}>Add photos</Button>
          <Button variant="subtle" size="xs" onClick={() => setShowPhotos((v) => !v)}>
            {showPhotos ? 'Hide' : 'Show'}
          </Button>
        </Group>
        {showPhotos && (
          <Group mt="md" gap="sm" align="flex-start">
            {inputImages.map((img) => {
              const styledName = img.replace(/\.[^.]+$/, '.png');
              const styled = styledImages.includes(styledName);
              return (
                <Stack key={img} gap={2} w={120}>
                  <Image w={120} radius="sm" alt={img}
                    style={styling && !styled ? { opacity: 0.45 } : undefined}
                    src={fileUrl(name, styled ? 'styled' : 'input',
                      styled ? styledName : img, styled ? mediaV : 0)} />
                  <Group justify="space-between" gap={4} wrap="nowrap">
                    <Text size="xs" c="dimmed" truncate title={img}>
                      {styling && !styled ? 'styling…' : img}
                    </Text>
                    <UnstyledButton title={`Delete ${img}`} onClick={() => deletePhoto(img)}
                      style={{ color: 'var(--mantine-color-red-5)', fontSize: 13 }}>
                      ✕
                    </UnstyledButton>
                  </Group>
                  {styled && (
                    <Button variant="subtle" size="compact-xs" disabled={locked}
                      loading={busyAction === `restyle ${styledName}`}
                      onClick={() => regenerateImage(img)}
                      title="Re-style this photo into a fresh cartoon frame (spends image credits). Adjacent clips are marked outdated.">
                      Regenerate image
                    </Button>
                  )}
                </Stack>
              );
            })}
            {inputImages.length === 0 && (
              <Text size="sm" c="dimmed">
                No photos yet — add some above, or run Storyboard with an idea instead.
              </Text>
            )}
          </Group>
        )}
      </Card>

      {(snap.jobs || []).length > 0 && (
        <Card withBorder padding="md">
          <Text fw={600} mb={4}>Jobs</Text>
          {(snap.jobs || []).map((j) => (
            <JobRow key={j.id} job={j} onShowLog={showLog} onCancel={cancelJob}
              cancelBusy={busyAction === `cancel ${j.id}`} />
          ))}
        </Card>
      )}

      {storyboard && (
        <>
          <Group mt="md" gap="md">
            <Title order={4}>Storyboard — {storyboard.transitions.length} clips</Title>
            {dirty.size > 0 && (
              <Button size="xs" loading={busyAction === 'save'} onClick={saveEdits}>
                Save {dirty.size} edit{dirty.size > 1 ? 's' : ''}
              </Button>
            )}
          </Group>
          {placeholderIds.size > 0 && (
            <Alert color="red" variant="light"
              title={`${placeholderIds.size} of ${storyboard.transitions.length} transitions still have the generic fallback prompt`}>
              Planning failed for them (OpenAI quota or rate limit) — run Storyboard
              again to re-plan exactly these. Clips rendered from the generic prompt
              lose their tailored motion.
            </Alert>
          )}
          <Card withBorder padding="md">
            <Textarea
              label="Global motion prompt"
              description={'Prepended to every clip — whole-movie facts, e.g. "two separate people appear throughout; never blend them"; keep it to a sentence or two'}
              placeholder="(none)" autosize minRows={2}
              value={storyboard.global_motion_prompt || ''}
              onChange={(e) => editGlobalMotion(e.target.value)} />
          </Card>
          {(storyboard.characters || []).length > 0 && (() => {
            // Epithets that name clothing. The whole movie is built from
            // photos taken far apart, so "the boy in the striped shirt"
            // matches nobody in the next frame — the video model looks for a
            // striped shirt, doesn't find one, and acts on whoever is
            // nearest. New plans can't produce these any more; a cast written
            // before that is left exactly as it is (its wording is baked into
            // prompts already planned) and flagged here instead.
            const fragile = new Set(snap.storyboard?.fragile_epithets || []);
            return (
              <Card withBorder padding="md">
                <Text size="sm" fw={500}>Cast</Text>
                <Text size="xs" c="dimmed" mb="xs">
                  How motion prompts name each person — the video model sees
                  only pixels, so every mention uses these exact words. Name
                  what a person carries from photo to photo (age or relative
                  size, hair, beard, glasses), never what they happen to be
                  wearing. Edits apply to clips planned from now on: re-plan a
                  clip to pick up a change.
                </Text>
                {fragile.size > 0 && (
                  <Alert color="orange" variant="light" mb="xs"
                    title={`${fragile.size} name${fragile.size > 1 ? 's' : ''} `
                      + 'identify someone by their clothes'}>
                    These were written before the planner was taught not to.
                    They only work in the photo they came from — in every other
                    frame that shirt is gone, and the video model puts the
                    action on the wrong person. Rewrite them as something the
                    person keeps (“the smaller boy with curly hair”, “the
                    taller boy”), then re-plan the clips you want to use the
                    new wording.
                  </Alert>
                )}
                <Stack gap="xs">
                  {(storyboard.characters || []).map((c) => (
                    <TextInput key={c.id} size="xs" label={c.id} value={c.epithet}
                      error={fragile.has(c.id)
                        ? 'names clothing — use a feature that does not change'
                        : undefined}
                      onChange={(e) => editCharacter(c.id, e.target.value)} />
                  ))}
                </Stack>
              </Card>
            );
          })()}
          {(storyboard.characters || []).length > 0 && (
            <Card withBorder padding="md">
              <Group gap="sm" mb={4}>
                <Text size="sm" fw={500}>Who's in each photo</Text>
                <Badge variant="light"
                  color={taggedFrames === totalFrames ? 'green' : 'gray'}>
                  {taggedFrames}/{totalFrames} tagged
                </Badge>
                <div style={{ flex: 1 }} />
                <Button size="compact-xs" variant="default" disabled={locked}
                  onClick={askForTagSuggestions}
                  title="One vision call proposes who is in each untagged photo; you correct it. Already-tagged photos are untouched.">
                  Let the AI propose…
                </Button>
                <Button size="compact-xs" variant="default" disabled={locked}
                  onClick={replanAll}
                  title="Tags only reach the planner when a pair is planned again. This re-plans every pair with them (one vision call); clips whose plan changes are marked outdated, never re-rendered automatically.">
                  Re-plan all with these tags…
                </Button>
                <Button size="compact-xs" variant="subtle"
                  onClick={() => setShowTagger((v) => !v)}>
                  {showTagger ? 'Hide' : 'Show'}
                </Button>
              </Group>
              <Text size="xs" c="dimmed">
                Pick a person, then click their face in the photo. This tells
                the planner who is who as FACT — that the child here is the
                same child as in the last photo, or a different one who merely
                looks alike — which is the judgement it gets wrong on its own
                and the reason two people sometimes morph into each other.
                Free; applies to clips planned from now on, so re-plan a clip
                to use it.
              </Text>
              {showTagger && (
                <Stack gap="sm" mt="sm">
                  {(storyboard.frames || []).map((f) => (
                    <FrameTagger key={f.output_path} project={name} frame={f}
                      cast={storyboard.characters || []} mediaV={mediaV}
                      onChange={(people) => editFramePeople(f.output_path, people)} />
                  ))}
                </Stack>
              )}
            </Card>
          )}
          {storyboard.transitions.map((tr) => (
            <TransitionCard key={tr.id} project={name} tr={tr} framesById={framesById}
              clip={clipsById[tr.output_path.split('/').pop()?.replace(/\.mp4$/, '')]}
              edited={dirty.has(tr.id)} placeholder={placeholderIds.has(tr.id)}
              verdict={(snap.feedback?.by_transition || {})[tr.id]}
              planBehind={outdatedPlans.has(tr.id)}
              onEdit={editTransition}
              onRegenerate={regenerate} onReplan={replanPrompt} onRedoAudio={redoAudio}
              onFeedback={openFeedback}
              busy={busyAction === `render ${tr.id}`}
              replanBusy={busyAction === `re-plan ${tr.id}`}
              audioBusy={busyAction === `audio ${tr.id}`}
              mediaV={mediaV} />
          ))}
        </>
      )}
      {snap.storyboard_error && (
        <Text c="red" size="sm">Storyboard unreadable: {snap.storyboard_error}</Text>
      )}
    </Stack>
  );
}
