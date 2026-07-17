// One project's workspace, laid out as the pipeline it is:
//   1 Storyboard → 2 Render → 3 Audio → 4 Combine
// Each step is a tile showing its status; clicking it opens that step's
// options + an explanation. Every action that changes files or spends money
// goes through a confirmation modal that says exactly what will happen.
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { api, fileUrl } from './api.js';
import { Btn, C, ConfirmModal, Field, S, Toggle, stepChip } from './ui.jsx';

// A transition's start_frame/end_frame is the styled image path in practice
// ("styled_images/img1.png"), but may also be a frame id — handle both.
function frameName(framesById, frameRef) {
  if (frameRef && frameRef.includes('/')) return frameRef.split('/').pop();
  const f = framesById[frameRef];
  return f ? f.output_path.split('/').pop() : '';
}

const CLIP_PRICE = '≈ $0.35 (5s) – $0.70 (10s) per clip';

function JobRow({ job, onShowLog, onCancel, cancelBusy }) {
  const color = job.state === 'done' ? C.ok
    : job.state === 'failed' ? C.err
      : job.state === 'cancelled' ? C.muted
        : job.state === 'cancelling' ? C.accentSoft
          : C.run;
  const cancellable = job.state === 'queued' || job.state === 'running';
  return (
    <div style={{ display: 'flex', gap: 10, alignItems: 'center', padding: '6px 0', borderBottom: `1px solid ${C.border}` }}>
      <span style={S.chip(color)}>{job.state}</span>
      <span style={{ fontWeight: 600 }}>{job.command}</span>
      <span style={{ color: C.muted, fontSize: 12, flex: 1 }}>
        {(job.started_at || job.created_at || '').replace('T', ' ').slice(0, 19)}
      </span>
      {job.error && <span style={{ ...S.err, maxWidth: 380, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{job.error}</span>}
      {(cancellable || job.state === 'cancelling') && (
        <Btn ghost busy={cancelBusy} disabled={!cancellable}
          style={{ color: C.err, borderColor: `${C.err}88` }}
          title="Queued jobs stop now; running jobs finish the current item first. Re-running later resumes."
          onClick={() => onCancel(job.id)}>
          {job.state === 'cancelling' ? 'cancelling…' : 'cancel'}
        </Btn>
      )}
      <Btn ghost onClick={() => onShowLog(job.id)}>log</Btn>
    </div>
  );
}

function TransitionCard({ project, tr, framesById, clip, edited, placeholder, onEdit, onRegenerate, onReplan, onRedoAudio, busy, replanBusy, audioBusy }) {
  const startImg = frameName(framesById, tr.start_frame);
  const endImg = frameName(framesById, tr.end_frame);
  const clipFile = tr.output_path.split('/').pop();
  return (
    <div style={S.card}>
      <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 8, flexWrap: 'wrap' }}>
        <strong>{tr.id}</strong>
        <span style={S.chip(clip?.rendered ? C.ok : C.accentSoft)}>
          {clip?.rendered ? (clip.sfx ? 'rendered · sfx' : 'rendered · silent') : 'not rendered'}
        </span>
        {clip?.stale && (
          <span style={S.chip('#E0A75B')}
            title="The storyboard changed after this clip was rendered (re-planned motion). The clip is kept as-is — click regenerate when you want the new plan applied (spends credits).">
            outdated
          </span>
        )}
        <span style={{ color: C.muted, fontSize: 12 }}>{tr.duration}s</span>
        {edited && <span style={S.chip(C.accentSoft)}>edited</span>}
        {placeholder && !edited && (
          <span style={S.chip(C.err)} title="Planning failed for this pair; it still has the generic fallback prompt. Re-running Storyboard re-plans it.">
            generic prompt
          </span>
        )}
      </div>
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          {startImg && <img src={fileUrl(project, 'styled', startImg)} alt={tr.start_frame}
            style={{ width: 128, borderRadius: 6 }} />}
          <span style={{ color: C.muted }}>→</span>
          {endImg && <img src={fileUrl(project, 'styled', endImg)} alt={tr.end_frame}
            style={{ width: 128, borderRadius: 6 }} />}
        </div>
        {clip?.rendered && (
          <video controls preload="metadata" style={{ width: 260, borderRadius: 6, background: '#000' }}
            src={fileUrl(project, 'clips', clipFile)} />
        )}
      </div>
      <label style={{ ...S.label, marginTop: 10 }}>Motion prompt</label>
      <textarea style={{ ...S.input, minHeight: 60, resize: 'vertical' }} value={tr.motion_prompt}
        onChange={(e) => onEdit({ ...tr, motion_prompt: e.target.value })} />
      <div style={{ display: 'flex', gap: 10, marginTop: 8, alignItems: 'flex-end', flexWrap: 'wrap' }}>
        <Field label="Duration">
          <select style={{ ...S.input, width: 90 }} value={tr.duration}
            onChange={(e) => onEdit({ ...tr, duration: Number(e.target.value) })}>
            <option value={5}>5s</option>
            <option value={10}>10s</option>
          </select>
        </Field>
        <Field label="Sound prompt (optional)" style={{ flex: 1, minWidth: 220 }}>
          <input style={S.input} value={tr.sound_prompt || ''}
            onChange={(e) => onEdit({ ...tr, sound_prompt: e.target.value })} />
        </Field>
        <Btn ghost busy={replanBusy} onClick={() => onReplan(tr.id)}
          title="Ask the AI planner to write a fresh motion prompt for this pair (small OpenAI call; the clip is not re-rendered)">
          Re-plan prompt
        </Btn>
        {clip?.rendered && (
          <Btn ghost busy={audioBusy} onClick={() => onRedoAudio(tr.id)}
            title="Redo just this clip's SFX (use after editing its sound prompt)">
            Redo audio
          </Btn>
        )}
        <Btn ghost busy={busy} onClick={() => onRegenerate(tr.id)}
          title="Re-render just this clip (save edits first)">
          {clip?.rendered ? 'Regenerate clip' : 'Render this clip'}
        </Btn>
      </div>
    </div>
  );
}

// ----------------------- per-step option panels ----------------------------- #
// Each panel = a plain-words explanation of the step, its CLI options, and a
// primary button that opens the confirmation modal (nothing runs directly).

function PanelIntro({ children }) {
  return <p style={{ color: C.muted, fontSize: 13, margin: '0 0 12px', lineHeight: 1.5 }}>{children}</p>;
}

function StoryboardPanel({ ask, locked, info }) {
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
      <Field label="Idea (leave empty to storyboard from the project's photos; fill to invent frames from text)">
        <textarea style={{ ...S.input, minHeight: 48, resize: 'vertical' }} value={idea}
          placeholder="(from photos)" onChange={(e) => setIdea(e.target.value)} />
      </Field>
      <div style={{ display: 'flex', gap: 10, marginTop: 8, alignItems: 'flex-end', flexWrap: 'wrap' }}>
        {fromIdea && (
          <Field label="Frames (0 = model decides)">
            <input style={{ ...S.input, width: 110 }} type="number" min="0" value={frameCount}
              placeholder="config" onChange={(e) => setFrameCount(e.target.value)} />
          </Field>
        )}
        <Field label="Force every clip to" style={{ width: 130 }}>
          <select style={S.input} value={duration} onChange={(e) => setDuration(e.target.value)}>
            <option value="">planner mixes</option>
            <option value="5">5s</option>
            <option value="10">10s</option>
          </select>
        </Field>
        <Field label="Style prompt override (optional)" style={{ flex: 1, minWidth: 220 }}>
          <input style={S.input} value={stylePrompt} placeholder="(config style_prompt)"
            onChange={(e) => setStylePrompt(e.target.value)} />
        </Field>
      </div>
      <div style={{ display: 'flex', gap: 16, marginTop: 10, alignItems: 'center', flexWrap: 'wrap' }}>
        <Toggle label="Analyze frames (per-clip motion plans)" checked={analyze} onChange={setAnalyze}
          title="Off: skip the vision analysis and give every clip the one generic motion prompt" />
        <Toggle label="Force re-style all photos" checked={force} onChange={setForce}
          title="Re-style photos that are already styled (spends OpenAI credits on every photo again)" />
        <span style={{ flex: 1 }} />
        <Btn disabled={locked} onClick={start}>Run storyboard…</Btn>
      </div>
    </div>
  );
}

function RenderPanel({ ask, locked, info }) {
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
      <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end', flexWrap: 'wrap' }}>
        <Field label="Motion prompt override for this run (optional)" style={{ flex: 1, minWidth: 240 }}>
          <input style={S.input} value={motionPrompt} placeholder="(per-clip storyboard prompts)"
            onChange={(e) => setMotionPrompt(e.target.value)} />
        </Field>
        <Field label="Force duration" style={{ width: 120 }}>
          <select style={S.input} value={duration} onChange={(e) => setDuration(e.target.value)}>
            <option value="">storyboard</option>
            <option value="5">5s</option>
            <option value="10">10s</option>
          </select>
        </Field>
        <Toggle label="Dry run" checked={dryRun} onChange={setDryRun}
          title="Show what would be rendered without spending credits" />
        <Btn disabled={locked} onClick={start}>
          {`Render ${info.missing || 'missing'} clip(s)…`}
        </Btn>
      </div>
    </div>
  );
}

function AudioPanel({ ask, locked, info }) {
  const [musicPrompt, setMusicPrompt] = useState('');
  const start = () => {
    const o = {};
    if (musicPrompt.trim()) o.music_prompt = musicPrompt.trim();
    ask({
      title: 'Run audio?',
      lines: [
        info.silentRendered > 0
          ? `Adds synced sound effects to ${info.silentRendered} silent clip(s) (clips that already have SFX are skipped).`
          : 'All rendered clips already have SFX — they are skipped.',
        'Generates the background music track if it doesn\'t exist yet, then rebuilds the final video with everything mixed.',
        'Audio jobs are much cheaper than clip renders.'
      ],
      cost: 'fal', label: 'Run audio', command: 'audio', options: o
    });
  };
  return (
    <div>
      <PanelIntro>
        Two layers of sound: per-clip effects synced to the action, and one
        background music track mixed over the whole movie. Optional — skip it
        for a silent film.
      </PanelIntro>
      <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end', flexWrap: 'wrap' }}>
        <Field label="Music prompt override (optional)" style={{ flex: 1, minWidth: 240 }}>
          <input style={S.input} value={musicPrompt} placeholder="(config music_prompt)"
            onChange={(e) => setMusicPrompt(e.target.value)} />
        </Field>
        <Btn disabled={locked} onClick={start}>Run audio…</Btn>
      </div>
    </div>
  );
}

function CombinePanel({ ask, locked, info }) {
  // null = follow config; cycling the toggle: config → on → off → config.
  const [intro, setIntro] = useState(null);
  const [credits, setCredits] = useState(null);
  const [letter, setLetter] = useState(null);
  const tri = (value, set, label, title) => (
    <Toggle label={label + (value === null ? ' (config)' : value ? ' (on)' : ' (off)')}
      checked={value === true} indeterminate={value === null} title={title}
      onChange={() => set(value === null ? true : value === true ? false : null)} />
  );
  const start = (finalize) => {
    const o = finalize
      ? { intro_clip: true, credits_photos: true, force: true }
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
        1920×1080 movie, with the music bed when audio is on. Free and
        repeatable — rerun it any time a clip changes.
      </PanelIntro>
      <div style={{ display: 'flex', gap: 16, alignItems: 'center', flexWrap: 'wrap' }}>
        {tri(intro, setIntro, 'Intro clip', 'Prepend the shared intro.mp4')}
        {tri(credits, setCredits, 'Credits photos', 'End-credits montage of the original photos')}
        {tri(letter, setLetter, 'Closing letter', "Scroll the project's letter.txt at the end")}
        <span style={{ flex: 1 }} />
        <Btn ghost disabled={locked} onClick={() => start(true)}
          title="The delivery preset: intro + photo credits + rebuild">Finalize…</Btn>
        <Btn disabled={locked} onClick={() => start(false)}>Combine…</Btn>
      </div>
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
      <div style={{ display: 'flex', gap: 16, alignItems: 'center', flexWrap: 'wrap' }}>
        <Toggle label="Stop after clips (no final video)" checked={noCombine} onChange={setNoCombine} />
        <span style={{ flex: 1 }} />
        <Btn disabled={locked} style={{ background: C.err }} onClick={start}>Run everything…</Btn>
      </div>
    </div>
  );
}

// ------------------------------ main view ---------------------------------- #

// Sentinel id in the `dirty` set for the storyboard-level global motion
// prompt (transition ids never collide with it).
const GLOBAL_EDIT = '__global_motion__';

const STEPS = [
  { id: 'storyboard', n: 1, name: 'Storyboard', caption: 'Style photos & plan each clip' },
  { id: 'render', n: 2, name: 'Render', caption: 'Generate the video clips' },
  { id: 'audio', n: 3, name: 'Audio', caption: 'Sound effects + music' },
  { id: 'combine', n: 4, name: 'Combine', caption: 'Build the final movie' }
];

export default function ProjectDetail({ name, onBack, notify }) {
  const [snap, setSnap] = useState(null);
  const [storyboard, setStoryboard] = useState(null); // parsed, editable copy
  const [dirty, setDirty] = useState(new Set());
  const [busyAction, setBusyAction] = useState('');
  const [logJob, setLogJob] = useState(null);
  const [showPhotos, setShowPhotos] = useState(false);
  const [openPanel, setOpenPanel] = useState('');
  const [uploading, setUploading] = useState(false);
  const [confirm, setConfirm] = useState(null);
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
          return fresh;
        });
      } catch { setStoryboard(null); }
    } else setStoryboard(null);
    return data;
  }, [name]);

  const dirtyRef = useRef(dirty);
  dirtyRef.current = dirty;

  useEffect(() => { load().catch((e) => notify(`Load failed: ${e.message}`)); }, [load, notify]);

  // Poll while a job is queued/running, refresh once when it settles.
  useEffect(() => {
    const active = (snap?.jobs || []).some((j) => ['queued', 'running', 'cancelling'].includes(j.state));
    if (!active) return undefined;
    pollRef.current = setInterval(() => load().catch(() => {}), 3000);
    return () => clearInterval(pollRef.current);
  }, [snap, load]);

  if (!snap) return <p style={{ color: C.muted }}>Loading {name}…</p>;

  const framesById = Object.fromEntries((storyboard?.frames || []).map((f) => [f.id, f]));
  const clipsById = Object.fromEntries((snap.clips || []).map((c) => [c.id, c]));
  // Transitions the backend planner never succeeded on (still carrying the
  // config fallback prompt) — flagged so nobody renders a whole order generic.
  const placeholderIds = new Set(snap.storyboard?.placeholder_transitions || []);
  const [stepText, stepColor] = stepChip(snap.next_step);
  const activeJob = (snap.jobs || []).find((j) => ['running', 'queued', 'cancelling'].includes(j.state));
  const locked = Boolean(activeJob);

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
    silentRendered: (snap.clips || []).filter((c) => c.rendered && !c.sfx).length,
    finalExists: Boolean(snap.final_video)
  };

  const stepStatus = (id) => {
    const done = {
      storyboard: Boolean(storyboard),
      render: total > 0 && rendered === total,
      audio: rendered > 0 && info.silentRendered === 0,
      combine: info.finalExists
    }[id];
    if (done) return 'done';
    if (snap.next_step === id) return 'next';
    return id === 'audio' ? 'optional' : 'todo';
  };

  const run = async (command, options = {}, label = command) => {
    setBusyAction(label);
    try {
      await api.runAction(name, command, options);
      notify(`${label} started`);
      setOpenPanel('');
      await load();
    } catch (e) { notify(`${label} failed: ${e.message}`); }
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

  const saveEdits = async () => {
    setBusyAction('save');
    try {
      await api.saveStoryboard(name, storyboard);
      setDirty(new Set());
      notify('Storyboard saved');
      await load();
    } catch (e) { notify(`Save failed: ${e.message}`); }
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

  const needsSave = () => {
    if (dirty.size) { notify('Save your storyboard edits first.'); return true; }
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

  const upload = async (files) => {
    if (!files?.length) return;
    setUploading(true);
    try {
      const res = await api.uploadPhotos(name, Array.from(files));
      notify(`Uploaded ${res.saved.length} photo(s)`);
      setShowPhotos(true);
      await load();
    } catch (e) { notify(`Upload failed: ${e.message}`); }
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
        } catch (e) { notify(`Delete failed: ${e.message}`); }
      }
    });
  };

  const showLog = async (jobId) => {
    try { setLogJob(await api.job(jobId)); }
    catch (e) { notify(`Log failed: ${e.message}`); }
  };

  const cancelJob = async (jobId) => {
    setBusyAction(`cancel ${jobId}`);
    try {
      const res = await api.cancelJob(jobId);
      notify(res.job.state === 'cancelling'
        ? 'Cancelling — the clip being generated will finish, then the job stops.'
        : 'Job cancelled.');
      await load();
    } catch (e) { notify(`Cancel failed: ${e.message}`); }
    finally { setBusyAction(''); }
  };

  const statusChipFor = (status) =>
    status === 'done' ? <span style={S.chip(C.ok)}>✓ done</span>
      : status === 'next' ? <span style={S.chip(C.accent)}>next step</span>
        : status === 'optional' ? <span style={{ color: C.muted, fontSize: 11 }}>optional</span>
          : null;

  const panels = {
    storyboard: <StoryboardPanel ask={ask} locked={locked} info={info} />,
    render: <RenderPanel ask={ask} locked={locked} info={info} />,
    audio: <AudioPanel ask={ask} locked={locked} info={info} />,
    combine: <CombinePanel ask={ask} locked={locked} info={info} />,
    runall: <RunAllPanel ask={ask} locked={locked} info={info} />
  };

  return (
    <div>
      <ConfirmModal confirm={confirm} onCancel={() => setConfirm(null)} onConfirm={confirmed} />

      <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 10, flexWrap: 'wrap' }}>
        <Btn ghost onClick={onBack}>← back</Btn>
        <h2 style={{ margin: 0, fontSize: 18 }}>{name}</h2>
        <span style={S.chip(stepColor)}>{stepText}</span>
        {snap.order?.customer && <span style={{ color: C.muted, fontSize: 13 }}>
          {snap.order.customer} · {snap.order.order_id}</span>}
        {activeJob && <span style={S.chip(activeJob.state === 'cancelling' ? C.accentSoft : C.run)}>
          {activeJob.command} {activeJob.state}…</span>}
      </div>

      <div style={{ ...S.card, padding: 10 }}>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {STEPS.map((step) => {
            const status = stepStatus(step.id);
            const selected = openPanel === step.id;
            return (
              <div key={step.id} role="button" tabIndex={0}
                onClick={() => setOpenPanel(selected ? '' : step.id)}
                onKeyDown={(e) => e.key === 'Enter' && setOpenPanel(selected ? '' : step.id)}
                style={{
                  flex: '1 1 150px', minWidth: 150, cursor: 'pointer',
                  background: C.panel, borderRadius: 8, padding: '10px 12px',
                  border: `1px solid ${selected ? C.accent : status === 'next' ? `${C.accent}88` : C.border}`
                }}>
                <div style={{ display: 'flex', gap: 8, alignItems: 'center', justifyContent: 'space-between' }}>
                  <strong style={{ fontSize: 14 }}>
                    <span style={{ color: C.muted, fontWeight: 400 }}>{step.n} · </span>{step.name}
                  </strong>
                  {statusChipFor(status)}
                </div>
                <div style={{ color: C.muted, fontSize: 12, marginTop: 3 }}>{step.caption}</div>
              </div>
            );
          })}
          <div role="button" tabIndex={0}
            onClick={() => setOpenPanel(openPanel === 'runall' ? '' : 'runall')}
            onKeyDown={(e) => e.key === 'Enter' && setOpenPanel(openPanel === 'runall' ? '' : 'runall')}
            title="All remaining steps as one unattended job"
            style={{
              flex: '0 1 150px', minWidth: 130, cursor: 'pointer',
              background: 'transparent', borderRadius: 8, padding: '10px 12px',
              border: `1px dashed ${openPanel === 'runall' ? C.err : C.border}`
            }}>
            <strong style={{ fontSize: 14, color: C.muted }}>⚡ Run everything</strong>
            <div style={{ color: C.muted, fontSize: 12, marginTop: 3 }}>All steps, no stops</div>
          </div>
        </div>
        {openPanel && (
          <div style={{ borderTop: `1px solid ${C.border}`, marginTop: 10, paddingTop: 12 }}>
            {panels[openPanel]}
          </div>
        )}
      </div>

      {snap.final_video && (
        <div style={S.card}>
          <strong>Final video</strong>
          <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginTop: 8, flexWrap: 'wrap' }}>
            <video controls preload="metadata" style={{ width: 420, maxWidth: '100%', borderRadius: 8, background: '#000' }}
              src={fileUrl(name, 'output', 'final_video.mp4')} />
            <a style={{ color: C.accentSoft }} href={fileUrl(name, 'output', 'final_video.mp4')}
              download={`${name}.mp4`}>Download</a>
          </div>
        </div>
      )}

      <div style={S.card}>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
          <strong>Photos ({inputImages.length})</strong>
          <span style={{ color: C.muted, fontSize: 12, flex: 1 }}>
            Movie order follows the filenames (sorted). Styled versions shown when available.
          </span>
          <input ref={fileInputRef} type="file" accept="image/*" multiple style={{ display: 'none' }}
            onChange={(e) => upload(e.target.files)} />
          <Btn ghost busy={uploading} onClick={() => fileInputRef.current?.click()}>Add photos</Btn>
          <Btn ghost onClick={() => setShowPhotos((v) => !v)}>{showPhotos ? 'Hide' : 'Show'}</Btn>
        </div>
        {showPhotos && (
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 10 }}>
            {inputImages.map((img) => {
              const styledName = img.replace(/\.[^.]+$/, '.png');
              const styled = styledImages.includes(styledName);
              return (
                <div key={img}>
                  <img style={{ width: 120, borderRadius: 6 }} alt={img}
                    src={fileUrl(name, styled ? 'styled' : 'input', styled ? styledName : img)} />
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ color: C.muted, fontSize: 11 }}>{img}</span>
                    <button title={`Delete ${img}`} onClick={() => deletePhoto(img)}
                      style={{ background: 'none', border: 'none', color: C.err, cursor: 'pointer', fontSize: 13 }}>
                      ✕
                    </button>
                  </div>
                </div>
              );
            })}
            {inputImages.length === 0 && (
              <p style={{ color: C.muted, fontSize: 13 }}>
                No photos yet — add some above, or run Storyboard with an idea instead.
              </p>
            )}
          </div>
        )}
      </div>

      {(snap.jobs || []).length > 0 && (
        <div style={S.card}>
          <strong>Jobs</strong>
          {(snap.jobs || []).map((j) => (
            <JobRow key={j.id} job={j} onShowLog={showLog} onCancel={cancelJob}
              cancelBusy={busyAction === `cancel ${j.id}`} />
          ))}
        </div>
      )}

      {logJob && (
        <div style={S.card}>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <strong>{logJob.command} — {logJob.state}</strong>
            <Btn ghost onClick={() => setLogJob(null)}>close</Btn>
          </div>
          <pre style={{ fontSize: 11, color: C.muted, overflowX: 'auto', maxHeight: 300 }}>
            {(logJob.log || []).join('\n') || '(no log lines)'}
          </pre>
        </div>
      )}

      {storyboard && (
        <>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, margin: '18px 0 8px' }}>
            <h3 style={{ margin: 0, fontSize: 16 }}>
              Storyboard — {storyboard.transitions.length} clips
            </h3>
            {dirty.size > 0 && (
              <Btn busy={busyAction === 'save'} onClick={saveEdits}>
                Save {dirty.size} edit{dirty.size > 1 ? 's' : ''}
              </Btn>
            )}
          </div>
          {placeholderIds.size > 0 && (
            <div style={{ ...S.card, border: `1px solid ${C.err}88` }}>
              <strong style={{ color: C.err }}>
                {placeholderIds.size} of {storyboard.transitions.length} transitions still have
                the generic fallback prompt
              </strong>
              <p style={{ color: C.muted, fontSize: 13, margin: '6px 0 0' }}>
                Planning failed for them (OpenAI quota or rate limit) — run Storyboard again
                to re-plan exactly these. Clips rendered from the generic prompt lose their
                tailored motion.
              </p>
            </div>
          )}
          <div style={S.card}>
            <label style={S.label}>
              Global motion prompt — prepended to every clip (whole-movie facts, e.g.
              “two separate people appear throughout; never blend them”; keep it to a
              sentence or two)
            </label>
            <textarea style={{ ...S.input, minHeight: 44, resize: 'vertical' }}
              placeholder="(none)" value={storyboard.global_motion_prompt || ''}
              onChange={(e) => editGlobalMotion(e.target.value)} />
          </div>
          {storyboard.transitions.map((tr) => (
            <TransitionCard key={tr.id} project={name} tr={tr} framesById={framesById}
              clip={clipsById[tr.output_path.split('/').pop()?.replace(/\.mp4$/, '')]}
              edited={dirty.has(tr.id)} placeholder={placeholderIds.has(tr.id)}
              onEdit={editTransition}
              onRegenerate={regenerate} onReplan={replanPrompt} onRedoAudio={redoAudio}
              busy={busyAction === `render ${tr.id}`}
              replanBusy={busyAction === `re-plan ${tr.id}`}
              audioBusy={busyAction === `audio ${tr.id}`} />
          ))}
        </>
      )}
      {snap.storyboard_error && <p style={S.err}>Storyboard unreadable: {snap.storyboard_error}</p>}
    </div>
  );
}
