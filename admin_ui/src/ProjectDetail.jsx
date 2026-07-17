// One project's workspace: photos, storyboard review/edit, per-clip render,
// audio, finalize — every pipeline subcommand with its CLI options.
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { api, fileUrl } from './api.js';
import { Btn, C, Field, S, Toggle, stepChip } from './ui.jsx';

// A transition's start_frame/end_frame is the styled image path in practice
// ("styled_images/img1.png"), but may also be a frame id — handle both.
function frameName(framesById, frameRef) {
  if (frameRef && frameRef.includes('/')) return frameRef.split('/').pop();
  const f = framesById[frameRef];
  return f ? f.output_path.split('/').pop() : '';
}

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

function TransitionCard({ project, tr, framesById, clip, edited, placeholder, onEdit, onRegenerate, onRedoAudio, busy, audioBusy }) {
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

// ----------------------- per-command option panels ------------------------- #
// Each panel mirrors that subcommand's CLI flags; empty/indeterminate fields
// are simply not sent, so the pipeline/config defaults still apply.

function StoryboardOptions({ run, busy }) {
  const [idea, setIdea] = useState('');
  const [frameCount, setFrameCount] = useState('');
  const [stylePrompt, setStylePrompt] = useState('');
  const [duration, setDuration] = useState('');
  const [analyze, setAnalyze] = useState(true);
  const [force, setForce] = useState(false);
  const start = () => {
    const o = {};
    if (idea.trim()) o.idea = idea.trim();
    if (idea.trim() && frameCount !== '') o.frame_count = Number(frameCount);
    if (stylePrompt.trim()) o.style_prompt = stylePrompt.trim();
    if (duration) o.duration = Number(duration);
    if (!analyze) o.analyze_frames = false;
    if (force) o.force = true;
    run('storyboard', o);
  };
  return (
    <div>
      <Field label="Idea (leave empty to storyboard from the project's photos; fill to invent frames from text — like --idea)">
        <textarea style={{ ...S.input, minHeight: 48, resize: 'vertical' }} value={idea}
          placeholder="(from photos)" onChange={(e) => setIdea(e.target.value)} />
      </Field>
      <div style={{ display: 'flex', gap: 10, marginTop: 8, alignItems: 'flex-end', flexWrap: 'wrap' }}>
        {idea.trim() && (
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
          title="Off = --no-analyze: one generic motion prompt for every clip" />
        <Toggle label="Force re-style" checked={force} onChange={setForce}
          title="--force: re-style images that already exist (spends OpenAI credits)" />
        <span style={{ flex: 1 }} />
        <Btn busy={busy} onClick={start}>Run storyboard</Btn>
      </div>
    </div>
  );
}

function RenderOptions({ run, busy }) {
  const [motionPrompt, setMotionPrompt] = useState('');
  const [duration, setDuration] = useState('');
  const [dryRun, setDryRun] = useState(false);
  const start = () => {
    const o = {};
    if (motionPrompt.trim()) o.motion_prompt = motionPrompt.trim();
    if (duration) o.duration = Number(duration);
    if (dryRun) o.dry_run = true;
    run('render', o);
  };
  return (
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
        title="--dry-run: show what would be rendered without spending credits" />
      <Btn busy={busy} onClick={start}>Render missing clips</Btn>
    </div>
  );
}

function AudioOptions({ run, busy }) {
  const [musicPrompt, setMusicPrompt] = useState('');
  const start = () => {
    const o = {};
    if (musicPrompt.trim()) o.music_prompt = musicPrompt.trim();
    run('audio', o);
  };
  return (
    <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end', flexWrap: 'wrap' }}>
      <Field label="Music prompt override (optional)" style={{ flex: 1, minWidth: 240 }}>
        <input style={S.input} value={musicPrompt} placeholder="(config music_prompt)"
          onChange={(e) => setMusicPrompt(e.target.value)} />
      </Field>
      <Btn busy={busy} onClick={start}
        title="Per-clip SFX + music bed, then rebuild the final video">Run audio</Btn>
    </div>
  );
}

function CombineOptions({ run, busy }) {
  // null = follow config; the toggles below are explicitly sent either way
  // once touched, mirroring --intro/--no-intro etc.
  const [intro, setIntro] = useState(null);
  const [credits, setCredits] = useState(null);
  const [letter, setLetter] = useState(null);
  const [force, setForce] = useState(true);
  const tri = (value, set, label, title) => (
    <Toggle label={label + (value === null ? ' (config)' : value ? ' (on)' : ' (off)')}
      checked={value === true} indeterminate={value === null} title={title}
      onChange={() => set(value === null ? true : value === true ? false : null)} />
  );
  const start = () => {
    const o = {};
    if (intro !== null) o.intro_clip = intro;
    if (credits !== null) o.credits_photos = credits;
    if (letter !== null) o.closing_letter = letter;
    if (force) o.force = true;
    run('combine', o);
  };
  return (
    <div style={{ display: 'flex', gap: 16, alignItems: 'center', flexWrap: 'wrap' }}>
      {tri(intro, setIntro, 'Intro clip', 'Prepend the shared intro.mp4')}
      {tri(credits, setCredits, 'Credits photos', 'End-credits montage of the original photos')}
      {tri(letter, setLetter, 'Closing letter', "Scroll the project's letter.txt at the end")}
      <Toggle label="Rebuild existing final" checked={force} onChange={setForce} title="--force" />
      <span style={{ flex: 1 }} />
      <Btn busy={busy} onClick={start}>Combine</Btn>
    </div>
  );
}

function RunOptions({ run, busy }) {
  const [noCombine, setNoCombine] = useState(false);
  const start = () => {
    if (!window.confirm(
      'Run the whole flow (storyboard if missing → render ALL missing clips → '
      + 'final video) without further confirmation? This spends real credits.'
    )) return;
    const o = {};
    if (noCombine) o.no_combine = true;
    run('run', o);
  };
  return (
    <div style={{ display: 'flex', gap: 16, alignItems: 'center', flexWrap: 'wrap' }}>
      <span style={{ color: C.muted, fontSize: 13, flex: 1 }}>
        The whole flow in one job — the CLI's <code>run</code>. Confirmation gates
        auto-proceed on the server, so this renders every missing clip.
      </span>
      <Toggle label="Stop after clips (no final video)" checked={noCombine} onChange={setNoCombine} />
      <Btn busy={busy} style={{ background: C.err }} onClick={start}>Run everything</Btn>
    </div>
  );
}

// ------------------------------ main view ---------------------------------- #

// Sentinel id in the `dirty` set for the storyboard-level global motion
// prompt (transition ids never collide with it).
const GLOBAL_EDIT = '__global_motion__';

export default function ProjectDetail({ name, onBack, notify }) {
  const [snap, setSnap] = useState(null);
  const [storyboard, setStoryboard] = useState(null); // parsed, editable copy
  const [dirty, setDirty] = useState(new Set());
  const [busyAction, setBusyAction] = useState('');
  const [logJob, setLogJob] = useState(null);
  const [showPhotos, setShowPhotos] = useState(false);
  const [openPanel, setOpenPanel] = useState('');
  const [uploading, setUploading] = useState(false);
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

  const regenerate = async (clipId) => {
    if (dirty.size) { notify('Save your storyboard edits first.'); return; }
    await run('render', { clips: [clipId] }, `render ${clipId}`);
  };

  const redoAudio = async (clipId) => {
    if (dirty.size) { notify('Save your storyboard edits first.'); return; }
    await run('audio', { clips: [clipId] }, `audio ${clipId}`);
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

  const deletePhoto = async (filename) => {
    if (!window.confirm(`Delete input photo ${filename}? (Styled frames and clips are kept.)`)) return;
    try {
      await api.deletePhoto(name, filename);
      notify(`Deleted ${filename}`);
      await load();
    } catch (e) { notify(`Delete failed: ${e.message}`); }
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

  const panelBtn = (id, label, primary) => (
    <Btn ghost={!primary} busy={busyAction === id} disabled={Boolean(activeJob)}
      style={openPanel === id ? { borderColor: C.accent, color: C.accentSoft } : {}}
      onClick={() => setOpenPanel(openPanel === id ? '' : id)}>
      {label} {openPanel === id ? '▴' : '▾'}
    </Btn>
  );

  const panels = {
    storyboard: <StoryboardOptions run={run} busy={busyAction === 'storyboard'} />,
    render: <RenderOptions run={run} busy={busyAction === 'render'} />,
    audio: <AudioOptions run={run} busy={busyAction === 'audio'} />,
    combine: <CombineOptions run={run} busy={busyAction === 'combine'} />,
    run: <RunOptions run={run} busy={busyAction === 'run'} />
  };

  return (
    <div>
      <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 10, flexWrap: 'wrap' }}>
        <Btn ghost onClick={onBack}>← back</Btn>
        <h2 style={{ margin: 0, fontSize: 18 }}>{name}</h2>
        <span style={S.chip(stepColor)}>{stepText}</span>
        {snap.order?.customer && <span style={{ color: C.muted, fontSize: 13 }}>
          {snap.order.customer} · {snap.order.order_id}</span>}
        {activeJob && <span style={S.chip(activeJob.state === 'cancelling' ? C.accentSoft : C.run)}>
          {activeJob.command} {activeJob.state}…</span>}
      </div>

      <div style={{ ...S.card, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        {panelBtn('storyboard', 'Storyboard')}
        {panelBtn('render', 'Render', true)}
        {panelBtn('audio', 'Audio')}
        {panelBtn('combine', 'Combine')}
        <Btn ghost busy={busyAction === 'finalize'} disabled={Boolean(activeJob)}
          onClick={() => run('combine', { intro_clip: true, credits_photos: true, force: true }, 'finalize')}>
          Finalize (intro + credits)
        </Btn>
        {panelBtn('run', 'Run everything')}
        <span style={{ flex: 1 }} />
        <Btn ghost onClick={() => setShowPhotos((v) => !v)}>
          {showPhotos ? 'Hide photos' : `Photos (${(snap.input_images || []).length})`}
        </Btn>
      </div>
      {openPanel && <div style={S.card}>{panels[openPanel]}</div>}

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

      {showPhotos && (
        <div style={S.card}>
          <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 8 }}>
            <strong>Photos</strong>
            <span style={{ color: C.muted, fontSize: 12 }}>
              Movie order follows the filenames (sorted). Styled versions shown when available.
            </span>
            <span style={{ flex: 1 }} />
            <input ref={fileInputRef} type="file" accept="image/*" multiple style={{ display: 'none' }}
              onChange={(e) => upload(e.target.files)} />
            <Btn ghost busy={uploading} onClick={() => fileInputRef.current?.click()}>Add photos</Btn>
          </div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {(snap.input_images || []).map((img) => {
              const styledName = img.replace(/\.[^.]+$/, '.png');
              const styled = (snap.styled_images || []).includes(styledName);
              return (
                <div key={img} style={{ position: 'relative' }}>
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
            {(snap.input_images || []).length === 0 && (
              <p style={{ color: C.muted, fontSize: 13 }}>
                No photos yet — add some above, or run Storyboard with an idea instead.
              </p>
            )}
          </div>
        </div>
      )}

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
              onRegenerate={regenerate} onRedoAudio={redoAudio}
              busy={busyAction === `render ${tr.id}`} audioBusy={busyAction === `audio ${tr.id}`} />
          ))}
        </>
      )}
      {snap.storyboard_error && <p style={S.err}>Storyboard unreadable: {snap.storyboard_error}</p>}
    </div>
  );
}
