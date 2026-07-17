// Projects tab: every local project workspace + creating a new one from
// scratch (the UI twin of `pipeline.py init`).
import React, { useEffect, useState } from 'react';
import { api } from './api.js';
import { Btn, C, S, stepChip } from './ui.jsx';

export default function ProjectsTab({ onOpenProject, notify }) {
  const [projects, setProjects] = useState(null);
  const [newName, setNewName] = useState('');
  const [creating, setCreating] = useState(false);

  useEffect(() => {
    api.projects().then((d) => setProjects(d.projects))
      .catch((e) => notify(`Projects failed: ${e.message}`));
  }, [notify]);

  const create = async () => {
    const name = newName.trim();
    if (!name) return;
    setCreating(true);
    try {
      await api.createProject(name);
      notify(`Project "${name}" created — add photos, then run Storyboard.`);
      onOpenProject(name);
    } catch (e) { notify(`Create failed: ${e.message}`); }
    finally { setCreating(false); }
  };

  if (projects === null) return <p style={{ color: C.muted }}>Loading projects…</p>;
  return (
    <div>
      <div style={{ ...S.card, display: 'flex', gap: 10, alignItems: 'flex-end' }}>
        <div style={{ flex: 1 }}>
          <label style={S.label}>New project (letters, digits, - and _)</label>
          <input style={S.input} value={newName} placeholder="my-movie"
            onChange={(e) => setNewName(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && create()} />
        </div>
        <Btn busy={creating} disabled={!newName.trim()} onClick={create}>Create</Btn>
      </div>
      {projects.map((p) => {
        const [text, color] = p.error ? ['error', C.err] : stepChip(p.next_step);
        const rendered = (p.clips || []).filter((c) => c.rendered).length;
        return (
          <div key={p.project} style={{ ...S.card, display: 'flex', gap: 12, alignItems: 'center', cursor: 'pointer' }}
            onClick={() => onOpenProject(p.project)}>
            <div style={{ flex: 1 }}>
              <div style={{ fontWeight: 700 }}>{p.project}
                {p.order?.customer ? <span style={{ color: C.muted, fontWeight: 400 }}> · {p.order.customer}</span> : null}
              </div>
              <div style={{ color: C.muted, fontSize: 12 }}>
                {p.error || `${(p.input_images || []).length} photos · ${rendered}/${(p.clips || []).length} clips` +
                  (p.final_video ? ' · final ready' : '')}
              </div>
            </div>
            <span style={S.chip(color)}>{text}</span>
          </div>
        );
      })}
      {projects.length === 0 && <p style={{ color: C.muted }}>No projects yet — create one above or ingest an order.</p>}
    </div>
  );
}
