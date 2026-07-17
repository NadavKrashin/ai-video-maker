// Thin client for the pipeline admin API (`pipeline.py serve`).
//
// The panel is normally served BY that same server (admin_ui/dist mounted
// at /), so the API base defaults to same-origin and only the admin token
// needs entering. A custom base is still supported for the panel hosted
// elsewhere (static hosting + a tunnel to the pipeline machine).

const LS_BASE = 'avm_admin_api_base';
const LS_TOKEN = 'avm_admin_api_token';

export const getBase = () => (localStorage.getItem(LS_BASE) || '').replace(/\/+$/, '');
export const getToken = () => localStorage.getItem(LS_TOKEN) || '';
export const saveSettings = (base, token) => {
  localStorage.setItem(LS_BASE, (base || '').trim());
  localStorage.setItem(LS_TOKEN, (token || '').trim());
};
export const hasSettings = () => Boolean(getToken());

async function request(path, options = {}) {
  const res = await fetch(`${getBase()}${path}`, {
    ...options,
    headers: {
      Authorization: `Bearer ${getToken()}`,
      ...(options.body && !(options.body instanceof FormData)
        ? { 'Content-Type': 'application/json' } : {}),
      ...(options.headers || {})
    }
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try { detail = (await res.json()).detail || detail; } catch { /* not json */ }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }
  return res.json();
}

export const api = {
  health: () => request('/api/health'),
  orders: () => request('/api/orders'),
  ingestOrder: (order, storyboard) =>
    request('/api/orders/ingest', { method: 'POST', body: JSON.stringify({ order, storyboard }) }),
  pollWatcher: () => request('/api/watch/poll', { method: 'POST' }),
  projects: () => request('/api/projects'),
  createProject: (name) =>
    request('/api/projects', { method: 'POST', body: JSON.stringify({ name }) }),
  project: (name) => request(`/api/projects/${encodeURIComponent(name)}`),
  uploadPhotos: (name, files) => {
    const fd = new FormData();
    for (const f of files) fd.append('files', f, f.name);
    return request(`/api/projects/${encodeURIComponent(name)}/photos`, {
      method: 'POST', body: fd
    });
  },
  deletePhoto: (name, filename) =>
    request(`/api/projects/${encodeURIComponent(name)}/photos/${encodeURIComponent(filename)}`, {
      method: 'DELETE'
    }),
  saveStoryboard: (name, storyboard) =>
    request(`/api/projects/${encodeURIComponent(name)}/storyboard`, {
      method: 'PUT', body: JSON.stringify(storyboard)
    }),
  runAction: (name, command, options = {}) =>
    request(`/api/projects/${encodeURIComponent(name)}/actions/${command}`, {
      method: 'POST', body: JSON.stringify(options)
    }),
  jobs: (project) =>
    request(`/api/jobs${project ? `?project=${encodeURIComponent(project)}` : ''}`),
  job: (id) => request(`/api/jobs/${id}`),
  cancelJob: (id) => request(`/api/jobs/${id}/cancel`, { method: 'POST' })
};

// <img>/<video> tags can't send an Authorization header — the API accepts the
// token as a query param for exactly this case.
export const fileUrl = (project, kind, filename) =>
  `${getBase()}/api/projects/${encodeURIComponent(project)}/files/${kind}/` +
  `${encodeURIComponent(filename)}?token=${encodeURIComponent(getToken())}`;
