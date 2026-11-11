// AI Video Maker admin panel — a browser UI over the pipeline's admin API
// (`pipeline.py serve`), which also serves this app's build at /.
import React, { useEffect, useState } from 'react';
import {
  Button, Card, Container, Group, PasswordInput, SegmentedControl, Text, Title
} from '@mantine/core';
import { api, getToken, hasSettings, saveSettings } from './api.js';
import OrdersTab from './OrdersTab.jsx';
import ProjectsTab from './ProjectsTab.jsx';
import ProjectDetail from './ProjectDetail.jsx';
import SpendingTab from './SpendingTab.jsx';
import LearningTab from './LearningTab.jsx';
import { LightboxProvider } from './ui.jsx';

function Settings({ onConnected }) {
  const [token, setToken] = useState(getToken());
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  const connect = async () => {
    setBusy(true); setError('');
    saveSettings(token);
    try {
      await api.health();
      await api.projects(); // exercises auth, not just reachability
      onConnected();
    } catch (e) {
      setError(`Could not connect: ${e.message}`);
    } finally { setBusy(false); }
  };

  return (
    <Card withBorder maw={480} mx="auto" mt="15vh" padding="xl">
      <Title order={3} mb="xs">Connect to the pipeline</Title>
      <Text size="sm" c="dimmed" mb="md">
        The <code>ADMIN_API_TOKEN</code> value from the pipeline's{' '}
        <code>.env</code> — it authenticates every request from this panel.
      </Text>
      <PasswordInput label="Admin token" placeholder="ADMIN_API_TOKEN from .env"
        value={token} onChange={(e) => setToken(e.target.value)}
        onKeyDown={(e) => e.key === 'Enter' && connect()}
        error={error || undefined} />
      <Group justify="flex-end" mt="lg">
        <Button loading={busy} onClick={connect}>Connect</Button>
      </Group>
    </Card>
  );
}

const TABS = ['orders', 'projects', 'spending', 'learning'];

// '#/project/<name>' -> that project; '#/<tab>' -> that tab; anything else
// (including a hash from an old bookmark) -> orders.
function readRoute() {
  const hash = (window.location.hash || '').replace(/^#\/?/, '');
  const [head, ...rest] = hash.split('/');
  if (head === 'project' && rest.length) {
    let name = rest.join('/');
    try { name = decodeURIComponent(name); } catch { /* keep it raw */ }
    return { tab: 'project', project: name };
  }
  return { tab: TABS.includes(head) ? head : 'orders', project: '' };
}

export default function App() {
  const [connected, setConnected] = useState(hasSettings());
  // The tab and the open project live in the URL hash (#/projects,
  // #/project/<name>), so a refresh keeps your place, the browser's back
  // button works, and a project can be bookmarked or sent as a link. The
  // hash is never sent to the server, so nothing changes for the API.
  const [route, setRoute] = useState(readRoute);
  useEffect(() => {
    const onHash = () => setRoute(readRoute());
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);
  const { tab, project } = route;
  const go = (nextTab, nextProject = '') => {
    const hash = nextTab === 'project'
      ? `#/project/${encodeURIComponent(nextProject)}` : `#/${nextTab}`;
    if (window.location.hash !== hash) window.location.hash = hash;
    setRoute({ tab: nextTab, project: nextProject });
  };

  if (!connected) {
    return <Settings onConnected={() => setConnected(true)} />;
  }

  const openProject = (name) => go('project', name);

  return (
    <LightboxProvider>
      <Container size="lg" py="lg">
        <Group mb="xl" gap="lg">
          <Title order={2} c="orange.4" style={{ letterSpacing: '-0.5px' }}>
            AI Video Maker
          </Title>
          <SegmentedControl
            value={tab === 'project' ? 'projects' : tab}
            onChange={(value) => go(value)}
            data={[
              { label: 'Orders', value: 'orders' },
              { label: 'Projects', value: 'projects' },
              { label: 'Spending', value: 'spending' },
              { label: 'Learning', value: 'learning' }
            ]} />
          <div style={{ flex: 1 }} />
          <Button variant="default" size="xs" onClick={() => setConnected(false)}>
            Settings
          </Button>
        </Group>
        {tab === 'orders' && <OrdersTab onOpenProject={openProject} />}
        {tab === 'projects' && <ProjectsTab onOpenProject={openProject} />}
        {tab === 'spending' && <SpendingTab onOpenProject={openProject} />}
        {tab === 'learning' && <LearningTab onOpenProject={openProject} />}
        {tab === 'project' && project && (
          <ProjectDetail key={project} name={project} onBack={() => go('projects')} />
        )}
      </Container>
    </LightboxProvider>
  );
}
