// AI Video Maker admin panel — a browser UI over the pipeline's admin API
// (`pipeline.py serve`), which also serves this app's build at /.
import React, { useState } from 'react';
import {
  Button, Card, Container, Group, PasswordInput, SegmentedControl, Text, Title
} from '@mantine/core';
import { api, getToken, hasSettings, saveSettings } from './api.js';
import OrdersTab from './OrdersTab.jsx';
import ProjectsTab from './ProjectsTab.jsx';
import ProjectDetail from './ProjectDetail.jsx';

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

export default function App() {
  const [connected, setConnected] = useState(hasSettings());
  const [tab, setTab] = useState('orders');
  const [project, setProject] = useState('');

  if (!connected) {
    return <Settings onConnected={() => setConnected(true)} />;
  }

  const openProject = (name) => { setProject(name); setTab('project'); };

  return (
    <Container size="lg" py="lg">
      <Group mb="xl" gap="lg">
        <Title order={2} c="orange.4" style={{ letterSpacing: '-0.5px' }}>
          AI Video Maker
        </Title>
        <SegmentedControl
          value={tab === 'project' ? 'projects' : tab}
          onChange={(value) => { setTab(value); setProject(''); }}
          data={[
            { label: 'Orders', value: 'orders' },
            { label: 'Projects', value: 'projects' }
          ]} />
        <div style={{ flex: 1 }} />
        <Button variant="default" size="xs" onClick={() => setConnected(false)}>
          Settings
        </Button>
      </Group>
      {tab === 'orders' && <OrdersTab onOpenProject={openProject} />}
      {tab === 'projects' && <ProjectsTab onOpenProject={openProject} />}
      {tab === 'project' && project && (
        <ProjectDetail name={project} onBack={() => setTab('projects')} />
      )}
    </Container>
  );
}
