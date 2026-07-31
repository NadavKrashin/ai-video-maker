// Shared pieces: toasts, status colors, cost badges, and the confirmation
// modal every mutating action goes through.
import React from 'react';
import { Badge, Button, Group, List, Modal } from '@mantine/core';
import { notifications } from '@mantine/notifications';

export const notify = (message, color = 'gray') =>
  notifications.show({ message, color });

// next_step -> chip for a whole project/order card.
export const stepChip = (next) =>
  next === 'storyboard' ? { label: 'needs storyboard', color: 'yellow' }
    : next === 'render' ? { label: 'needs render', color: 'blue' }
      : next === 'combine' ? { label: 'needs combine', color: 'blue' }
        : next === 'publish' ? { label: 'needs publish', color: 'grape' }
          : { label: 'complete', color: 'green' };

// Money, always rendered the same way. Every figure in the panel is an
// ESTIMATE priced from config.pricing (see costs.py) — never a provider
// invoice — so anywhere one is shown prominently must say so in words.
export const usd = (amount) => {
  const value = Number(amount || 0);
  if (!value) return '$0';
  // Sub-cent figures (a single planning call) would all render as "$0.00"
  // and look like nothing was spent.
  return value < 0.01 ? `$${value.toFixed(3)}` : `$${value.toFixed(2)}`;
};

// What an action costs, shown in every confirmation dialog: money honesty
// is the whole point of the modal.
export const COST = {
  free: { text: 'Free — no API credits', color: 'green' },
  openai: { text: 'Spends OpenAI credits', color: 'yellow' },
  fal: { text: 'Spends fal.ai credits', color: 'yellow' },
  both: { text: 'Spends OpenAI + fal.ai credits', color: 'red' }
};

// The verification step before anything that changes files or spends money:
// says exactly what will happen and what it costs, then asks.
export function ConfirmModal({ confirm, onConfirm, onCancel }) {
  const cost = COST[confirm?.cost || 'free'];
  return (
    <Modal opened={Boolean(confirm)} onClose={onCancel} centered
      title={confirm?.title} size="lg">
      {confirm && (
        <>
          <List spacing="xs" size="sm" mb="md">
            {confirm.lines.map((line, i) => <List.Item key={i}>{line}</List.Item>)}
          </List>
          <Badge variant="light" color={cost.color} mb="lg">{cost.text}</Badge>
          <Group justify="flex-end">
            <Button variant="default" onClick={onCancel}>Cancel</Button>
            <Button color={confirm.danger ? 'red' : 'orange'} onClick={onConfirm}>
              {confirm.label || 'Confirm'}
            </Button>
          </Group>
        </>
      )}
    </Modal>
  );
}
