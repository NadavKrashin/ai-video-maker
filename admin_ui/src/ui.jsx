// Shared pieces: toasts, status colors, cost badges, the image lightbox, and
// the confirmation modal every mutating action goes through.
import React, { useEffect, useState } from 'react';
import {
  Anchor, Badge, Button, Card, Group, Image, List, Modal, Stack,
  Text, TextInput, UnstyledButton
} from '@mantine/core';
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
  // `typeToConfirm` is for the handful of actions that destroy something no
  // amount of money can bring back. Reading a list and clicking a red button
  // is the right amount of friction for spending credits; it is not enough
  // for deleting a workspace full of them, so the name has to be typed.
  const [typed, setTyped] = useState('');
  const needsTyping = Boolean(confirm?.typeToConfirm);
  useEffect(() => { setTyped(''); }, [confirm?.typeToConfirm, confirm?.title]);
  const ready = !needsTyping || typed.trim() === confirm.typeToConfirm;
  return (
    <Modal opened={Boolean(confirm)} onClose={onCancel} centered
      title={confirm?.title} size="lg">
      {confirm && (
        <>
          <List spacing="xs" size="sm" mb="md">
            {confirm.lines.map((line, i) => <List.Item key={i}>{line}</List.Item>)}
          </List>
          <Badge variant="light" color={cost.color} mb="lg">{cost.text}</Badge>
          {needsTyping && (
            <TextInput mb="md" value={typed} data-autofocus
              onChange={(e) => setTyped(e.currentTarget.value)}
              label={`Type ${confirm.typeToConfirm} to confirm`}
              description="This cannot be undone, so it has to be typed out."
              placeholder={confirm.typeToConfirm} />
          )}
          <Group justify="flex-end">
            <Button variant="default" onClick={onCancel}>Cancel</Button>
            <Button color={confirm.danger ? 'red' : 'orange'} disabled={!ready}
              onClick={onConfirm}>
              {confirm.label || 'Confirm'}
            </Button>
          </Group>
        </>
      )}
    </Modal>
  );
}

// ---------------------------------------------------------------------------
// Lightbox: click any thumbnail in the panel to see it at full size.
//
// The panel's thumbnails are deliberately small (120px in the Photos step,
// 128px on a clip card, 64px for a cast reference) so a 30-photo order fits on
// one screen — but that is far too small for the judgement those pictures are
// actually there to support: whether a face survived styling, whether a head
// got cropped, whether the reference portrait is a clean crop of the right
// person. Rather than grow the grid, clicking opens the picture big.
//
// It takes a LIST of images, not one, because at every site the useful unit is
// a comparison: original vs styled in the Photos step (the card's own comment
// says telling a likeness slip from the source needs both in one glance), and
// start vs end frame on a clip card. Blowing up one half of a pair would give
// you the size and take away the comparison.

const LightboxContext = React.createContext(null);

export function LightboxProvider({ children }) {
  const [shot, setShot] = useState(null);
  return (
    <LightboxContext.Provider value={setShot}>
      {children}
      <Modal opened={Boolean(shot)} onClose={() => setShot(null)} centered
        size="auto" title={shot?.title} padding="md">
        {/* nowrap, and each image capped to its share of the viewport: a pair
            that wraps puts the second picture below the fold, which is the
            one thing this must not do — you would be scrolling between two
            images instead of comparing them. */}
        <Group align="flex-start" gap="md" wrap="nowrap" justify="center">
          {(shot?.images || []).map((img, i) => (
            <Stack key={i} gap={4} align="center" style={{ minWidth: 0 }}>
              <img src={img.src} alt={img.label || shot.title}
                style={{
                  display: 'block', objectFit: 'contain', borderRadius: 6,
                  maxHeight: '76vh',
                  maxWidth: shot.images.length > 1 ? '42vw' : '86vw'
                }} />
              {img.label && (
                <Text size="xs" c="dimmed">
                  {img.label}
                  {' · '}
                  <Anchor href={img.src} target="_blank" rel="noreferrer" size="xs">
                    open full size
                  </Anchor>
                </Text>
              )}
            </Stack>
          ))}
        </Group>
      </Modal>
    </LightboxContext.Provider>
  );
}

// Returns `zoom({ title, images: [{ src, label }] })`. Safe to call outside a
// provider — it becomes a no-op rather than throwing, so a component rendered
// in isolation (or a future tab that forgets the provider) still works.
export const useLightbox = () => React.useContext(LightboxContext) || (() => {});

// Drop-in for Mantine's <Image> that opens the lightbox when clicked.
// `zoomImages` overrides what gets shown — that is how a thumbnail opens its
// pair rather than only itself.
export function ZoomableImage({ zoomTitle, zoomImages, style, ...props }) {
  const zoom = useLightbox();
  return (
    <Image {...props} style={{ cursor: 'zoom-in', ...(style || {}) }}
      title={props.title || 'Click to see it bigger'}
      onClick={(event) => {
        // Several of these sit inside cards that have their own click
        // handlers (the face tagger places a tag wherever you click).
        event.stopPropagation();
        zoom({
          title: zoomTitle || props.alt,
          images: zoomImages || [{ src: props.src, label: props.alt }]
        });
      }} />
  );
}

// ---------------------------------------------------------------------------
// A card whose body folds away, remembering the choice per browser.
//
// The Cast and Cast faces sections grow with the movie — a real order came in
// with a 69-entry cast, which is a screen and a half of text inputs sitting
// between you and everything below them. Folding is the fix.
//
// Two rules it has to keep. Collapsing must never hide the fact that
// something needs attention, so anything the open card would warn about is
// summarised as a `badges` chip on the header, which stays visible either
// way. And the body stays MOUNTED when folded (hidden, not conditionally
// rendered) so a half-typed epithet is still there when you open it again.

const _sectionKey = (id) => `avm_section_open_${id}`;

const _readOpen = (id, fallback) => {
  // Storage throws outright in some contexts (private windows, thumbnailers),
  // so every read is guarded and the default is what a first visit sees.
  try {
    const raw = localStorage.getItem(_sectionKey(id));
    return raw === null ? fallback : raw === '1';
  } catch { return fallback; }
};

export function CollapsibleCard({
  id, title, badges, actions, children, defaultOpen = true
}) {
  const [open, setOpen] = useState(() => _readOpen(id, defaultOpen));
  const toggle = () => {
    setOpen((was) => {
      try { localStorage.setItem(_sectionKey(id), was ? '0' : '1'); }
      catch { /* a forgotten preference is not worth an error */ }
      return !was;
    });
  };
  return (
    <Card withBorder padding="md">
      <Group gap="sm" wrap="nowrap" mb={open ? 4 : 0}>
        <UnstyledButton onClick={toggle} style={{ flex: 1, minWidth: 0 }}
          title={open ? 'Collapse this section' : 'Expand this section'}>
          <Group gap={6} wrap="nowrap">
            <Text size="sm" c="dimmed" style={{ width: 10 }}>
              {open ? '▾' : '▸'}
            </Text>
            <Text size="sm" fw={500}>{title}</Text>
            {badges}
          </Group>
        </UnstyledButton>
        {actions}
      </Group>
      {/* display:none rather than Mantine's <Collapse>: Collapse renders
          closed on first mount even with in={true}, and an animation is not
          worth debugging for a fold. Hiding this way keeps the subtree
          MOUNTED, so a half-typed epithet survives a fold and unfold. */}
      <div style={{ display: open ? undefined : 'none' }}>{children}</div>
    </Card>
  );
}
