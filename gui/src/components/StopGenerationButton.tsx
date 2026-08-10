import React, { useState } from 'react';
import { Check, Loader2, Square } from 'lucide-react';
import { api } from '../api';
import { useI18n } from '../i18n';

interface Props {
  /** True while the runtime reports a generation in flight. */
  active: boolean;
  /** Hidden entirely when no model is loaded — there is nothing to stop. */
  modelLoaded: boolean;
}

/**
 * Frees the GPU when a generation is stuck.
 *
 * Deliberately clickable even when nothing looks active: a client that abandons
 * a stream without closing its socket leaves the engine running while the
 * runtime's own "generating" flag has already been cleared, and that is exactly
 * the situation where the user needs this button. Pressing it with nothing
 * running is harmless and says so.
 */
export const StopGenerationButton: React.FC<Props> = ({ active, modelLoaded }) => {
  const { t } = useI18n();
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<'stopped' | 'idle' | null>(null);

  if (!modelLoaded) return null;

  const stop = async () => {
    setBusy(true);
    setResult(null);
    try {
      const { stopped } = await api.stopGeneration();
      setResult(stopped ? 'stopped' : 'idle');
    } catch {
      setResult('idle');
    } finally {
      setBusy(false);
      setTimeout(() => setResult(null), 2600);
    }
  };

  const label =
    result === 'stopped' ? t('stop.stopped')
    : result === 'idle' ? t('stop.nothing')
    : t('stop.button');

  const accent = active ? 'var(--accent-red)' : 'var(--text-secondary)';

  return (
    <button
      onClick={stop}
      disabled={busy}
      title={t('stop.hint')}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '7px',
        padding: '6px 12px',
        borderRadius: 'var(--radius-md)',
        background: active ? 'rgba(239,68,68,0.14)' : 'rgba(255,255,255,0.05)',
        border: `1px solid ${active ? 'rgba(239,68,68,0.4)' : 'var(--bg-card-border)'}`,
        color: result === 'stopped' ? 'var(--accent-emerald)' : accent,
        cursor: busy ? 'wait' : 'pointer',
        fontFamily: 'var(--font-main)',
        fontSize: '11.5px',
        fontWeight: 700,
      }}
    >
      {busy ? (
        <Loader2 size={14} className="spin" />
      ) : result === 'stopped' ? (
        <Check size={14} />
      ) : (
        <Square size={12} fill={active ? 'currentColor' : 'none'} />
      )}
      {label}
    </button>
  );
};
