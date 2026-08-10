import React, { useState } from 'react';
import { CheckCircle2, Cpu, Download, RefreshCw } from 'lucide-react';
import { api, EngineCandidate, streamPost } from '../api';
import { useAsync } from '../hooks';
import { ErrorBox, Spinner } from './Common';
import { useI18n } from '../i18n';
import { formatBytes } from '../format';

interface Props {
  onInstalled: () => void;
}

/**
 * Installs AILOFlow's own inference engine.
 *
 * The point of this panel is that the runtime is self-sufficient: pick the
 * build that matches the detected GPU and AILOFlow fetches, unpacks and
 * registers it, with no external daemon involved.
 */
export const EngineSetup: React.FC<Props> = ({ onInstalled }) => {
  const { t } = useI18n();
  const engine = useAsync(() => api.engine(), []);
  const [candidates, setCandidates] = useState<{ release: string; candidates: EngineCandidate[] } | null>(null);
  const [loadingCandidates, setLoadingCandidates] = useState(false);
  const [installing, setInstalling] = useState<string | null>(null);
  const [progress, setProgress] = useState<{ phase: string; percent: number; bytesPerSecond: number } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadCandidates = async () => {
    setLoadingCandidates(true);
    setError(null);
    try {
      setCandidates(await api.engineCandidates());
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoadingCandidates(false);
    }
  };

  const install = async (candidate: EngineCandidate) => {
    if (!candidates) return;
    setInstalling(candidate.variant);
    setProgress(null);
    setError(null);

    try {
      await streamPost('/engine/install', { variant: candidate.variant, release: candidates.release }, {
        progress: (p) => setProgress({ phase: p.phase, percent: p.percent, bytesPerSecond: p.bytesPerSecond }),
        done: () => {
          engine.reload();
          onInstalled();
        },
        error: (d) => setError(d.message),
      });
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setInstalling(null);
      setProgress(null);
    }
  };

  const active = engine.data?.active;

  return (
    <div className="glass-card" style={{ padding: '22px' }}>
      <h3 style={{ fontSize: '15px', fontWeight: 800, marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}>
        <Cpu size={17} color="var(--accent-emerald)" /> {t('settings.engineTitle')}
      </h3>

      {error && <ErrorBox message={error} />}
      {engine.loading && <Spinner />}

      {active ? (
        <div
          style={{
            display: 'flex',
            gap: '11px',
            alignItems: 'flex-start',
            padding: '12px 14px',
            borderRadius: 'var(--radius-md)',
            background: 'rgba(16,185,129,0.08)',
            border: '1px solid rgba(16,185,129,0.3)',
          }}
        >
          <CheckCircle2 size={18} color="var(--accent-emerald)" style={{ flexShrink: 0, marginTop: '1px' }} />
          <div style={{ minWidth: 0 }}>
            <div style={{ fontSize: '13px', fontWeight: 700 }}>
              llama.cpp {active.release} — build {active.variant.toUpperCase()}
            </div>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '3px', wordBreak: 'break-all' }}>
              {active.serverPath}
            </div>
            {active.version && (
              <div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginTop: '2px' }}>{active.version}</div>
            )}
          </div>
        </div>
      ) : (
        !engine.loading && (
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.7 }}>
            {t('settings.engineNone')}
          </p>
        )
      )}

      <div style={{ marginTop: '14px', display: 'flex', gap: '9px', flexWrap: 'wrap' }}>
        <button className="btn-secondary" onClick={loadCandidates} disabled={loadingCandidates}>
          {loadingCandidates ? <RefreshCw size={15} className="spin" /> : <Download size={15} />}
          {active ? t('settings.checkUpdates') : t('settings.findBuilds')}
        </button>
      </div>

      {candidates && (
        <div style={{ marginTop: '14px' }}>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '8px' }}>
            {t('settings.release', { release: candidates.release })}
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {candidates.candidates.map((c) => (
              <div
                key={c.variant}
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  gap: '12px',
                  padding: '11px 13px',
                  background: 'rgba(255,255,255,0.04)',
                  borderRadius: 'var(--radius-md)',
                }}
              >
                <div style={{ minWidth: 0 }}>
                  <div style={{ fontSize: '13px', fontWeight: 700 }}>
                    {c.variant.toUpperCase()}
                    <span style={{ fontWeight: 400, color: 'var(--text-muted)', fontSize: '11px', marginLeft: '8px' }}>
                      {formatBytes(c.sizeBytes)}
                      {c.companionAssets.length > 0 ? ` + ${c.companionAssets.length} runtime` : ''}
                    </span>
                  </div>
                  <div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginTop: '2px' }}>{c.rationale}</div>
                </div>

                <button
                  className="btn-primary"
                  style={{ padding: '7px 13px', fontSize: '12px', flexShrink: 0 }}
                  onClick={() => install(c)}
                  disabled={installing !== null}
                >
                  {installing === c.variant ? <RefreshCw size={14} className="spin" /> : <Download size={14} />}
                  {t('settings.install')}
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {progress && (
        <div style={{ marginTop: '14px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: 'var(--text-muted)', marginBottom: '5px' }}>
            <span>
              {progress.phase === 'download' ? t('settings.downloadPhase') : progress.phase === 'extract' ? t('settings.extractPhase') : t('settings.verifyPhase')}
            </span>
            <span>
              {progress.percent}%{progress.bytesPerSecond ? ` · ${formatBytes(progress.bytesPerSecond)}/s` : ''}
            </span>
          </div>
          <div style={{ height: '7px', background: 'rgba(255,255,255,0.06)', borderRadius: '4px', overflow: 'hidden' }}>
            <div style={{ width: `${progress.percent}%`, height: '100%', background: 'var(--gradient-brand)', transition: 'width 0.25s' }} />
          </div>
        </div>
      )}
    </div>
  );
};
