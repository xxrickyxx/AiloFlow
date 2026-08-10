import React from 'react';
import { CheckCircle2, Download, Trash2, X, XCircle } from 'lucide-react';
import { DownloadJob, cancelDownload, clearFinishedDownloads } from '../downloads';
import { useI18n } from '../i18n';
import { formatBytes, formatDuration } from '../format';

interface Props {
  jobs: DownloadJob[];
  onChanged: () => void;
  /** Compact mode renders a single summary line for the navbar. */
  compact?: boolean;
}

const STATUS_COLOR: Record<DownloadJob['status'], string> = {
  running: 'var(--accent-cyan)',
  retrying: 'var(--accent-amber)',
  completed: 'var(--accent-emerald)',
  failed: 'var(--accent-red)',
  cancelled: 'var(--text-muted)',
};

export const DownloadsPanel: React.FC<Props> = ({ jobs, onChanged, compact }) => {
  const { t } = useI18n();

  if (jobs.length === 0) {
    return compact ? null : (
      <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{t('dl.none')}</p>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '9px' }}>
      {jobs.map((job) => {
        const color = STATUS_COLOR[job.status];
        const statusLabel =
          job.status === 'running' ? t('dl.running')
          : job.status === 'retrying' ? t('dl.retrying', { attempt: job.attempts })
          : job.status === 'completed' ? t('dl.completed')
          : job.status === 'failed' ? t('dl.failed')
          : t('dl.cancelled');

        return (
          <div
            key={job.id}
            style={{
              padding: compact ? '9px 11px' : '13px 15px',
              background: 'rgba(255,255,255,0.04)',
              border: `1px solid ${job.status === 'failed' ? 'rgba(239,68,68,0.35)' : 'var(--bg-card-border)'}`,
              borderRadius: 'var(--radius-md)',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '10px' }}>
              <div style={{ minWidth: 0 }}>
                <div
                  style={{
                    fontSize: compact ? '12px' : '13px',
                    fontWeight: 700,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}
                  title={`${job.repoId} — ${job.fileName}`}
                >
                  {job.fileName}
                </div>
                <div style={{ fontSize: '10.5px', color: 'var(--text-muted)', marginTop: '2px' }}>
                  {job.repoId}
                </div>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '7px', flexShrink: 0 }}>
                <span style={{ fontSize: '11px', fontWeight: 700, color, fontFamily: 'var(--font-mono)' }}>
                  {job.percent.toFixed(1)}%
                </span>
                {job.status === 'running' || job.status === 'retrying' ? (
                  <button
                    className="btn-secondary"
                    style={{ padding: '4px 7px' }}
                    title={t('common.cancel')}
                    onClick={async () => {
                      await cancelDownload(job.id);
                      onChanged();
                    }}
                  >
                    <X size={13} />
                  </button>
                ) : job.status === 'completed' ? (
                  <CheckCircle2 size={15} color="var(--accent-emerald)" />
                ) : (
                  <XCircle size={15} color={color} />
                )}
              </div>
            </div>

            <div style={{ marginTop: '8px', height: '6px', background: 'rgba(255,255,255,0.06)', borderRadius: '3px', overflow: 'hidden' }}>
              <div
                style={{
                  width: `${Math.min(100, job.percent)}%`,
                  height: '100%',
                  background: job.status === 'running' ? 'var(--gradient-brand)' : color,
                  transition: 'width 0.4s ease-out',
                }}
              />
            </div>

            <div style={{ fontSize: '10.5px', color: 'var(--text-muted)', marginTop: '6px' }}>
              {statusLabel} · {formatBytes(job.receivedBytes)} / {formatBytes(job.totalBytes)}
              {job.totalParts > 1 ? ` · ${t('dl.part', { index: job.partIndex, total: job.totalParts })}` : ''}
              {job.status === 'running' && job.bytesPerSecond > 0 ? ` · ${formatBytes(job.bytesPerSecond)}/s` : ''}
              {job.status === 'running' && job.etaSeconds !== null
                ? ` · ${t('dl.remaining', { time: formatDuration(job.etaSeconds * 1000) })}`
                : ''}
            </div>

            {job.error && (
              <div style={{ fontSize: '11px', color: 'var(--accent-red)', marginTop: '5px' }}>{job.error}</div>
            )}
            {job.status === 'completed' && !compact && (
              <div style={{ fontSize: '11px', color: 'var(--accent-emerald)', marginTop: '5px' }}>
                {t('dl.completedIn')}
              </div>
            )}
          </div>
        );
      })}

      {!compact && jobs.some((j) => j.status !== 'running') && (
        <button
          className="btn-secondary"
          style={{ alignSelf: 'flex-start', padding: '6px 11px', fontSize: '12px' }}
          onClick={async () => {
            await clearFinishedDownloads();
            onChanged();
          }}
        >
          <Trash2 size={13} /> {t('dl.clearFinished')}
        </button>
      )}
    </div>
  );
};

/** Small always-visible indicator for the navbar. */
export const DownloadIndicator: React.FC<{ jobs: DownloadJob[]; onClick: () => void }> = ({ jobs, onClick }) => {
  const { t } = useI18n();
  const active = jobs.filter((j) => j.status === 'running');
  if (active.length === 0) return null;

  const overall =
    active.reduce((sum, j) => sum + j.percent, 0) / active.length;

  return (
    <button
      onClick={onClick}
      title={t('dl.keepsRunning')}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        padding: '6px 12px',
        borderRadius: 'var(--radius-md)',
        background: 'rgba(0,242,254,0.12)',
        border: '1px solid rgba(0,242,254,0.3)',
        color: 'var(--accent-cyan)',
        cursor: 'pointer',
        fontFamily: 'var(--font-main)',
      }}
    >
      <Download size={14} className="pulse" />
      <span style={{ fontSize: '11.5px', fontWeight: 700 }}>
        {active.length > 1 ? `${active.length} × ` : ''}
        {overall.toFixed(1)}%
      </span>
    </button>
  );
};
