import React from 'react';
import { AlertCircle, Loader2 } from 'lucide-react';
import { NOT_AVAILABLE, hasValue } from '../format';
import { useI18n } from '../i18n';

/** A gauge bar that refuses to draw a fill for an unmeasured value. */
export const MeterBar: React.FC<{ percent: number | null; color: string; height?: number }> = ({
  percent,
  color,
  height = 10,
}) => (
  <div
    style={{
      width: '100%',
      height,
      background: 'rgba(255,255,255,0.06)',
      borderRadius: height / 2,
      overflow: 'hidden',
      position: 'relative',
    }}
  >
    {hasValue(percent) ? (
      <div
        style={{
          width: `${Math.max(0, Math.min(100, percent))}%`,
          height: '100%',
          background: color,
          borderRadius: height / 2,
          boxShadow: `0 0 12px ${color}`,
          transition: 'width 0.4s ease-out',
        }}
      />
    ) : (
      <div
        style={{
          position: 'absolute',
          inset: 0,
          backgroundImage:
            'repeating-linear-gradient(45deg, rgba(255,255,255,0.07) 0 6px, transparent 6px 12px)',
        }}
      />
    )}
  </div>
);

export const StatCard: React.FC<{
  label: string;
  value: string;
  hint?: string;
  accent?: string;
  unavailable?: boolean;
}> = ({ label, value, hint, accent = 'var(--accent-cyan)', unavailable }) => (
  <div className="glass-card" style={{ padding: '18px 20px' }}>
    <span style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text-muted)', letterSpacing: '0.4px' }}>
      {label}
    </span>
    <div
      style={{
        fontSize: '28px',
        fontWeight: 800,
        marginTop: '4px',
        color: unavailable ? 'var(--text-muted)' : accent,
        fontFamily: 'var(--font-mono)',
      }}
    >
      {value}
    </div>
    {hint && (
      <span style={{ fontSize: '11px', color: 'var(--text-muted)', display: 'block', marginTop: '4px' }}>
        {hint}
      </span>
    )}
  </div>
);

export const EmptyState: React.FC<{ title: string; description: string; action?: React.ReactNode }> = ({
  title,
  description,
  action,
}) => (
  <div
    className="glass-card"
    style={{ padding: '32px', textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '10px' }}
  >
    <h3 style={{ fontSize: '16px', fontWeight: 700, color: 'var(--text-secondary)' }}>{title}</h3>
    <p style={{ fontSize: '13px', color: 'var(--text-muted)', maxWidth: '620px', lineHeight: 1.6 }}>{description}</p>
    {action}
  </div>
);

export const ErrorBox: React.FC<{ message: string }> = ({ message }) => {
  const { t } = useI18n();
  return (
    <div
      className="glass-card"
      style={{
        padding: '16px 20px',
        borderColor: 'rgba(239,68,68,0.4)',
        background: 'rgba(239,68,68,0.08)',
        display: 'flex',
        gap: '12px',
        alignItems: 'flex-start',
      }}
    >
      <AlertCircle size={20} color="var(--accent-red)" style={{ flexShrink: 0, marginTop: '2px' }} />
      <div>
        <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--accent-red)' }}>{t('common.error')}</div>
        <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginTop: '2px', whiteSpace: 'pre-wrap' }}>
          {message}
        </p>
      </div>
    </div>
  );
};

export const Spinner: React.FC<{ label?: string }> = ({ label }) => {
  const { t } = useI18n();
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: 'var(--text-muted)', fontSize: '13px', padding: '14px' }}>
      <Loader2 size={16} className="spin" />
      {label || t('common.loading')}
    </div>
  );
};

/** Badge marking whether a figure was measured or is a class-based estimate. */
export const MeasuredBadge: React.FC<{ measured: boolean }> = ({ measured }) => {
  const { t } = useI18n();
  return (
    <span
      className="badge-tag"
      style={
        measured
          ? { color: 'var(--accent-emerald)', borderColor: 'rgba(16,185,129,0.35)', background: 'rgba(16,185,129,0.12)' }
          : { color: 'var(--accent-amber)', borderColor: 'rgba(245,158,11,0.35)', background: 'rgba(245,158,11,0.12)' }
      }
    >
      {measured ? t('common.measured') : t('common.estimate')}
    </span>
  );
};

export const NotAvailable: React.FC = () => (
  <span style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{NOT_AVAILABLE}</span>
);
