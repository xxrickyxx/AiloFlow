import React from 'react';
import { AlertTriangle, CheckCircle, HelpCircle, Moon } from 'lucide-react';
import { TelemetrySnapshot } from '../api';
import { formatBandwidth } from '../format';

interface Props {
  bottleneck: TelemetrySnapshot['bottleneck'];
}

type Tone = 'warn' | 'ok' | 'neutral';

const TONE_BY_TYPE: Record<string, Tone> = {
  STORAGE: 'warn',
  VRAM: 'warn',
  RAM: 'warn',
  CPU: 'warn',
  GPU_COMPUTE: 'warn',
  CACHE: 'warn',
  NONE: 'ok',
  IDLE: 'neutral',
  UNKNOWN: 'neutral',
};

const PALETTE: Record<Tone, { color: string; border: string; background: string; Icon: typeof AlertTriangle }> = {
  warn: { color: 'var(--accent-amber)', border: 'rgba(245,158,11,0.4)', background: 'rgba(245,158,11,0.08)', Icon: AlertTriangle },
  ok: { color: 'var(--accent-emerald)', border: 'rgba(16,185,129,0.3)', background: 'rgba(16,185,129,0.06)', Icon: CheckCircle },
  neutral: { color: 'var(--text-secondary)', border: 'var(--bg-card-border)', background: 'rgba(255,255,255,0.03)', Icon: HelpCircle },
};

export const BottleneckBanner: React.FC<Props> = ({ bottleneck }) => {
  const tone = TONE_BY_TYPE[bottleneck.type] ?? 'neutral';
  const palette = PALETTE[tone];
  const Icon = bottleneck.type === 'IDLE' ? Moon : palette.Icon;

  return (
    <div
      className="glass-card"
      style={{
        margin: '0 24px 20px 24px',
        padding: '15px 20px',
        borderColor: palette.border,
        background: palette.background,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: '16px',
        flexWrap: 'wrap',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '13px', minWidth: 0 }}>
        <div
          style={{
            width: '38px',
            height: '38px',
            borderRadius: '10px',
            background: 'rgba(255,255,255,0.06)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: palette.color,
            flexShrink: 0,
          }}
        >
          <Icon size={20} />
        </div>
        <div style={{ minWidth: 0 }}>
          <h3 style={{ fontSize: '14px', fontWeight: 800, color: palette.color }}>{bottleneck.title}</h3>
          <p style={{ fontSize: '12.5px', color: 'var(--text-secondary)', marginTop: '2px', lineHeight: 1.5 }}>
            {bottleneck.description}
          </p>
        </div>
      </div>

      {bottleneck.requestedBandwidthMBps !== undefined && bottleneck.availableBandwidthMBps !== undefined && (
        <div style={{ display: 'flex', gap: '18px', borderLeft: '1px solid var(--bg-card-border)', paddingLeft: '18px' }}>
          <div>
            <span style={{ fontSize: '10px', color: 'var(--text-muted)', fontWeight: 700 }}>RICHIESTA</span>
            <div style={{ fontSize: '14px', fontWeight: 800, color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)' }}>
              {formatBandwidth(bottleneck.requestedBandwidthMBps)}
            </div>
          </div>
          <div>
            <span style={{ fontSize: '10px', color: 'var(--text-muted)', fontWeight: 700 }}>MISURATA</span>
            <div style={{ fontSize: '14px', fontWeight: 800, color: palette.color, fontFamily: 'var(--font-mono)' }}>
              {formatBandwidth(bottleneck.availableBandwidthMBps)}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
