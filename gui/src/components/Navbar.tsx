import React from 'react';
import { Activity, Cloud, Cpu, HardDrive, Layers, MessageSquare, Settings } from 'lucide-react';
import { ConnectionState } from '../hooks';
import { DownloadJob } from '../downloads';
import { DownloadIndicator } from './DownloadsPanel';
import { TranslationKey, useI18n } from '../i18n';

export type NavTab = 'dashboard' | 'storage' | 'models' | 'catalog' | 'chat' | 'settings';

interface NavbarProps {
  activeTab: NavTab;
  setActiveTab: (tab: NavTab) => void;
  connection: ConnectionState;
  loadedModelName: string | null;
  downloads: DownloadJob[];
  onToggleDownloads: () => void;
}

const TABS: Array<{ id: NavTab; labelKey: TranslationKey; icon: typeof Cpu }> = [
  { id: 'dashboard', labelKey: 'nav.dashboard', icon: Cpu },
  { id: 'storage', labelKey: 'nav.storage', icon: HardDrive },
  { id: 'models', labelKey: 'nav.models', icon: Layers },
  { id: 'catalog', labelKey: 'nav.catalog', icon: Cloud },
  { id: 'chat', labelKey: 'nav.chat', icon: MessageSquare },
  { id: 'settings', labelKey: 'nav.settings', icon: Settings },
];

const CONNECTION_STYLE: Record<ConnectionState, { key: TranslationKey; color: string; background: string; border: string }> = {
  online: { key: 'conn.online', color: 'var(--accent-emerald)', background: 'rgba(16,185,129,0.12)', border: 'rgba(16,185,129,0.3)' },
  connecting: { key: 'conn.connecting', color: 'var(--accent-amber)', background: 'rgba(245,158,11,0.12)', border: 'rgba(245,158,11,0.3)' },
  offline: { key: 'conn.offline', color: 'var(--accent-red)', background: 'rgba(239,68,68,0.12)', border: 'rgba(239,68,68,0.3)' },
};

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  connection,
  loadedModelName,
  downloads,
  onToggleDownloads,
}) => {
  const { t } = useI18n();
  const status = CONNECTION_STYLE[connection];

  return (
    <header
      className="glass-card"
      style={{
        margin: '16px 24px',
        padding: '11px 20px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: '16px',
        borderRadius: 'var(--radius-xl)',
        flexWrap: 'wrap',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '13px' }}>
        <div
          style={{
            width: '40px',
            height: '40px',
            borderRadius: '12px',
            background: 'var(--gradient-brand)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 0 18px rgba(0,242,254,0.35)',
          }}
        >
          <Activity size={22} color="#fff" />
        </div>
        <div>
          <h1
            style={{
              fontSize: '18px',
              fontWeight: 800,
              letterSpacing: '0.5px',
              background: 'var(--gradient-brand)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            AILOFLOW
          </h1>
          <p style={{ fontSize: '10.5px', color: 'var(--text-muted)', fontWeight: 500 }}>
            {loadedModelName ? t('nav.running', { name: loadedModelName }) : t('nav.noModel')}
          </p>
        </div>
      </div>

      <nav
        style={{
          display: 'flex',
          gap: '6px',
          background: 'rgba(10,12,16,0.6)',
          padding: '4px',
          borderRadius: 'var(--radius-lg)',
          border: '1px solid var(--bg-card-border)',
        }}
      >
        {TABS.map(({ id, labelKey, icon: Icon }) => {
          const active = activeTab === id;
          return (
            <button
              key={id}
              onClick={() => setActiveTab(id)}
              className="btn-secondary"
              style={{
                padding: '8px 14px',
                fontSize: '13px',
                background: active ? 'rgba(0,242,254,0.15)' : 'transparent',
                borderColor: active ? 'var(--accent-cyan)' : 'transparent',
                color: active ? 'var(--accent-cyan)' : 'var(--text-secondary)',
              }}
            >
              <Icon size={15} /> {t(labelKey)}
            </button>
          );
        })}
      </nav>

      <div style={{ display: 'flex', alignItems: 'center', gap: '9px' }}>
        <DownloadIndicator jobs={downloads} onClick={onToggleDownloads} />

        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '6px 12px',
            borderRadius: 'var(--radius-md)',
            background: status.background,
            border: `1px solid ${status.border}`,
          }}
        >
          <span
            style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: status.color,
              boxShadow: connection === 'online' ? `0 0 8px ${status.color}` : 'none',
            }}
          />
          <span style={{ fontSize: '11.5px', fontWeight: 700, color: status.color }}>{t(status.key)}</span>
        </div>
      </div>
    </header>
  );
};
