import React, { useEffect, useState } from 'react';
import { CheckCircle2, FolderPlus, Globe, Save, Server, Trash2, XCircle } from 'lucide-react';
import { api, AiloConfig } from '../api';
import { useAsync } from '../hooks';
import { ErrorBox, Spinner } from './Common';
import { EngineSetup } from './EngineSetup';
import { LANGUAGE_NAMES, Language, useI18n } from '../i18n';

export const SettingsView: React.FC = () => {
  const { t, language, setLanguage } = useI18n();
  const config = useAsync(() => api.config(), []);
  const backends = useAsync(() => api.backends(), []);
  const system = useAsync(() => api.system(), []);

  const [draft, setDraft] = useState<AiloConfig | null>(null);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [newDirectory, setNewDirectory] = useState('');

  useEffect(() => {
    if (config.data) setDraft(config.data);
  }, [config.data]);

  const save = async (patch: Partial<AiloConfig>) => {
    setSaving(true);
    setError(null);
    try {
      const next = await api.updateConfig(patch);
      setDraft(next);
      setSaved(true);
      setTimeout(() => setSaved(false), 2500);
      backends.reload();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSaving(false);
    }
  };

  if (config.loading || !draft) return <div style={{ padding: '0 24px' }}><Spinner /></div>;

  const gpuBackends = system.data?.profile.gpus[0]?.supportedBackends || ['CPU'];

  return (
    <div style={{ padding: '0 24px', display: 'flex', flexDirection: 'column', gap: '16px', maxWidth: '900px' }}>
      {error && <ErrorBox message={error} />}
      {config.error && <ErrorBox message={config.error} />}

      {/* Language first: everything else is easier to read once it is set. */}
      <div className="glass-card" style={{ padding: '18px 22px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '16px', flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '11px' }}>
          <Globe size={18} color="var(--accent-blue)" />
          <span style={{ fontSize: '13px', fontWeight: 700 }}>{t('settings.language')}</span>
        </div>
        <div style={{ display: 'flex', gap: '7px' }}>
          {(Object.keys(LANGUAGE_NAMES) as Language[]).map((code) => {
            const active = language === code;
            return (
              <button
                key={code}
                onClick={() => setLanguage(code)}
                style={{
                  background: active ? 'rgba(0,242,254,0.15)' : 'rgba(255,255,255,0.05)',
                  border: `1px solid ${active ? 'var(--accent-cyan)' : 'var(--bg-card-border)'}`,
                  color: active ? 'var(--accent-cyan)' : 'var(--text-secondary)',
                  borderRadius: 'var(--radius-md)',
                  padding: '7px 15px',
                  fontSize: '12.5px',
                  fontWeight: 600,
                  cursor: 'pointer',
                  fontFamily: 'var(--font-main)',
                }}
              >
                {LANGUAGE_NAMES[code]}
              </button>
            );
          })}
        </div>
      </div>

      <EngineSetup onInstalled={() => { backends.reload(); config.reload(); }} />

      {/* Engines */}
      <div className="glass-card" style={{ padding: '22px' }}>
        <h3 style={{ fontSize: '15px', fontWeight: 800, marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Server size={17} color="var(--accent-cyan)" /> {t('settings.engines')}
        </h3>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {backends.data?.backends.map((b) => (
            <div
              key={b.id}
              style={{
                display: 'flex',
                gap: '11px',
                alignItems: 'flex-start',
                padding: '12px 14px',
                borderRadius: 'var(--radius-md)',
                background: 'rgba(255,255,255,0.04)',
                border: `1px solid ${b.available ? 'rgba(16,185,129,0.3)' : 'rgba(245,158,11,0.3)'}`,
              }}
            >
              {b.available ? (
                <CheckCircle2 size={18} color="var(--accent-emerald)" style={{ flexShrink: 0, marginTop: '1px' }} />
              ) : (
                <XCircle size={18} color="var(--accent-amber)" style={{ flexShrink: 0, marginTop: '1px' }} />
              )}
              <div>
                <div style={{ fontSize: '13px', fontWeight: 700 }}>{b.name}</div>
                <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '2px', wordBreak: 'break-all' }}>
                  {b.available ? b.detail : b.reason}
                </p>
              </div>
            </div>
          ))}
        </div>

        <div style={{ marginTop: '16px' }}>
          <label style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text-secondary)' }}>
            {t('settings.llamaPath')}
          </label>
          <div style={{ display: 'flex', gap: '8px', marginTop: '5px' }}>
            <input
              className="glass-input"
              style={{ flex: 1, fontSize: '12px' }}
              placeholder={t('settings.llamaPathPlaceholder')}
              value={draft.llamaServerPath || ''}
              onChange={(e) => setDraft({ ...draft, llamaServerPath: e.target.value || null })}
            />
            <button className="btn-primary" onClick={() => save({ llamaServerPath: draft.llamaServerPath })} disabled={saving}>
              <Save size={15} /> {t('settings.save')}
            </button>
          </div>
          <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '5px' }}>
            Serve per eseguire direttamente file .gguf. Senza binario, i modelli GGUF locali risultano
            non eseguibili invece di sembrare disponibili.
          </p>
        </div>

        <div style={{ marginTop: '16px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
          <div>
            <label style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text-secondary)' }}>{t('settings.ollamaUrl')}</label>
            <input
              className="glass-input"
              style={{ width: '100%', marginTop: '5px', fontSize: '12px' }}
              value={draft.ollamaBaseUrl}
              onChange={(e) => setDraft({ ...draft, ollamaBaseUrl: e.target.value })}
              onBlur={() => save({ ollamaBaseUrl: draft.ollamaBaseUrl })}
            />
          </div>
          <div>
            <label style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text-secondary)' }}>{t('settings.gpuBackend')}</label>
            <select
              className="glass-input"
              style={{ width: '100%', marginTop: '5px', fontSize: '12px' }}
              value={draft.backendOverride || ''}
              onChange={(e) => save({ backendOverride: e.target.value || null })}
            >
              <option value="">{t('settings.automatic', { backend: system.data?.profile.selectedBackend || t('common.notAvailable') })}</option>
              {gpuBackends.map((b) => (
                <option key={b} value={b}>{b}</option>
              ))}
            </select>
          </div>
        </div>

        <label style={{ display: 'flex', alignItems: 'center', gap: '9px', marginTop: '14px', fontSize: '13px', cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={draft.ollamaEnabled}
            onChange={(e) => save({ ollamaEnabled: e.target.checked })}
            style={{ accentColor: 'var(--accent-cyan)' }}
          />
          {t('settings.useOllama')}
        </label>
      </div>

      {/* Model directories */}
      <div className="glass-card" style={{ padding: '22px' }}>
        <h3 style={{ fontSize: '15px', fontWeight: 800, marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <FolderPlus size={17} color="var(--accent-blue)" /> {t('settings.modelFolders')}
        </h3>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '7px' }}>
          {draft.modelDirectories.length === 0 && (
            <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{t('settings.noFolders')}</p>
          )}
          {draft.modelDirectories.map((dir) => (
            <div
              key={dir}
              style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '10px', padding: '9px 12px', background: 'rgba(255,255,255,0.04)', borderRadius: 'var(--radius-md)' }}
            >
              <span style={{ fontSize: '12px', fontFamily: 'var(--font-mono)', wordBreak: 'break-all' }}>{dir}</span>
              <button
                className="btn-secondary"
                style={{ padding: '5px 10px' }}
                onClick={() => save({ modelDirectories: draft.modelDirectories.filter((d) => d !== dir) })}
              >
                <Trash2 size={13} />
              </button>
            </div>
          ))}
        </div>

        <div style={{ display: 'flex', gap: '8px', marginTop: '12px' }}>
          <input
            className="glass-input"
            style={{ flex: 1, fontSize: '12px' }}
            placeholder={t('settings.folderPlaceholder')}
            value={newDirectory}
            onChange={(e) => setNewDirectory(e.target.value)}
          />
          <button
            className="btn-primary"
            disabled={!newDirectory.trim() || saving}
            onClick={() => {
              save({ modelDirectories: [...draft.modelDirectories, newDirectory.trim()] });
              setNewDirectory('');
            }}
          >
            <FolderPlus size={15} /> {t('settings.add')}
          </button>
        </div>
      </div>

      {/* Runtime */}
      <div className="glass-card" style={{ padding: '22px' }}>
        <h3 style={{ fontSize: '15px', fontWeight: 800, marginBottom: '14px' }}>{t('settings.runtime')}</h3>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
          <div>
            <label style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text-secondary)' }}>
              {t('settings.apiPort')}
            </label>
            <input
              className="glass-input"
              type="number"
              style={{ width: '100%', marginTop: '5px', fontSize: '12px' }}
              value={draft.apiPort}
              onChange={(e) => setDraft({ ...draft, apiPort: Number(e.target.value) })}
              onBlur={() => save({ apiPort: draft.apiPort })}
            />
          </div>
          <div>
            <label style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text-secondary)' }}>
              {t('settings.gpuLayers')}
            </label>
            <input
              className="glass-input"
              type="number"
              style={{ width: '100%', marginTop: '5px', fontSize: '12px' }}
              value={draft.gpuLayers ?? ''}
              placeholder={t('settings.auto')}
              onChange={(e) => setDraft({ ...draft, gpuLayers: e.target.value === '' ? null : Number(e.target.value) })}
              onBlur={() => save({ gpuLayers: draft.gpuLayers })}
            />
          </div>

          <div style={{ gridColumn: '1 / -1' }}>
            <label style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text-secondary)' }}>
              {t('settings.contextLength')}
            </label>
            <input
              className="glass-input"
              type="number"
              style={{ width: '100%', marginTop: '5px', fontSize: '12px' }}
              value={draft.contextLength ?? ''}
              placeholder={t('settings.auto')}
              onChange={(e) => setDraft({ ...draft, contextLength: e.target.value === '' ? null : Number(e.target.value) })}
              onBlur={() => save({ contextLength: draft.contextLength })}
            />
            <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '5px' }}>
              {t('settings.contextLengthHint')}
            </p>
          </div>
        </div>

        <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '14px', lineHeight: 1.6 }}>
          {t('settings.privacy')}
        </p>
      </div>

      {saved && (
        <div style={{ fontSize: '12px', color: 'var(--accent-emerald)', fontWeight: 600 }}>{t('settings.saved')}</div>
      )}
    </div>
  );
};
