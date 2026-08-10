import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Navbar, NavTab } from './components/Navbar';
import { DashboardView } from './components/DashboardView';
import { StorageVisualizer } from './components/StorageVisualizer';
import { ModelManager } from './components/ModelManager';
import { CatalogView } from './components/CatalogView';
import { ChatView } from './components/ChatView';
import { SettingsView } from './components/SettingsView';
import { BottleneckBanner } from './components/BottleneckBanner';
import { DownloadsPanel } from './components/DownloadsPanel';
import { useMetricsStream } from './hooks';
import { useDownloads } from './downloads';
import { api } from './api';
import {
  I18nContext,
  LANGUAGE_STORAGE_KEY,
  Language,
  createTranslator,
  detectInitialLanguage,
  useI18n,
} from './i18n';

export const App: React.FC = () => {
  const [language, setLanguageState] = useState<Language>(() => detectInitialLanguage());

  const setLanguage = useCallback((next: Language) => {
    setLanguageState(next);
    try {
      localStorage.setItem(LANGUAGE_STORAGE_KEY, next);
    } catch {
      // Storage unavailable; the choice still applies for this session.
    }
    document.documentElement.lang = next;
  }, []);

  useEffect(() => {
    document.documentElement.lang = language;
  }, [language]);

  const i18n = useMemo(
    () => ({ language, setLanguage, t: createTranslator(language) }),
    [language, setLanguage]
  );

  return (
    <I18nContext.Provider value={i18n}>
      <Shell />
    </I18nContext.Provider>
  );
};

const Shell: React.FC = () => {
  const { t } = useI18n();
  const [activeTab, setActiveTab] = useState<NavTab>('dashboard');
  const { snapshot, connection } = useMetricsStream();
  const downloads = useDownloads();
  const [showDownloads, setShowDownloads] = useState(false);
  const [loadedModelName, setLoadedModelName] = useState<string | null>(null);
  // Remounts the model-dependent views after a load so they refetch state.
  const [modelEpoch, setModelEpoch] = useState(0);

  const refreshLoadedModel = useCallback(async () => {
    try {
      const { loaded } = await api.backends();
      setLoadedModelName(loaded ? loaded.model.displayName : null);
    } catch {
      setLoadedModelName(null);
    }
  }, []);

  useEffect(() => {
    if (connection === 'online') void refreshLoadedModel();
  }, [connection, modelEpoch, refreshLoadedModel]);

  const handleModelLoaded = useCallback(() => setModelEpoch((n) => n + 1), []);

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        connection={connection}
        loadedModelName={loadedModelName}
        downloads={downloads.jobs}
        onToggleDownloads={() => setShowDownloads((v) => !v)}
      />

      {/* Downloads stay reachable from every screen, because the transfer
          itself is owned by the runtime rather than by this view. */}
      {showDownloads && (
        <div className="glass-card" style={{ margin: '0 24px 16px 24px', padding: '18px 20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
            <h3 style={{ fontSize: '14px', fontWeight: 800 }}>{t('dl.active')}</h3>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{t('dl.keepsRunning')}</span>
          </div>
          <DownloadsPanel jobs={downloads.jobs} onChanged={downloads.refresh} />
        </div>
      )}

      {(activeTab === 'dashboard' || activeTab === 'storage') && snapshot && (
        <BottleneckBanner bottleneck={snapshot.bottleneck} />
      )}

      <main style={{ flex: 1, paddingBottom: '30px' }}>
        {activeTab === 'dashboard' && (
          <DashboardView key={`dash-${modelEpoch}`} snapshot={snapshot} offline={connection === 'offline'} />
        )}
        {activeTab === 'storage' && <StorageVisualizer />}
        {activeTab === 'models' && <ModelManager key={`models-${modelEpoch}`} onModelLoaded={handleModelLoaded} />}
        {activeTab === 'catalog' && <CatalogView downloads={downloads} />}
        {activeTab === 'chat' && <ChatView key={`chat-${modelEpoch}`} />}
        {activeTab === 'settings' && <SettingsView />}
      </main>
    </div>
  );
};

export default App;
