import React from 'react';
import { Activity, Cpu, Database, HardDrive, Layers, Zap } from 'lucide-react';
import { api, LoadedModelState, TelemetrySnapshot } from '../api';
import { useAsync } from '../hooks';
import { EmptyState, ErrorBox, MeasuredBadge, MeterBar, Spinner, StatCard } from './Common';
import { useI18n } from '../i18n';
import {
  formatBandwidth,
  formatBytes,
  formatDuration,
  formatNumber,
  formatPercent,
  formatRelativeTime,
  hasValue,
} from '../format';

interface Props {
  snapshot: TelemetrySnapshot | null;
  offline: boolean;
}

export const DashboardView: React.FC<Props> = ({ snapshot, offline }) => {
  const { t } = useI18n();
  const system = useAsync(() => api.system(), []);
  const backends = useAsync(() => api.backends(), []);
  const sweep = useAsync(() => api.lastSweep(), []);

  if (offline) {
    return (
      <div style={{ padding: '0 24px' }}>
        <EmptyState title={t('dash.offlineTitle')} description={t('dash.offlineBody')} />
      </div>
    );
  }

  const profile = system.data?.profile;
  const optimizer = system.data?.optimizer;
  const loaded: LoadedModelState | null = backends.data?.loaded ?? null;
  const gen = snapshot?.generation;
  const lastSweep = sweep.data?.lastSweep ?? null;

  const gauges = [
    {
      label: 'CPU',
      percent: snapshot ? snapshot.cpuUsagePercent : null,
      value: snapshot ? formatPercent(snapshot.cpuUsagePercent, 1) : '—',
      color: 'var(--accent-cyan)',
      icon: Cpu,
      hint: profile ? t('dash.cpuNote', { threads: profile.cpu.logicalThreads, simd: profile.cpu.recommendedOptimization }) : '',
    },
    {
      label: 'RAM',
      percent: snapshot ? snapshot.ramUsagePercent : null,
      value: snapshot
        ? `${formatBytes(snapshot.ramUsedBytes)} / ${formatBytes(snapshot.ramTotalBytes)}`
        : '—',
      color: 'var(--accent-blue)',
      icon: Database,
      hint: profile?.ram.speedMHz ? `${profile.ram.memoryType || 'RAM'} @ ${profile.ram.speedMHz} MHz` : '',
    },
    {
      label: 'GPU',
      percent: snapshot?.gpuUsagePercent ?? null,
      value: snapshot && hasValue(snapshot.gpuUsagePercent) ? formatPercent(snapshot.gpuUsagePercent) : t('common.notAvailable'),
      color: 'var(--accent-purple)',
      icon: Zap,
      hint: snapshot && !hasValue(snapshot.gpuUsagePercent)
        ? t('dash.gpuNoCounter')
        : profile?.gpus[0]?.model || '',
    },
    {
      label: 'VRAM',
      percent: snapshot?.vramUsagePercent ?? null,
      value:
        snapshot && hasValue(snapshot.vramUsedBytes) && hasValue(snapshot.vramTotalBytes)
          ? `${formatBytes(snapshot.vramUsedBytes)} / ${formatBytes(snapshot.vramTotalBytes)}`
          : t('common.notAvailable'),
      color: 'var(--accent-magenta)',
      icon: Activity,
      hint: snapshot && hasValue(snapshot.gpuTemperatureC) ? `${snapshot.gpuTemperatureC} °C` : '',
    },
    {
      label: 'STORAGE PIPELINE',
      percent:
        snapshot && hasValue(snapshot.storage.bandwidthMBps) && profile?.measuredStorageReadBandwidthMBps
          ? (snapshot.storage.bandwidthMBps / profile.measuredStorageReadBandwidthMBps) * 100
          : null,
      value: snapshot ? formatBandwidth(snapshot.storage.bandwidthMBps) : '—',
      color: 'var(--accent-amber)',
      icon: HardDrive,
      hint:
        snapshot && !hasValue(snapshot.storage.bandwidthMBps)
          ? t('dash.pipelineIdle')
          : t('dash.storageHint', { iops: snapshot?.storage.iops ?? 0, queue: snapshot?.storage.queueDepth ?? 0 }),
    },
  ];

  return (
    <div style={{ padding: '0 24px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {system.error && <ErrorBox message={system.error} />}

      {/* Top row: what is actually running */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(230px, 1fr))', gap: '16px' }}>
        <div className="glass-card" style={{ padding: '18px 20px' }}>
          <span style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text-muted)' }}>{t('dash.loadedModel')}</span>
          {loaded ? (
            <>
              <h2 style={{ fontSize: '18px', fontWeight: 800, margin: '6px 0', color: 'var(--text-primary)' }}>
                {loaded.model.displayName}
              </h2>
              <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                <span className="badge-tag">{loaded.backendId}</span>
                <span className="badge-tag">{formatBytes(loaded.model.fileSizeBytes)}</span>
              </div>
              <span style={{ fontSize: '11px', color: 'var(--text-muted)', display: 'block', marginTop: '8px' }}>
                {t('dash.loadedIn', { duration: formatDuration(loaded.loadDurationMs) })}
              </span>
            </>
          ) : (
            <>
              <h2 style={{ fontSize: '18px', fontWeight: 700, margin: '6px 0', color: 'var(--text-muted)' }}>
                {t('common.none')}
              </h2>
              <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                {t('dash.chooseModel')}
              </span>
            </>
          )}
        </div>

        <StatCard
          label={t('dash.genSpeed')}
          value={gen && hasValue(gen.tokensPerSecond) ? `${gen.tokensPerSecond}` : '—'}
          hint={
            gen && hasValue(gen.tokensPerSecond)
              ? t('dash.genSpeedHint', { when: formatRelativeTime(gen.measuredAt) })
              : t('dash.noGeneration')
          }
          unavailable={!gen || !hasValue(gen.tokensPerSecond)}
        />

        <StatCard
          label={t('dash.firstToken')}
          value={gen && hasValue(gen.firstTokenLatencyMs) ? formatDuration(gen.firstTokenLatencyMs) : '—'}
          accent="var(--accent-blue)"
          hint={gen && hasValue(gen.promptTokensPerSecond) ? t('dash.promptRate', { rate: gen.promptTokensPerSecond }) : t('dash.awaitingData')}
          unavailable={!gen || !hasValue(gen.firstTokenLatencyMs)}
        />

        <StatCard
          label={t('dash.cache')}
          value={snapshot?.cache && hasValue(snapshot.cache.hitRatePercent) ? formatPercent(snapshot.cache.hitRatePercent, 1) : '—'}
          accent="var(--accent-emerald)"
          hint={
            snapshot?.cache
              ? t('dash.cacheHint', { hits: snapshot.cache.hits, misses: snapshot.cache.misses })
              : t('dash.cacheIdle')
          }
          unavailable={!snapshot?.cache || !hasValue(snapshot.cache.hitRatePercent)}
        />
      </div>

      {/* Live hardware */}
      <div className="glass-card" style={{ padding: '24px' }}>
        <h3 style={{ fontSize: '15px', fontWeight: 700, marginBottom: '18px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Activity size={17} color="var(--accent-cyan)" /> {t('dash.hardware')}
          {snapshot && (
            <span style={{ fontSize: '11px', fontWeight: 500, color: 'var(--text-muted)', marginLeft: 'auto' }}>
              {t('dash.sampledAt', { time: new Date(snapshot.timestamp).toLocaleTimeString() })}
            </span>
          )}
        </h3>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {gauges.map((g) => {
            const Icon = g.icon;
            return (
              <div key={g.label}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px', gap: '12px' }}>
                  <span style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <Icon size={14} color={g.color} /> {g.label}
                    {g.hint && (
                      <span style={{ fontWeight: 400, color: 'var(--text-muted)', fontSize: '11px' }}>· {g.hint}</span>
                    )}
                  </span>
                  <span style={{ fontSize: '12px', fontWeight: 700, fontFamily: 'var(--font-mono)', color: hasValue(g.percent) ? g.color : 'var(--text-muted)' }}>
                    {g.value}
                  </span>
                </div>
                <MeterBar percent={g.percent} color={g.color} />
              </div>
            );
          })}
        </div>
      </div>

      {/* Every detected GPU, each with its own live counters */}
      <div className="glass-card" style={{ padding: '22px' }}>
        <h3 style={{ fontSize: '15px', fontWeight: 700, marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Zap size={17} color="var(--accent-purple)" /> {t('dash.gpus')}
          {profile && <span className="badge-tag">{profile.gpus.length}</span>}
        </h3>

        {system.loading && <Spinner />}

        {profile && profile.gpus.length === 0 && (
          <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
            {t('dash.noGpu')}
          </p>
        )}

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '12px' }}>
          {profile?.gpus.map((gpu) => {
            // Prefer a per-adapter sample; otherwise fall back to the
            // system-wide counter row, which is explicitly labelled as such.
            const own = snapshot?.gpus.find((s) => s.adapterKey === gpu.id);
            const shared = snapshot?.gpus.find((s) => s.adapterKey === null);
            const live = own || shared || null;
            const isShared = !own && !!shared;
            const vramTotal = gpu.vramTotalBytes ?? live?.memoryTotalBytes ?? null;
            const vramUsed = live?.memoryUsedBytes ?? null;
            const vramPercent = vramUsed !== null && vramTotal ? (vramUsed / vramTotal) * 100 : null;

            return (
              <div key={gpu.id} style={{ background: 'rgba(255,255,255,0.04)', borderRadius: 'var(--radius-md)', padding: '14px 16px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px', alignItems: 'flex-start' }}>
                  <div>
                    <div style={{ fontSize: '13px', fontWeight: 700 }}>{gpu.model}</div>
                    <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>
                      {gpu.vendor} · {gpu.supportedBackends.join('/')} → {gpu.recommendedBackend}
                    </div>
                  </div>
                  {hasValue(live?.temperatureC ?? null) && (
                    <span className="badge-tag">{live!.temperatureC} °C</span>
                  )}
                </div>

                <div style={{ marginTop: '12px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', marginBottom: '4px' }}>
                    <span style={{ color: 'var(--text-secondary)' }}>
                      {isShared ? t('dash.utilizationShared') : t('dash.utilization')}
                    </span>
                    <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-purple)' }}>
                      {formatPercent(live?.utilizationPercent ?? null, 1)}
                    </span>
                  </div>
                  <MeterBar percent={live?.utilizationPercent ?? null} color="var(--accent-purple)" height={7} />
                </div>

                <div style={{ marginTop: '10px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', marginBottom: '4px' }}>
                    <span style={{ color: 'var(--text-secondary)' }}>VRAM</span>
                    <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-magenta)' }}>
                      {vramUsed !== null ? `${formatBytes(vramUsed)} / ` : ''}
                      {vramTotal !== null ? formatBytes(vramTotal) : 'n/d'}
                    </span>
                  </div>
                  <MeterBar percent={vramPercent} color="var(--accent-magenta)" height={7} />
                  <span style={{ fontSize: '10px', color: 'var(--text-muted)', display: 'block', marginTop: '4px' }}>
                    {t('dash.vramSource', { source: gpu.vramSource })}{live ? ` · ${t('dash.counters', { source: live.source })}` : ''}
                  </span>
                </div>

                {live?.engineBreakdown && Object.keys(live.engineBreakdown).length > 0 && (
                  <div style={{ marginTop: '10px', display: 'flex', flexWrap: 'wrap', gap: '5px' }}>
                    {Object.entries(live.engineBreakdown)
                      .filter(([, v]) => v > 0.05)
                      .sort((a, b) => b[1] - a[1])
                      .slice(0, 4)
                      .map(([engine, value]) => (
                        <span key={engine} className="badge-tag" style={{ textTransform: 'none' }}>
                          {engine} {value}%
                        </span>
                      ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {snapshot?.gpuUnavailableReason && (
          <p style={{ fontSize: '11px', color: 'var(--accent-amber)', marginTop: '12px' }}>
            {snapshot.gpuUnavailableReason}
          </p>
        )}
      </div>

      {/* Automatic configuration — real values from the optimizer */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '16px' }}>
        <div className="glass-card" style={{ padding: '22px' }}>
          <h3 style={{ fontSize: '15px', fontWeight: 700, marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Layers size={17} color="var(--accent-blue)" /> {t('dash.autoConfig')}
          </h3>

          {system.loading && <Spinner />}
          {optimizer && profile && (
            <div style={{ display: 'grid', gap: '10px', fontSize: '13px' }}>
              <Row label={t('dash.backend')} value={optimizer.backend} />
              <Row label={t('dash.ramCache')} value={formatBytes(optimizer.ramCacheBytes)} />
              <Row
                label={t('dash.vramCache')}
                value={optimizer.vramCacheBytes > 0 ? formatBytes(optimizer.vramCacheBytes) : t('dash.vramNone')}
              />
              <Row label={t('dash.threads')} value={String(optimizer.threadCount)} />
              <Row label={t('dash.prefetch')} value={t('dash.prefetchLayers', { n: optimizer.prefetchDepthLayers })} />
              <Row label={t('dash.ioQueue')} value={String(optimizer.ioQueueDepth)} />
              <Row label={t('dash.shardStrategy')} value={optimizer.shardingStrategy} />
              <div style={{ marginTop: '4px' }}>
                <MeasuredBadge measured={optimizer.basedOnMeasuredStorage} />
                <span style={{ fontSize: '11px', color: 'var(--text-muted)', marginLeft: '8px' }}>
                  {optimizer.basedOnMeasuredStorage
                    ? t('dash.tunedMeasured')
                    : t('dash.tunedEstimated')}
                </span>
              </div>
            </div>
          )}
        </div>

        <div className="glass-card" style={{ padding: '22px' }}>
          <h3 style={{ fontSize: '15px', fontWeight: 700, marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <HardDrive size={17} color="var(--accent-amber)" /> {t('dash.pipeline')}
          </h3>

          {lastSweep ? (
            <div style={{ display: 'grid', gap: '10px', fontSize: '13px' }}>
              <Row label={t('dash.pipelineModel')} value={lastSweep.modelName} />
              <Row label={t('dash.pipelineLayers')} value={`${lastSweep.layersCompleted}/${lastSweep.totalLayers}`} />
              <Row label={t('dash.pipelineRead')} value={formatBytes(lastSweep.bytesReadFromStorage)} />
              <Row label={t('dash.pipelineBandwidth')} value={formatBandwidth(lastSweep.effectiveBandwidthMBps)} />
              <Row label={t('dash.pipelineLatency')} value={formatNumber(lastSweep.averageLayerLatencyMs, ' ms')} />
              <Row label={t('dash.pipelineCache')} value={formatPercent(lastSweep.cacheHitRatePercent, 1)} />
              <Row label={t('dash.pipelinePerToken')} value={formatBytes(lastSweep.bytesPerTokenStreamed)} />
              <Row label={t('dash.pipelineDrives')} value={lastSweep.drivesUsed.join(', ') || t('common.none')} />
            </div>
          ) : (
            <p style={{ fontSize: '13px', color: 'var(--text-muted)', lineHeight: 1.6 }}>
              {t('dash.pipelineEmpty')}
            </p>
          )}
        </div>
      </div>
    </div>
  );
};

const Row: React.FC<{ label: string; value: string }> = ({ label, value }) => (
  <div style={{ display: 'flex', justifyContent: 'space-between', gap: '16px', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '6px' }}>
    <span style={{ color: 'var(--text-secondary)' }}>{label}</span>
    <span style={{ fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)', textAlign: 'right' }}>
      {value}
    </span>
  </div>
);
