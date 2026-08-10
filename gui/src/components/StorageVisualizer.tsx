import React, { useState } from 'react';
import { Gauge, HardDrive, RefreshCw } from 'lucide-react';
import { api, BenchmarkResult, StorageDriveInfo } from '../api';
import { useAsync } from '../hooks';
import { EmptyState, ErrorBox, MeasuredBadge, MeterBar, Spinner } from './Common';
import { useI18n } from '../i18n';
import { formatBandwidth, formatBytes, formatNumber } from '../format';

export const StorageVisualizer: React.FC = () => {
  const { t } = useI18n();
  const storage = useAsync(() => api.storage(), []);
  const [benchmarking, setBenchmarking] = useState<string | null>(null);
  const [results, setResults] = useState<Record<string, BenchmarkResult>>({});
  const [error, setError] = useState<string | null>(null);

  const runBenchmark = async (mountPoint?: string) => {
    setBenchmarking(mountPoint || 'ALL');
    setError(null);
    try {
      const res = await api.benchmarkStorage(mountPoint);
      const next = { ...results };
      for (const r of res.results) next[r.mountPoint] = r;
      setResults(next);
      if (res.failures.length) {
        setError(res.failures.map((f) => `${f.mountPoint}: ${f.error}`).join('\n'));
      }
      storage.reload();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBenchmarking(null);
    }
  };

  const drives = storage.data?.drives || [];
  // Scale each bar against the fastest drive actually present.
  const peakMBps = Math.max(1, ...drives.map((d) => d.performanceProfile?.seqReadMBps || 0));

  return (
    <div style={{ padding: '0 24px', display: 'flex', flexDirection: 'column', gap: '18px' }}>
      <div className="glass-card" style={{ padding: '20px 24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '16px', flexWrap: 'wrap' }}>
        <div>
          <h2 style={{ fontSize: '17px', fontWeight: 800, display: 'flex', alignItems: 'center', gap: '9px' }}>
            <HardDrive size={19} color="var(--accent-amber)" /> {t('storage.title')}
          </h2>
          <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px' }}>
            {t('storage.summary', {
              count: drives.length,
              total: formatBytes(storage.data?.totalStorageBytes ?? null),
              bandwidth:
                storage.data?.measuredReadBandwidthMBps === null ||
                storage.data?.measuredReadBandwidthMBps === undefined
                  ? t('storage.notMeasured')
                  : formatBandwidth(storage.data.measuredReadBandwidthMBps),
            })}
          </p>
        </div>

        <button className="btn-primary" onClick={() => runBenchmark()} disabled={benchmarking !== null}>
          {benchmarking ? <RefreshCw size={16} className="spin" /> : <Gauge size={16} />}
          {benchmarking === 'ALL' ? t('storage.benchmarking') : t('storage.benchmarkAll')}
        </button>
      </div>

      {error && <ErrorBox message={error} />}
      {storage.error && <ErrorBox message={storage.error} />}
      {storage.loading && <Spinner label={t('storage.detecting')} />}

      {!storage.loading && drives.length === 0 && (
        <EmptyState
          title={t('storage.noDrives')}
          description={t('storage.noDrivesBody')}
        />
      )}

      {drives.map((drive) => (
        <DriveCard
          key={drive.id}
          drive={drive}
          peakMBps={peakMBps}
          benchmarking={benchmarking === drive.mountPoint}
          onBenchmark={() => runBenchmark(drive.mountPoint)}
          result={results[drive.mountPoint]}
        />
      ))}
    </div>
  );
};

const DriveCard: React.FC<{
  drive: StorageDriveInfo;
  peakMBps: number;
  benchmarking: boolean;
  onBenchmark: () => void;
  result?: BenchmarkResult;
}> = ({ drive, peakMBps, benchmarking, onBenchmark, result }) => {
  const { t } = useI18n();
  const perf = drive.performanceProfile;
  const usedBytes = drive.totalSizeBytes - drive.freeSizeBytes;
  const usedPercent = drive.totalSizeBytes > 0 ? (usedBytes / drive.totalSizeBytes) * 100 : null;
  const speedPercent = perf ? (perf.seqReadMBps / peakMBps) * 100 : null;

  return (
    <div className="glass-card" style={{ padding: '20px 24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '16px', flexWrap: 'wrap' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <h3 style={{ fontSize: '15px', fontWeight: 800 }}>{drive.mountPoint}</h3>
            <span className="badge-tag">{drive.type}</span>
            {perf && <MeasuredBadge measured={perf.measured} />}
          </div>
          <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '3px' }}>
            {drive.label} · {drive.filesystem} · {drive.devicePath}
          </p>
        </div>

        <button className="btn-secondary" onClick={onBenchmark} disabled={benchmarking}>
          {benchmarking ? <RefreshCw size={15} className="spin" /> : <Gauge size={15} />}
          {benchmarking ? 'Misurazione...' : 'Misura questo disco'}
        </button>
      </div>

      <div style={{ marginTop: '16px', display: 'grid', gap: '14px' }}>
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '5px' }}>
            <span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>{t('storage.usage')}</span>
            <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>
              {formatBytes(usedBytes)} / {formatBytes(drive.totalSizeBytes)}
            </span>
          </div>
          <MeterBar percent={usedPercent} color="var(--accent-blue)" height={8} />
        </div>

        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '5px' }}>
            <span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>{t('storage.seqRead')}</span>
            <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-amber)' }}>
              {perf ? formatBandwidth(perf.seqReadMBps) : 'n/d'}
            </span>
          </div>
          <MeterBar percent={speedPercent} color="var(--accent-amber)" height={8} />
        </div>
      </div>

      {perf && (
        <div style={{ marginTop: '16px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: '12px' }}>
          <Metric label={t('storage.seqWrite')} value={formatBandwidth(perf.seqWriteMBps)} />
          <Metric label={t('storage.randRead')} value={formatBandwidth(perf.randReadMBps)} />
          <Metric label={t('storage.latency')} value={formatNumber(perf.latencyUs, ' µs', 0)} />
          <Metric label={t('storage.iops')} value={perf.iops.toLocaleString()} />
          <Metric label={t('storage.free')} value={formatBytes(drive.freeSizeBytes)} />
        </div>
      )}

      {!perf?.measured && (
        <p style={{ fontSize: '11px', color: 'var(--accent-amber)', marginTop: '12px' }}>
          {t('storage.estimateWarning')}
        </p>
      )}

      {result && (
        <p style={{ fontSize: '11px', color: 'var(--accent-emerald)', marginTop: '10px' }}>
          {t('storage.benchmarkDone', {
            duration: `${(result.benchmarkDurationMs / 1000).toFixed(1)} s`,
            time: new Date(result.timestamp).toLocaleTimeString(),
          })}
        </p>
      )}
    </div>
  );
};

const Metric: React.FC<{ label: string; value: string }> = ({ label, value }) => (
  <div style={{ background: 'rgba(255,255,255,0.04)', borderRadius: 'var(--radius-md)', padding: '10px 12px' }}>
    <span style={{ fontSize: '10px', color: 'var(--text-muted)', fontWeight: 700, letterSpacing: '0.3px' }}>
      {label.toUpperCase()}
    </span>
    <div style={{ fontSize: '14px', fontWeight: 700, fontFamily: 'var(--font-mono)', marginTop: '2px' }}>{value}</div>
  </div>
);
