import React, { useState } from 'react';
import { Box, CheckCircle2, Cpu, Layers, Play, RefreshCw, Scissors, Search, XCircle } from 'lucide-react';
import { api, DiscoveredModel, LayerSweepResult, ModelInspection, streamPost } from '../api';
import { useAsync } from '../hooks';
import { EmptyState, ErrorBox, Spinner } from './Common';
import { TuningPanel } from './TuningPanel';
import { useI18n } from '../i18n';
import { formatBytes, formatNumber } from '../format';

interface Props {
  onModelLoaded: () => void;
}

export const ModelManager: React.FC<Props> = ({ onModelLoaded }) => {
  const { t } = useI18n();
  const models = useAsync(() => api.models(), []);
  const backends = useAsync(() => api.backends(), []);

  const [selected, setSelected] = useState<DiscoveredModel | null>(null);
  const [inspection, setInspection] = useState<ModelInspection | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [shardLog, setShardLog] = useState<string[]>([]);
  const [sweep, setSweep] = useState<LayerSweepResult | null>(null);
  const [sflowPath, setSflowPath] = useState<string | null>(null);
  const [preferredEngine, setPreferredEngine] = useState<string>('ailo-hierarchical');
  // Elapsed seconds while a load is in flight. A 200 GB model takes minutes to
  // fault in, and a button that just sits there reads as a hang.
  const [loadElapsed, setLoadElapsed] = useState(0);

  React.useEffect(() => {
    if (busy !== 'load') return;
    setLoadElapsed(0);
    const handle = window.setInterval(() => setLoadElapsed((s) => s + 1), 1000);
    return () => window.clearInterval(handle);
  }, [busy]);

  const select = async (model: DiscoveredModel) => {
    setSelected(model);
    setInspection(null);
    setSweep(null);
    setShardLog([]);
    setError(null);

    if (!model.inspectable) return;
    setBusy('inspect');
    try {
      setInspection(await api.inspectModel(model.id));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const load = async () => {
    if (!selected) return;
    setBusy('load');
    setError(null);
    try {
      await api.loadModel(selected.id, preferredEngine);
      backends.reload();
      onModelLoaded();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const shard = async () => {
    if (!selected?.filePath) return;
    setBusy('shard');
    setError(null);
    setShardLog([]);

    try {
      // Distribution is the whole point: shards written into one folder all sit
      // on the same physical disk and can never be read in parallel.
      await streamPost('/models/shard', { modelPath: selected.filePath, distributeAcrossDrives: true }, {
        start: (d) => setShardLog((l) => [...l, t('log.shardStart', { model: d.model, layers: d.layers, size: formatBytes(d.sizeBytes), directory: d.outputDirectory })]),
        progress: (d) => setShardLog((l) => [...l.slice(-40), t('log.shardProgress', { percent: d.percent, status: d.status, copied: formatBytes(d.bytesCopied || 0) })]),
        done: (d) => {
          setSflowPath(d.sflowPath);
          setShardLog((l) => [
            ...l,
            t('log.shardDone', { shards: d.shards, path: d.sflowPath }),
            t('log.shardDrives', { drives: (d.drivesUsed || []).join(', ') }),
            d.validation.valid
              ? t('log.integrityOk')
              : t('log.integrityFailed', { problems: d.validation.problems.slice(0, 3).join('; ') }),
          ]);
        },
        error: (d) => setError(d.message),
      });
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const runSweep = async () => {
    if (!sflowPath) return;
    setBusy('sweep');
    setError(null);
    try {
      await streamPost('/pipeline/sweep', { sflowPath }, {
        progress: (d) =>
          setShardLog((l) => [
            ...l.slice(-40),
            t('log.sweepLayer', { layer: d.layerIndex + 1, total: d.totalLayers, bandwidth: d.bandwidthMBps, cache: d.cacheHitRatePercent }),
          ]),
        done: (d) => setSweep(d),
        error: (d) => setError(d.message),
      });
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const list = models.data || [];
  const loadedId = backends.data?.loaded?.model.id;

  return (
    <div style={{ padding: '0 24px', display: 'grid', gridTemplateColumns: 'minmax(320px, 420px) 1fr', gap: '20px', alignItems: 'start' }}>
      {/* Model list */}
      <div className="glass-card" style={{ padding: '18px', display: 'flex', flexDirection: 'column', gap: '12px', maxHeight: 'calc(100vh - 190px)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h3 style={{ fontSize: '15px', fontWeight: 800, display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Box size={17} color="var(--accent-cyan)" /> {t('models.detected')}
          </h3>
          <button className="btn-secondary" style={{ padding: '6px 12px' }} onClick={() => models.reload()}>
            <RefreshCw size={14} /> {t('common.refresh')}
          </button>
        </div>

        {models.loading && <Spinner label={t('models.scanning')} />}
        {models.error && <ErrorBox message={models.error} />}

        {!models.loading && list.length === 0 && (
          <p style={{ fontSize: '12px', color: 'var(--text-muted)', lineHeight: 1.6 }}>
            {t('models.empty')}
          </p>
        )}

        <div style={{ overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {list.map((model) => {
            const isSelected = selected?.id === model.id;
            const isLoaded = loadedId === model.id;
            return (
              <button
                key={model.id}
                onClick={() => select(model)}
                style={{
                  textAlign: 'left',
                  background: isSelected ? 'rgba(0,242,254,0.12)' : 'rgba(255,255,255,0.04)',
                  border: `1px solid ${isSelected ? 'var(--accent-cyan)' : 'var(--bg-card-border)'}`,
                  borderRadius: 'var(--radius-md)',
                  padding: '11px 13px',
                  cursor: 'pointer',
                  color: 'var(--text-primary)',
                  fontFamily: 'var(--font-main)',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px', alignItems: 'center' }}>
                  <span style={{ fontSize: '13px', fontWeight: 700, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {model.displayName}
                  </span>
                  {isLoaded && <CheckCircle2 size={15} color="var(--accent-emerald)" />}
                </div>
                <div style={{ display: 'flex', gap: '6px', marginTop: '6px', flexWrap: 'wrap' }}>
                  <span className="badge-tag">{model.source}</span>
                  <span className="badge-tag">{formatBytes(model.fileSizeBytes)}</span>
                  {model.splitParts && (
                    <span
                      className="badge-tag"
                      style={
                        model.complete
                          ? undefined
                          : { color: 'var(--accent-amber)', borderColor: 'rgba(245,158,11,0.35)', background: 'rgba(245,158,11,0.1)' }
                      }
                    >
                      {t('models.splitParts', { n: model.splitParts.length })}
                    </span>
                  )}
                  {model.runnableWith.length === 0 && (
                    <span className="badge-tag" style={{ color: 'var(--accent-amber)', borderColor: 'rgba(245,158,11,0.35)', background: 'rgba(245,158,11,0.1)' }}>
                      {t('models.noEngine')}
                    </span>
                  )}
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Detail panel */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        {error && <ErrorBox message={error} />}

        {!selected && (
          <EmptyState
            title={t('models.selectTitle')}
            description={t('models.selectBody')}
          />
        )}

        {selected && (
          <div className="glass-card" style={{ padding: '22px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '14px', flexWrap: 'wrap' }}>
              <div>
                <h2 style={{ fontSize: '18px', fontWeight: 800 }}>{selected.displayName}</h2>
                <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '3px', wordBreak: 'break-all' }}>
                  {selected.filePath || t('models.noLocalFile')}
                </p>
              </div>

              <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center' }}>
                <select
                  value={preferredEngine}
                  onChange={(e) => setPreferredEngine(e.target.value)}
                  style={{
                    padding: '8px 12px',
                    borderRadius: '8px',
                    border: '1px solid var(--border-color)',
                    background: 'var(--bg-card)',
                    color: 'var(--text-primary)',
                    fontSize: '12px',
                    fontWeight: 600,
                  }}
                  disabled={busy !== null}
                >
                  <option value="ailo-hierarchical">⚡ AILOFlow Active Weights (Gerarchico)</option>
                  <option value="llama.cpp">🦙 llama.cpp Native Server</option>
                  <option value="ollama">🦙 Ollama Local Daemon</option>
                </select>

                <button
                  className="btn-primary"
                  onClick={load}
                  disabled={busy !== null || selected.runnableWith.length === 0}
                  title={selected.runnableWith.length === 0 ? t('models.noEngineForModel') : undefined}
                >
                  {busy === 'load' ? <RefreshCw size={15} className="spin" /> : <Play size={15} />}
                  {busy === 'load'
                    ? t('models.loading', { seconds: loadElapsed })
                    : t('models.load')}
                </button>

                {selected.source === 'gguf' || (selected.source === 'ollama' && selected.filePath) ? (
                  <button className="btn-secondary" onClick={shard} disabled={busy !== null}>
                    {busy === 'shard' ? <RefreshCw size={15} className="spin" /> : <Scissors size={15} />}
                    Shardizza in .sflow
                  </button>
                ) : null}

                {sflowPath && (
                  <button className="btn-secondary" onClick={runSweep} disabled={busy !== null}>
                    {busy === 'sweep' ? <RefreshCw size={15} className="spin" /> : <Layers size={15} />}
                    Misura pipeline
                  </button>
                )}
              </div>
            </div>

            {selected.splitParts && !selected.complete && (
              <p style={{ fontSize: '12px', color: 'var(--accent-amber)', marginTop: '12px', lineHeight: 1.6 }}>
                {t('models.incompleteSet')}
              </p>
            )}

            {selected.runnableWith.length === 0 && selected.complete && (
              <p style={{ fontSize: '12px', color: 'var(--accent-amber)', marginTop: '12px', lineHeight: 1.6 }}>
                {t('models.noEngineHint')}{' '}
                {backends.data?.backends.filter((b) => !b.available).map((b) => b.reason).join(' ')}
              </p>
            )}

            {busy === 'inspect' && <Spinner label={t('models.readingHeader')} />}

            {inspection && (
              <div style={{ marginTop: '18px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '12px' }}>
                <Field label={t('models.architecture')} value={inspection.architecture} />
                <Field label={t('models.parameters')} value={`${inspection.parameterCountBillions}B`} />
                <Field label={t('models.quantization')} value={inspection.quantization} />
                <Field label={t('models.layers')} value={String(inspection.blockCount)} />
                <Field label={t('models.context2')} value={inspection.contextLength.toLocaleString()} />
                <Field label={t('models.tensors')} value={inspection.tensorCount.toLocaleString()} />
                <Field label={t('models.fileSize')} value={formatBytes(inspection.fileSizeBytes)} />
                <Field label={t('models.weightsInRam')} value={formatBytes(inspection.estimatedRamRequiredBytes)} />
                <Field label={t('models.kvCache')} value={formatBytes(inspection.estimatedKvCacheBytes)} />
                <Field label={t('models.heads')} value={`${inspection.headCount} / ${inspection.headCountKv}`} />
                {inspection.splitPartCount > 1 && (
                  <Field
                    label={t('models.aggregatedOver')}
                    value={t('models.splitParts', { n: inspection.splitPartCount })}
                  />
                )}
              </div>
            )}

            {!inspection && !busy && !selected.inspectable && (
              <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '14px' }}>
                {t('models.notInspectable')}
              </p>
            )}
          </div>
        )}

        {/* Tuning belongs next to the model, not in global settings: the right
            values depend on this model's shape as much as on the machine. */}
        {selected && selected.source !== 'sflow' && selected.filePath && (
          <TuningPanel modelId={selected.id} />
        )}

        {shardLog.length > 0 && (
          <div className="glass-card" style={{ padding: '18px' }}>
            <h3 style={{ fontSize: '14px', fontWeight: 700, marginBottom: '10px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Search size={16} color="var(--accent-blue)" /> {t('models.operationLog')}
            </h3>
            <pre
              style={{
                fontSize: '11px',
                fontFamily: 'var(--font-mono)',
                color: 'var(--text-secondary)',
                maxHeight: '220px',
                overflowY: 'auto',
                whiteSpace: 'pre-wrap',
                lineHeight: 1.6,
              }}
            >
              {shardLog.join('\n')}
            </pre>
          </div>
        )}

        {sweep && (
          <div className="glass-card" style={{ padding: '22px' }}>
            <h3 style={{ fontSize: '15px', fontWeight: 800, marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Cpu size={17} color="var(--accent-emerald)" /> {t('models.pipelineResult')}
            </h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '12px' }}>
              <Field label={t('models.sweepLayers')} value={`${sweep.layersCompleted}/${sweep.totalLayers}`} />
              <Field label={t('models.sweepRead')} value={formatBytes(sweep.bytesReadFromStorage)} />
              <Field label={t('models.sweepBandwidth')} value={`${sweep.effectiveBandwidthMBps} MB/s`} />
              <Field label={t('models.sweepDuration')} value={`${(sweep.durationMs / 1000).toFixed(2)} s`} />
              <Field label={t('models.sweepLatency')} value={formatNumber(sweep.averageLayerLatencyMs, ' ms')} />
              <Field label={t('models.sweepCache')} value={`${sweep.cacheHitRatePercent}%`} />
              <Field label={t('models.sweepPrefetch')} value={String(sweep.prefetchTriggered)} />
              <Field label={t('models.sweepPerToken')} value={formatBytes(sweep.bytesPerTokenStreamed)} />
            </div>
            {sweep.errors.length > 0 && (
              <div style={{ marginTop: '14px', display: 'flex', gap: '8px', alignItems: 'flex-start' }}>
                <XCircle size={16} color="var(--accent-red)" style={{ flexShrink: 0, marginTop: '2px' }} />
                <span style={{ fontSize: '12px', color: 'var(--accent-red)' }}>
                  {t('models.sweepErrors', { count: sweep.errors.length, first: sweep.errors[0] })}
                </span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

const Field: React.FC<{ label: string; value: string }> = ({ label, value }) => (
  <div style={{ background: 'rgba(255,255,255,0.04)', borderRadius: 'var(--radius-md)', padding: '11px 13px' }}>
    <span style={{ fontSize: '10px', color: 'var(--text-muted)', fontWeight: 700, letterSpacing: '0.3px' }}>
      {label.toUpperCase()}
    </span>
    <div style={{ fontSize: '14px', fontWeight: 700, fontFamily: 'var(--font-mono)', marginTop: '3px', wordBreak: 'break-word' }}>
      {value}
    </div>
  </div>
);
