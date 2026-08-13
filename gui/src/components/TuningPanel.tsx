import React, { useEffect, useState } from 'react';
import { AlertTriangle, RotateCcw, Sliders } from 'lucide-react';
import { API_BASE, TuningPlan, TuningOverrides, TuningPreset } from '../api';
import { useI18n } from '../i18n';
import { ErrorBox, Spinner } from './Common';
import { formatBytes, formatNumber } from '../format';

interface Props {
  modelId: string;
}

/**
 * Hardware-aware tuning, with the last word left to the user.
 *
 * Each knob shows what the runtime chose and why, and can be overridden. The
 * consequences panel updates on every change, because the decisions that matter
 * here — how many experts a token routes to, how much context to keep — trade
 * quality against speed in ways only the person running the model can weigh.
 */
export const TuningPanel: React.FC<Props> = ({ modelId }) => {
  const { t } = useI18n();
  const [plan, setPlan] = useState<TuningPlan | null>(null);
  const [presets, setPresets] = useState<TuningPreset[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchPlan = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/tuning?id=${encodeURIComponent(modelId)}`);
      const body = await res.json();
      if (body.error) throw new Error(body.error);
      setPlan(body.plan);
      setPresets(body.presets || []);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void fetchPlan();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelId]);

  const setOverride = async (patch: Partial<TuningOverrides>) => {
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/tuning`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: modelId, overrides: patch }),
      });
      const body = await res.json();
      if (body.error) throw new Error(body.error);
      if (body.plan) setPlan(body.plan);
    } catch (err) {
      setError((err as Error).message);
    }
  };

  if (loading) return <div className="glass-card" style={{ padding: '20px' }}><Spinner /></div>;
  if (error) return <ErrorBox message={error} />;
  if (!plan) return null;

  const p = plan.projection;
  const experts = plan.expertsPerToken;
  const reduced = experts && experts.effective < (plan.model.expertsUsedByDefault ?? 0);

  return (
    <div className="glass-card" style={{ padding: '22px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '12px', marginBottom: '6px' }}>
        <h3 style={{ fontSize: '15px', fontWeight: 800, display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Sliders size={17} color="var(--accent-cyan)" /> {t('tuning.title')}
        </h3>
        <button
          className="btn-secondary"
          style={{ padding: '6px 11px', fontSize: '12px' }}
          onClick={() =>
            setOverride({
              expertsPerToken: null, gpuLayers: null, cpuMoeLayers: null, contextLength: null,
              kvCacheType: null, threads: null, batchSize: null, ubatchSize: null,
              flashAttention: null, loadMode: null,
            })
          }
        >
          <RotateCcw size={13} /> {t('tuning.resetAll')}
        </button>
      </div>

      <p style={{ fontSize: '12px', color: 'var(--text-muted)', lineHeight: 1.6, marginBottom: '16px' }}>
        {plan.diskBound ? t('tuning.diskBound') : t('tuning.fitsInMemory')}
      </p>

      {/* Starting points, not recommendations: which one wins depends on this
          machine's memory pressure and on how much quality loss is acceptable. */}
      {presets.length > 1 && (
        <div style={{ display: 'flex', gap: '7px', flexWrap: 'wrap', marginBottom: '16px' }}>
          {presets.map((preset) => (
            <button
              key={preset.id}
              className="btn-secondary"
              style={{ padding: '7px 12px', fontSize: '12px' }}
              title={preset.description}
              onClick={() =>
                setOverride({
                  expertsPerToken: null, gpuLayers: null, cpuMoeLayers: null, contextLength: null,
                  kvCacheType: null, threads: null, batchSize: null, ubatchSize: null,
                  flashAttention: null, loadMode: null,
                  ...preset.overrides,
                })
              }
            >
              {preset.label}
            </button>
          ))}
        </div>
      )}

      {/* Experts first: on a large MoE it is the only knob that changes the
          order of magnitude of what a token costs. */}
      {experts && plan.model.expertCount && (
        <div
          style={{
            padding: '15px 16px',
            borderRadius: 'var(--radius-md)',
            background: reduced ? 'rgba(245,158,11,0.08)' : 'rgba(0,242,254,0.07)',
            border: `1px solid ${reduced ? 'rgba(245,158,11,0.35)' : 'rgba(0,242,254,0.25)'}`,
            marginBottom: '16px',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: '12px' }}>
            <span style={{ fontSize: '13px', fontWeight: 700 }}>{t('tuning.experts')}</span>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '15px', fontWeight: 800, color: reduced ? 'var(--accent-amber)' : 'var(--accent-cyan)' }}>
              {experts.effective} / {plan.model.expertCount}
            </span>
          </div>

          <input
            type="range"
            min={1}
            max={plan.model.expertsUsedByDefault ?? 8}
            step={1}
            value={experts.effective}
            onChange={(e) => setOverride({ expertsPerToken: Number(e.target.value) })}
            style={{ width: '100%', marginTop: '10px', accentColor: reduced ? 'var(--accent-amber)' : 'var(--accent-cyan)' }}
          />

          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
            <span>1</span>
            <span>{t('tuning.modelDefault', { n: plan.model.expertsUsedByDefault ?? 0 })}</span>
          </div>

          <p style={{ fontSize: '11.5px', color: 'var(--text-secondary)', marginTop: '10px', lineHeight: 1.6 }}>
            {t('tuning.expertsEffect', {
              active: p.activeParamsB ?? 0,
              total: plan.model.totalParamsB,
              perToken: formatBytes(p.bytesPerToken),
            })}
          </p>

          {experts.override !== null && (
            <button
              className="btn-secondary"
              style={{ padding: '5px 10px', fontSize: '11px', marginTop: '9px' }}
              onClick={() => setOverride({ expertsPerToken: null })}
            >
              <RotateCcw size={12} /> {t('tuning.backToAuto')}
            </button>
          )}
        </div>
      )}

      {/* Consequences of the current plan */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '10px', marginBottom: '16px' }}>
        <Cell label={t('tuning.bytesPerToken')} value={formatBytes(p.bytesPerToken)} />
        <Cell
          label={t('tuning.activeParams')}
          value={p.activeParamsB !== null ? `${p.activeParamsB}B` : '—'}
          accent={reduced ? 'var(--accent-amber)' : undefined}
        />
        <Cell label={t('tuning.fromStorage')} value={formatBytes(p.bytesFromStorage)} />
        <Cell label={t('tuning.kvCache')} value={formatBytes(p.kvCacheBytes)} />
        <Cell
          label={t('tuning.storageCeiling')}
          value={p.storageCeilingTokensPerSecond !== null ? formatNumber(p.storageCeilingTokensPerSecond, ' tok/s', 2) : '—'}
          accent="var(--accent-emerald)"
        />
      </div>

      {/* The rest of the knobs */}
      <div style={{ display: 'grid', gap: '9px' }}>
        <NumberKnob label={t('tuning.gpuLayers')} decision={plan.gpuLayers} max={plan.model.layers}
          onChange={(v) => setOverride({ gpuLayers: v })} />
        <NumberKnob label={t('tuning.cpuMoeLayers')} decision={plan.cpuMoeLayers} max={plan.model.layers}
          onChange={(v) => setOverride({ cpuMoeLayers: v })} />
        <NumberKnob label={t('tuning.contextLength')} decision={plan.contextLength} max={plan.model.contextLength}
          onChange={(v) => setOverride({ contextLength: v })} />
        <NumberKnob label={t('tuning.threads')} decision={plan.threads} max={256}
          onChange={(v) => setOverride({ threads: v })} />
        <ChoiceKnob label={t('tuning.kvCacheType')} decision={plan.kvCacheType} options={['f16', 'q8_0', 'q4_0']}
          onChange={(v) => setOverride({ kvCacheType: v as TuningOverrides['kvCacheType'] })} />
        <ChoiceKnob label={t('tuning.flashAttention')} decision={plan.flashAttention} options={['auto', 'on', 'off']}
          onChange={(v) => setOverride({ flashAttention: v as TuningOverrides['flashAttention'] })} />
        <ChoiceKnob label={t('tuning.loadMode')} decision={plan.loadMode} options={['mmap', 'mlock', 'mmap+mlock']}
          onChange={(v) => setOverride({ loadMode: v as TuningOverrides['loadMode'] })} />
      </div>

      {plan.warnings.map((w, i) => (
        <div key={i} style={{ display: 'flex', gap: '8px', alignItems: 'flex-start', marginTop: '12px' }}>
          <AlertTriangle size={15} color="var(--accent-amber)" style={{ flexShrink: 0, marginTop: '2px' }} />
          <span style={{ fontSize: '11.5px', color: 'var(--accent-amber)', lineHeight: 1.6 }}>{w}</span>
        </div>
      ))}

      <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '14px', lineHeight: 1.6 }}>
        {t('tuning.appliesOnLoad')}
      </p>
    </div>
  );
};

const Cell: React.FC<{ label: string; value: string; accent?: string }> = ({ label, value, accent }) => (
  <div style={{ background: 'rgba(255,255,255,0.04)', borderRadius: 'var(--radius-md)', padding: '10px 12px' }}>
    <div style={{ fontSize: '10px', color: 'var(--text-muted)', fontWeight: 700 }}>{label.toUpperCase()}</div>
    <div style={{ fontSize: '14px', fontWeight: 700, fontFamily: 'var(--font-mono)', marginTop: '3px', color: accent || 'var(--text-primary)' }}>
      {value}
    </div>
  </div>
);

interface Decision<T> {
  effective: T;
  auto: T;
  override: T | null;
  reason: string;
}

const KnobRow: React.FC<{ label: string; reason: string; isOverride: boolean; children: React.ReactNode; onReset: () => void }> = ({
  label, reason, isOverride, children, onReset,
}) => {
  const { t } = useI18n();
  return (
    <div style={{ background: 'rgba(255,255,255,0.03)', borderRadius: 'var(--radius-md)', padding: '10px 12px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '10px' }}>
        <span style={{ fontSize: '12.5px', fontWeight: 600, color: 'var(--text-secondary)' }}>{label}</span>
        <div style={{ display: 'flex', alignItems: 'center', gap: '7px' }}>
          {children}
          <span
            className="badge-tag"
            style={
              isOverride
                ? { color: 'var(--accent-amber)', borderColor: 'rgba(245,158,11,0.35)', background: 'rgba(245,158,11,0.12)' }
                : undefined
            }
          >
            {isOverride ? t('tuning.user') : t('tuning.auto')}
          </span>
          {isOverride && (
            <button className="btn-secondary" style={{ padding: '3px 6px' }} onClick={onReset} title={t('tuning.backToAuto')}>
              <RotateCcw size={11} />
            </button>
          )}
        </div>
      </div>
      <p style={{ fontSize: '10.5px', color: 'var(--text-muted)', marginTop: '5px', lineHeight: 1.5 }}>{reason}</p>
    </div>
  );
};

/** The runtime sends no flag for this value; the engine decides it itself. */
const AUTO_ENGINE = -1;

const NumberKnob: React.FC<{ label: string; decision: Decision<number>; max: number; onChange: (v: number | null) => void }> = ({
  label, decision, max, onChange,
}) => {
  const { t } = useI18n();
  const engineDecides = decision.effective === AUTO_ENGINE;

  return (
    <KnobRow label={label} reason={decision.reason} isOverride={decision.override !== null} onReset={() => onChange(null)}>
      {engineDecides && (
        <span style={{ fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
          {t('tuning.engineDecides')}
        </span>
      )}
      <input
        type="number"
        className="glass-input"
        min={0}
        max={max}
        // An empty field is how "let the engine decide" is expressed, since
        // showing the sentinel as -1 would read as a real setting.
        value={engineDecides ? '' : decision.effective}
        placeholder={t('tuning.engineDecides')}
        onChange={(e) => onChange(e.target.value === '' ? null : Number(e.target.value))}
        style={{ width: '92px', fontSize: '12px', padding: '4px 8px', textAlign: 'right' }}
      />
    </KnobRow>
  );
};

const ChoiceKnob: React.FC<{ label: string; decision: Decision<string>; options: string[]; onChange: (v: string | null) => void }> = ({
  label, decision, options, onChange,
}) => (
  <KnobRow label={label} reason={decision.reason} isOverride={decision.override !== null} onReset={() => onChange(null)}>
    <select
      className="glass-input"
      value={decision.effective}
      onChange={(e) => onChange(e.target.value)}
      style={{ fontSize: '12px', padding: '4px 8px' }}
    >
      {options.map((o) => (
        <option key={o} value={o}>{o}</option>
      ))}
    </select>
  </KnobRow>
);
