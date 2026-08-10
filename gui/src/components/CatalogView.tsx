import React, { useState } from 'react';
import { Cloud, Download, Gauge, HardDrive, Search } from 'lucide-react';
import { api, CatalogEntry, EstimateResponse, HfFile, HfModelSummary, SizeClass } from '../api';
import { useAsync } from '../hooks';
import { DownloadsState, startDownload } from '../downloads';
import { ErrorBox, Spinner } from './Common';
import { DownloadsPanel } from './DownloadsPanel';
import { TranslationKey, useI18n } from '../i18n';
import { formatBytes, formatNumber } from '../format';

interface Props {
  downloads: DownloadsState;
}

export const CatalogView: React.FC<Props> = ({ downloads }) => {
  const { t } = useI18n();
  const catalog = useAsync(() => api.catalog(), []);

  const [selected, setSelected] = useState<CatalogEntry | null>(null);
  const [quantization, setQuantization] = useState('Q4_K_M');
  const [estimate, setEstimate] = useState<EstimateResponse | null>(null);
  const [estimating, setEstimating] = useState(false);

  const [query, setQuery] = useState('');
  const [results, setResults] = useState<HfModelSummary[] | null>(null);
  const [searching, setSearching] = useState(false);

  const [openRepo, setOpenRepo] = useState<string | null>(null);
  const [files, setFiles] = useState<HfFile[] | null>(null);
  const [loadingFiles, setLoadingFiles] = useState(false);

  const [error, setError] = useState<string | null>(null);

  // With ~40 entries a flat list is unusable, so the catalogue is filtered by
  // size class and free text before it is rendered.
  const [sizeFilter, setSizeFilter] = useState<SizeClass | 'tutti'>('tutti');
  const [catalogFilter, setCatalogFilter] = useState('');

  const selectEntry = async (entry: CatalogEntry) => {
    setSelected(entry);
    setEstimate(null);
    setEstimating(true);
    setError(null);
    try {
      setEstimate(await api.estimate({ catalogId: entry.id, quantization, contextUsed: 8192 }));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setEstimating(false);
    }
  };

  const search = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    setSearching(true);
    setError(null);
    try {
      setResults((await api.searchRepos(query)).results);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSearching(false);
    }
  };

  const openFiles = async (repoId: string) => {
    setOpenRepo(repoId);
    setFiles(null);
    setLoadingFiles(true);
    setError(null);
    try {
      setFiles((await api.repoFiles(repoId)).files);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoadingFiles(false);
    }
  };

  /**
   * Hand the transfer to the runtime and return immediately. Progress is then
   * read from the shared download registry, so leaving this screen — or
   * reloading the page entirely — does not interrupt it.
   */
  const beginDownload = async (repoId: string, file: HfFile) => {
    setError(null);
    try {
      await startDownload(repoId, file);
      downloads.refresh();
    } catch (err) {
      setError((err as Error).message);
    }
  };

  const allEntries = catalog.data?.entries || [];
  const needle = catalogFilter.trim().toLowerCase();
  const entries = allEntries.filter((entry) => {
    if (sizeFilter !== 'tutti' && entry.sizeClass !== sizeFilter) return false;
    if (!needle) return true;
    return (
      entry.name.toLowerCase().includes(needle) ||
      entry.publisher.toLowerCase().includes(needle) ||
      entry.tags.some((tag) => tag.includes(needle))
    );
  });

  const countFor = (size: SizeClass | 'tutti') =>
    size === 'tutti' ? allEntries.length : allEntries.filter((e) => e.sizeClass === size).length;

  return (
    <div style={{ padding: '0 24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
      {error && <ErrorBox message={error} />}
      {catalog.error && <ErrorBox message={catalog.error} />}

      {/* Destination */}
      <div className="glass-card" style={{ padding: '16px 20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '16px', flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '11px' }}>
          <HardDrive size={18} color="var(--accent-amber)" />
          <div>
            <div style={{ fontSize: '13px', fontWeight: 700 }}>{catalog.data?.downloadDirectory || '—'}</div>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
              {t('catalog.directory', { free: formatBytes(catalog.data?.freeBytes ?? null) })}
            </div>
          </div>
        </div>
        <span style={{ fontSize: '11px', color: 'var(--text-muted)', maxWidth: '440px' }}>
          {t('catalog.spaceHint')}
        </span>
      </div>

      {/* Transfers owned by the runtime, mirrored here for convenience */}
      {downloads.jobs.length > 0 && (
        <div className="glass-card" style={{ padding: '18px 20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '11px' }}>
            <h3 style={{ fontSize: '14px', fontWeight: 800 }}>{t('dl.active')}</h3>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{t('dl.keepsRunning')}</span>
          </div>
          <DownloadsPanel jobs={downloads.jobs} onChanged={downloads.refresh} />
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(340px, 460px) 1fr', gap: '18px', alignItems: 'start' }}>
        {/* Curated catalog */}
        <div className="glass-card" style={{ padding: '18px' }}>
          <h3 style={{ fontSize: '15px', fontWeight: 800, marginBottom: '10px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Cloud size={17} color="var(--accent-cyan)" /> {t('catalog.title')}
            <span className="badge-tag" style={{ marginLeft: 'auto' }}>{allEntries.length}</span>
          </h3>

          <div style={{ display: 'flex', gap: '5px', marginBottom: '9px', flexWrap: 'wrap' }}>
            {(['tutti', 'gigante', 'grande', 'medio', 'piccolo'] as const).map((size) => {
              const active = sizeFilter === size;
              return (
                <button
                  key={size}
                  onClick={() => setSizeFilter(size)}
                  style={{
                    background: active ? 'rgba(0,242,254,0.15)' : 'rgba(255,255,255,0.05)',
                    border: `1px solid ${active ? 'var(--accent-cyan)' : 'var(--bg-card-border)'}`,
                    color: active ? 'var(--accent-cyan)' : 'var(--text-secondary)',
                    borderRadius: 'var(--radius-sm)',
                    padding: '4px 9px',
                    fontSize: '11px',
                    fontWeight: 600,
                    cursor: 'pointer',
                    fontFamily: 'var(--font-main)',
                  }}
                >
                  {t(`catalog.${size}` as TranslationKey)} ({countFor(size)})
                </button>
              );
            })}
          </div>

          <input
            className="glass-input"
            style={{ width: '100%', fontSize: '12px', marginBottom: '10px' }}
            placeholder={t('catalog.filterPlaceholder')}
            value={catalogFilter}
            onChange={(e) => setCatalogFilter(e.target.value)}
          />

          {catalog.loading && <Spinner />}

          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', maxHeight: '440px', overflowY: 'auto' }}>
            {entries.length === 0 && !catalog.loading && (
              <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                {t('catalog.noMatch')}
              </span>
            )}
            {entries.map((entry) => {
              const active = selected?.id === entry.id;
              return (
                <button
                  key={entry.id}
                  onClick={() => selectEntry(entry)}
                  style={{
                    textAlign: 'left',
                    background: active ? 'rgba(0,242,254,0.12)' : 'rgba(255,255,255,0.04)',
                    border: `1px solid ${active ? 'var(--accent-cyan)' : 'var(--bg-card-border)'}`,
                    borderRadius: 'var(--radius-md)',
                    padding: '11px 13px',
                    cursor: 'pointer',
                    color: 'var(--text-primary)',
                    fontFamily: 'var(--font-main)',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px', alignItems: 'baseline' }}>
                    <span style={{ fontSize: '13px', fontWeight: 700 }}>{entry.name}</span>
                    <span style={{ fontSize: '10px', color: 'var(--text-muted)', flexShrink: 0 }}>
                      {t('catalog.inQ4', { size: formatBytes(entry.totalParamsB * 1e9 * 0.604) })}
                    </span>
                  </div>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>{entry.publisher}</div>
                  <div style={{ display: 'flex', gap: '5px', marginTop: '7px', flexWrap: 'wrap' }}>
                    <span className="badge-tag">{t('catalog.totalParams', { n: entry.totalParamsB })}</span>
                    <span
                      className="badge-tag"
                      style={
                        entry.architecture === 'moe'
                          ? { color: 'var(--accent-emerald)', borderColor: 'rgba(16,185,129,0.3)', background: 'rgba(16,185,129,0.1)' }
                          : undefined
                      }
                    >
                      {t('catalog.activeParams', { n: entry.activeParamsB })}
                    </span>
                    {entry.tags.slice(0, 2).map((tag) => (
                      <span key={tag} className="badge-tag" style={{ textTransform: 'none', opacity: 0.75 }}>
                        {tag}
                      </span>
                    ))}
                  </div>
                </button>
              );
            })}
          </div>

          {/* Free search */}
          <form onSubmit={search} style={{ display: 'flex', gap: '8px', marginTop: '14px' }}>
            <input
              className="glass-input"
              style={{ flex: 1, fontSize: '12px' }}
              placeholder={t('catalog.searchPlaceholder')}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
            <button className="btn-secondary" type="submit" disabled={searching}>
              <Search size={15} />
            </button>
          </form>

          {results && (
            <div style={{ marginTop: '10px', display: 'flex', flexDirection: 'column', gap: '6px', maxHeight: '220px', overflowY: 'auto' }}>
              {results.length === 0 && (
                <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{t('catalog.noRepos')}</span>
              )}
              {results.map((r) => (
                <button
                  key={r.repoId}
                  onClick={() => openFiles(r.repoId)}
                  style={{
                    textAlign: 'left',
                    background: 'rgba(255,255,255,0.04)',
                    border: '1px solid var(--bg-card-border)',
                    borderRadius: 'var(--radius-sm)',
                    padding: '8px 10px',
                    cursor: 'pointer',
                    color: 'var(--text-primary)',
                    fontSize: '12px',
                    fontFamily: 'var(--font-main)',
                  }}
                >
                  <div style={{ fontWeight: 600, wordBreak: 'break-all' }}>{r.repoId}</div>
                  <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>
                    {t('catalog.downloads', { count: r.downloads.toLocaleString() })}
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Detail: estimate + files */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {selected && (
            <div className="glass-card" style={{ padding: '22px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: '14px', flexWrap: 'wrap' }}>
                <div>
                  <h2 style={{ fontSize: '18px', fontWeight: 800 }}>{selected.name}</h2>
                  <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '4px', maxWidth: '620px', lineHeight: 1.6 }}>
                    {selected.notes}
                  </p>
                  <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '6px' }}>
                    {t('catalog.license', { license: selected.license, context: selected.contextLength.toLocaleString() })}
                  </p>
                </div>

                <div style={{ display: 'flex', gap: '8px', alignItems: 'flex-start' }}>
                  <select
                    className="glass-input"
                    style={{ fontSize: '12px' }}
                    value={quantization}
                    onChange={(e) => {
                      setQuantization(e.target.value);
                      setEstimate(null);
                    }}
                  >
                    {['Q8_0', 'Q6_K', 'Q5_K_M', 'Q4_K_M', 'IQ4_XS', 'Q3_K_M', 'IQ2_XXS'].map((q) => (
                      <option key={q} value={q}>{q}</option>
                    ))}
                  </select>
                  <button className="btn-primary" onClick={() => selectEntry(selected)} disabled={estimating}>
                    <Gauge size={15} /> {t('catalog.estimate')}
                  </button>
                </div>
              </div>

              {estimating && <Spinner label={t('catalog.computing')} />}
              {estimate && <EstimatePanel data={estimate} />}

              <div style={{ marginTop: '18px', display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                {selected.ggufRepos.map((repo) => (
                  <button key={repo} className="btn-secondary" style={{ fontSize: '12px' }} onClick={() => openFiles(repo)}>
                    <Download size={14} /> {repo}
                  </button>
                ))}
              </div>
            </div>
          )}

          {openRepo && (
            <div className="glass-card" style={{ padding: '20px' }}>
              <h3 style={{ fontSize: '14px', fontWeight: 800, marginBottom: '12px', wordBreak: 'break-all' }}>
                {t('catalog.filesIn', { repo: openRepo })}
              </h3>
              {loadingFiles && <Spinner label={t('catalog.readingRepo')} />}

              <div style={{ display: 'flex', flexDirection: 'column', gap: '7px', maxHeight: '380px', overflowY: 'auto' }}>
                {files?.map((file) => (
                  <div
                    key={file.path}
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      gap: '12px',
                      padding: '9px 12px',
                      background: 'rgba(255,255,255,0.04)',
                      borderRadius: 'var(--radius-md)',
                    }}
                  >
                    <div style={{ minWidth: 0 }}>
                      <div style={{ fontSize: '12px', fontWeight: 600, wordBreak: 'break-all' }}>{file.path}</div>
                      <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                        {formatBytes(file.totalSizeBytes)}
                        {file.splitParts ? ` · ${t('catalog.parts', { n: file.splitParts.length })}` : ''}
                        {file.quantization ? ` · ${file.quantization}` : ''}
                      </div>
                    </div>
                    <button
                      className="btn-primary"
                      style={{ padding: '7px 13px', fontSize: '12px', flexShrink: 0 }}
                      onClick={() => beginDownload(openRepo, file)}
                      disabled={downloads.active.some(
                        (j) => j.repoId === openRepo && j.fileName === file.path
                      )}
                    >
                      <Download size={14} /> {t('catalog.download')}
                    </button>
                  </div>
                ))}
                {files && files.length === 0 && (
                  <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{t('catalog.noGgufFiles')}</span>
                )}
              </div>
            </div>
          )}

          {!selected && !openRepo && (
            <div className="glass-card" style={{ padding: '30px', textAlign: 'center' }}>
              <h3 style={{ fontSize: '15px', fontWeight: 700, color: 'var(--text-secondary)', marginBottom: '8px' }}>
                {t('catalog.emptyTitle')}
              </h3>
              <p style={{ fontSize: '13px', color: 'var(--text-muted)', lineHeight: 1.7, maxWidth: '620px', margin: '0 auto' }}>
                {t('catalog.emptyBody')}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

const EstimatePanel: React.FC<{ data: EstimateResponse }> = ({ data }) => {
  const { t } = useI18n();
  const e = data.estimate;

  return (
    <div style={{ marginTop: '18px' }}>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '10px' }}>
        <Cell label={t('est.totalWeight')} value={formatBytes(e.totalWeightBytes)} />
        <Cell label={t('est.bytesPerToken')} value={formatBytes(e.activeBytesPerToken)} />
        <Cell label={t('est.kvCache')} value={formatBytes(e.kvCacheBytes)} />
        <Cell
          label={t('est.fitsInMemory')}
          value={e.fitsInMemory ? t('est.fitsYes') : t('est.fitsNo')}
          accent={e.fitsInMemory ? 'var(--accent-emerald)' : 'var(--accent-amber)'}
        />
        <Cell
          label={t('est.overlapped')}
          value={e.overlappedTokensPerSecond !== null ? `${e.overlappedTokensPerSecond} tok/s` : t('common.notAvailable')}
          accent="var(--accent-cyan)"
        />
        <Cell label={t('est.bottleneck')} value={e.bottleneckTier || t('common.notAvailable')} accent="var(--accent-amber)" />
      </div>

      <div style={{ marginTop: '14px' }}>
        <div style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text-muted)', marginBottom: '6px' }}>
          {t('est.placement')}
        </div>
        {e.placement.map((p) => (
          <div key={p.tier} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', padding: '4px 0', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
            <span style={{ color: 'var(--text-secondary)' }}>{p.tier}</span>
            <span style={{ fontFamily: 'var(--font-mono)' }}>
              {t('est.resident', { size: formatBytes(p.residentBytes) })} · {t('est.perToken', { size: formatBytes(p.bytesPerToken) })} ·{' '}
              {p.bandwidthMBps ? `${formatNumber(p.bandwidthMBps / 1024, ' GB/s', 1)}` : t('est.noBandwidth')}
            </span>
          </div>
        ))}
      </div>

      <div style={{ marginTop: '14px' }}>
        <div style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text-muted)', marginBottom: '6px' }}>
          {t('est.projections')}
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '8px' }}>
          {data.projections.map((p) => (
            <div key={p.storageBandwidthMBps} style={{ background: 'rgba(255,255,255,0.04)', borderRadius: 'var(--radius-sm)', padding: '9px 11px' }}>
              <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>
                {(p.storageBandwidthMBps / 1024).toFixed(1)} GB/s
              </div>
              <div style={{ fontSize: '15px', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>
                {p.estimate.overlappedTokensPerSecond ?? '—'}
              </div>
              <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>tok/s</div>
            </div>
          ))}
        </div>
      </div>

      <div style={{ marginTop: '14px', fontSize: '11px', color: 'var(--text-muted)', lineHeight: 1.7 }}>
        <div style={{ marginBottom: '4px' }}>
          {t('est.confidence')}: <strong style={{ color: e.confidence === 'measured' ? 'var(--accent-emerald)' : 'var(--accent-amber)' }}>{e.confidence}</strong>
        </div>
        {e.assumptions.map((a, i) => <div key={i}>· {a}</div>)}
        {e.warnings.map((w, i) => (
          <div key={`w${i}`} style={{ color: 'var(--accent-amber)' }}>! {w}</div>
        ))}
      </div>
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
