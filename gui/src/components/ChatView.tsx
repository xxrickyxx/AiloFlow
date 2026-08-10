import React, { useEffect, useRef, useState } from 'react';
import { Bot, RefreshCw, Send, Sliders, Square, Trash2, User } from 'lucide-react';
import { api, streamChat } from '../api';
import { useAsync } from '../hooks';
import { EmptyState, ErrorBox } from './Common';
import { useI18n } from '../i18n';
import { formatDuration, hasValue } from '../format';

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  /** Chain-of-thought from reasoning models, shown separately from the answer. */
  reasoning?: string;
  timestamp: string;
  metrics?: {
    tokensPerSecond: number | null;
    completionTokens: number | null;
    firstTokenLatencyMs: number | null;
  };
}

export const ChatView: React.FC = () => {
  const { t } = useI18n();
  const backends = useAsync(() => api.backends(), []);

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const logRef = useRef<HTMLDivElement | null>(null);

  const [systemPrompt, setSystemPrompt] = useState('');
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(0.9);
  const [topK, setTopK] = useState(40);
  const [repeatPenalty, setRepeatPenalty] = useState(1.1);
  // Unlimited by default: a reasoning model spends hundreds of tokens thinking,
  // and a low cap makes it stop before it ever writes the answer.
  const [unlimitedTokens, setUnlimitedTokens] = useState(true);
  const [maxTokens, setMaxTokens] = useState(4096);
  const [seed, setSeed] = useState('');
  const [stopSequences, setStopSequences] = useState('');

  const loaded = backends.data?.loaded ?? null;

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages]);

  const send = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || generating || !loaded) return;

    const userMessage: ChatMessage = {
      id: `u-${Date.now()}`,
      role: 'user',
      content: input,
      timestamp: new Date().toLocaleTimeString(),
    };
    const assistantId = `a-${Date.now()}`;

    // Send the conversation as it stands *before* the placeholder is added.
    const history = [...messages, userMessage].map((m) => ({ role: m.role, content: m.content }));

    setMessages((prev) => [
      ...prev,
      userMessage,
      { id: assistantId, role: 'assistant', content: '', timestamp: new Date().toLocaleTimeString() },
    ]);
    setInput('');
    setGenerating(true);
    setError(null);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      await streamChat(
        {
          model: loaded.model.id,
          messages: systemPrompt.trim()
            ? [{ role: 'system', content: systemPrompt }, ...history]
            : history,
          temperature,
          top_p: topP,
          top_k: topK,
          repeat_penalty: repeatPenalty,
          // Omitting the field lets the engine run until it stops on its own or
          // fills the context, which is what "no limit" has to mean.
          max_tokens: unlimitedTokens ? undefined : maxTokens,
          seed: seed.trim() ? Number(seed) : undefined,
          stop: stopSequences.trim() ? stopSequences.split(',').map((s) => s.trim()).filter(Boolean) : undefined,
        },
        {
          onToken: (text) =>
            setMessages((prev) =>
              prev.map((m) => (m.id === assistantId ? { ...m, content: m.content + text } : m))
            ),
          onReasoning: (text) =>
            setMessages((prev) =>
              prev.map((m) => (m.id === assistantId ? { ...m, reasoning: (m.reasoning || '') + text } : m))
            ),
          onMetrics: (metrics) =>
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? {
                      ...m,
                      metrics: {
                        tokensPerSecond: metrics.tokensPerSecond,
                        completionTokens: metrics.completionTokens,
                        firstTokenLatencyMs: metrics.firstTokenLatencyMs,
                      },
                    }
                  : m
              )
            ),
          onError: (message) => setError(message),
        },
        controller.signal
      );
    } catch (err) {
      if ((err as Error).name !== 'AbortError') setError((err as Error).message);
    } finally {
      setGenerating(false);
      abortRef.current = null;
    }
  };

  const stop = () => {
    abortRef.current?.abort();
    setGenerating(false);
  };

  if (!backends.loading && !loaded) {
    return (
      <div style={{ padding: '0 24px' }}>
        <EmptyState
          title={t('chat.noModelTitle')}
          description={`${t('chat.noModelBody')} ${
            backends.data?.backends.filter((b) => !b.available).map((b) => b.reason).join(' ') || ''
          }`}
        />
      </div>
    );
  }

  return (
    <div style={{ padding: '0 24px', display: 'grid', gridTemplateColumns: '1fr 320px', gap: '20px', height: 'calc(100vh - 150px)' }}>
      <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <div style={{ padding: '14px 20px', borderBottom: '1px solid var(--bg-card-border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', minWidth: 0 }}>
            <Bot size={19} color="var(--accent-cyan)" />
            <span style={{ fontWeight: 700, fontSize: '14px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {loaded?.model.displayName}
            </span>
            <span className="badge-tag">{loaded?.backendId}</span>
          </div>
          <button className="btn-secondary" style={{ padding: '6px 12px' }} onClick={() => setMessages([])} disabled={generating}>
            <Trash2 size={14} /> {t('chat.clear')}
          </button>
        </div>

        <div ref={logRef} style={{ flex: 1, padding: '20px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '14px' }}>
          {messages.length === 0 && (
            <p style={{ fontSize: '13px', color: 'var(--text-muted)', textAlign: 'center', marginTop: '30px' }}>
              {t('chat.intro')}
            </p>
          )}

          {messages.map((msg) => (
            <div
              key={msg.id}
              style={{ display: 'flex', gap: '11px', alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start', maxWidth: '88%' }}
            >
              {msg.role === 'assistant' && (
                <div style={{ width: '32px', height: '32px', borderRadius: '10px', background: 'var(--gradient-brand)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                  <Bot size={17} color="#fff" />
                </div>
              )}

              <div
                style={{
                  background: msg.role === 'user' ? 'rgba(0,242,254,0.13)' : 'rgba(255,255,255,0.05)',
                  border: `1px solid ${msg.role === 'user' ? 'rgba(0,242,254,0.3)' : 'var(--bg-card-border)'}`,
                  padding: '12px 16px',
                  borderRadius: '15px',
                  fontSize: '14px',
                  lineHeight: 1.65,
                }}
              >
                {msg.reasoning && (
                  <details style={{ marginBottom: '9px' }}>
                    <summary
                      style={{
                        cursor: 'pointer',
                        fontSize: '11px',
                        fontWeight: 700,
                        color: 'var(--accent-purple)',
                        userSelect: 'none',
                      }}
                    >
                      Ragionamento del modello ({msg.reasoning.length} caratteri)
                    </summary>
                    <p
                      style={{
                        whiteSpace: 'pre-wrap',
                        fontSize: '12px',
                        color: 'var(--text-muted)',
                        marginTop: '6px',
                        paddingLeft: '10px',
                        borderLeft: '2px solid rgba(127,0,255,0.35)',
                        maxHeight: '260px',
                        overflowY: 'auto',
                      }}
                    >
                      {msg.reasoning}
                    </p>
                  </details>
                )}

                <p style={{ whiteSpace: 'pre-wrap' }}>
                  {msg.content ||
                    (generating && msg.role === 'assistant'
                      ? msg.reasoning
                        ? 'sta ragionando...'
                        : '▋'
                      : '')}
                </p>
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: '14px', marginTop: '7px', fontSize: '11px', color: 'var(--text-muted)' }}>
                  <span>{msg.timestamp}</span>
                  {msg.metrics && (
                    <span style={{ color: 'var(--accent-cyan)', fontWeight: 600, fontFamily: 'var(--font-mono)' }}>
                      {hasValue(msg.metrics.tokensPerSecond) ? `${msg.metrics.tokensPerSecond} tok/s` : 'tok/s n/d'}
                      {hasValue(msg.metrics.completionTokens) ? ` · ${msg.metrics.completionTokens} token` : ''}
                      {hasValue(msg.metrics.firstTokenLatencyMs)
                        ? ` · TTFT ${formatDuration(msg.metrics.firstTokenLatencyMs)}`
                        : ''}
                    </span>
                  )}
                </div>
              </div>

              {msg.role === 'user' && (
                <div style={{ width: '32px', height: '32px', borderRadius: '10px', background: 'rgba(255,255,255,0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                  <User size={17} color="var(--text-primary)" />
                </div>
              )}
            </div>
          ))}

          {error && <ErrorBox message={error} />}
        </div>

        <form onSubmit={send} style={{ padding: '14px 20px', borderTop: '1px solid var(--bg-card-border)', display: 'flex', gap: '10px' }}>
          <input
            className="glass-input"
            style={{ flex: 1 }}
            placeholder={t('chat.placeholder')}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={generating}
          />
          {generating ? (
            <button type="button" className="btn-secondary" onClick={stop}>
              <Square size={15} /> {t('chat.stop')}
            </button>
          ) : (
            <button type="submit" className="btn-primary" disabled={!input.trim()}>
              <Send size={15} /> {t('chat.send')}
            </button>
          )}
        </form>
      </div>

      {/* Hyperparameters — every one of these is forwarded to the engine */}
      <div className="glass-card" style={{ padding: '18px', display: 'flex', flexDirection: 'column', gap: '16px', overflowY: 'auto' }}>
        <h3 style={{ fontSize: '14px', fontWeight: 800, display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Sliders size={17} color="var(--accent-cyan)" /> {t('chat.parameters')}
        </h3>

        <div>
          <label style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text-secondary)' }}>{t('chat.systemPrompt')}</label>
          <textarea
            className="glass-input"
            rows={3}
            style={{ width: '100%', marginTop: '5px', fontSize: '12px', resize: 'vertical' }}
            placeholder={t('chat.systemNone')}
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
          />
        </div>

        <Slider label={t('chat.temperature')} value={temperature} min={0} max={2} step={0.05} onChange={setTemperature} />
        <Slider label={t('chat.topP')} value={topP} min={0.05} max={1} step={0.05} onChange={setTopP} />
        <Slider label={t('chat.topK')} value={topK} min={1} max={100} step={1} onChange={setTopK} />
        <Slider label={t('chat.repeatPenalty')} value={repeatPenalty} min={1} max={2} step={0.05} onChange={setRepeatPenalty} />
        <label style={{ display: 'flex', alignItems: 'center', gap: '9px', fontSize: '12px', cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={unlimitedTokens}
            onChange={(e) => setUnlimitedTokens(e.target.checked)}
            style={{ accentColor: 'var(--accent-cyan)' }}
          />
          <span style={{ color: 'var(--text-secondary)' }}>{t('chat.unlimitedTokens')}</span>
        </label>
        {!unlimitedTokens && (
          <Slider label={t('chat.maxTokens')} value={maxTokens} min={128} max={32768} step={128} onChange={setMaxTokens} />
        )}

        <div>
          <label style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text-secondary)' }}>{t('chat.seed')}</label>
          <input
            className="glass-input"
            style={{ width: '100%', marginTop: '5px', fontSize: '12px' }}
            value={seed}
            onChange={(e) => setSeed(e.target.value.replace(/[^0-9-]/g, ''))}
            placeholder="casuale"
          />
        </div>

        <div>
          <label style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text-secondary)' }}>{t('chat.stopSequences')}</label>
          <input
            className="glass-input"
            style={{ width: '100%', marginTop: '5px', fontSize: '12px' }}
            value={stopSequences}
            onChange={(e) => setStopSequences(e.target.value)}
            placeholder="es. </s>, Utente:"
          />
        </div>

        {backends.loading && <RefreshCw size={14} className="spin" />}
      </div>
    </div>
  );
};

const Slider: React.FC<{
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}> = ({ label, value, min, max, step, onChange }) => (
  <div>
    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px' }}>
      <span style={{ color: 'var(--text-secondary)' }}>{label}</span>
      <span style={{ fontWeight: 700, color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)' }}>{value}</span>
    </div>
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(Number(e.target.value))}
      style={{ width: '100%', marginTop: '5px', accentColor: 'var(--accent-cyan)' }}
    />
  </div>
);
