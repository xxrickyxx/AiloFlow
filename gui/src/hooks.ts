import { useCallback, useEffect, useRef, useState } from 'react';
import { API_BASE, TelemetrySnapshot } from './api';

export type ConnectionState = 'connecting' | 'online' | 'offline';

/**
 * Subscribe to the server's telemetry stream.
 *
 * While the connection is down `snapshot` stays null so views render an
 * explicit offline state rather than the last known values pretending to be live.
 */
export function useMetricsStream(): { snapshot: TelemetrySnapshot | null; connection: ConnectionState } {
  const [snapshot, setSnapshot] = useState<TelemetrySnapshot | null>(null);
  const [connection, setConnection] = useState<ConnectionState>('connecting');

  useEffect(() => {
    let source: EventSource | null = null;
    let retryHandle: number | undefined;
    let disposed = false;

    const connect = () => {
      if (disposed) return;
      source = new EventSource(`${API_BASE}/metrics/stream`);

      source.onmessage = (event) => {
        try {
          setSnapshot(JSON.parse(event.data));
          setConnection('online');
        } catch {
          // Ignore a malformed frame; the next tick will replace it.
        }
      };

      source.onerror = () => {
        setConnection('offline');
        setSnapshot(null);
        source?.close();
        // The API may simply not be started yet — keep retrying quietly.
        retryHandle = window.setTimeout(connect, 3000);
      };
    };

    connect();

    return () => {
      disposed = true;
      if (retryHandle) window.clearTimeout(retryHandle);
      source?.close();
    };
  }, []);

  return { snapshot, connection };
}

export interface AsyncState<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
  reload: () => void;
}

/** Run an async loader, exposing loading and error states the UI must show. */
export function useAsync<T>(loader: () => Promise<T>, deps: unknown[] = []): AsyncState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [nonce, setNonce] = useState(0);
  const loaderRef = useRef(loader);
  loaderRef.current = loader;

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    loaderRef
      .current()
      .then((result) => {
        if (!cancelled) setData(result);
      })
      .catch((err: Error) => {
        if (!cancelled) {
          setError(err.message);
          setData(null);
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nonce, ...deps]);

  const reload = useCallback(() => setNonce((n) => n + 1), []);
  return { data, error, loading, reload };
}
