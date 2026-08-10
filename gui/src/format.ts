/**
 * Formatting helpers.
 *
 * The single rule here: a null measurement renders as "n/d", never as 0 or a
 * plausible placeholder. If the UI shows a number, that number was measured.
 */

export const NOT_AVAILABLE = 'n/d';

export function formatBytes(bytes: number | null | undefined, decimals = 1): string {
  if (bytes === null || bytes === undefined || Number.isNaN(bytes)) return NOT_AVAILABLE;
  if (bytes === 0) return '0 B';

  const units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB'];
  const i = Math.min(units.length - 1, Math.floor(Math.log(Math.abs(bytes)) / Math.log(1024)));
  return `${(bytes / Math.pow(1024, i)).toFixed(i === 0 ? 0 : decimals)} ${units[i]}`;
}

export function formatPercent(value: number | null | undefined, decimals = 0): string {
  if (value === null || value === undefined || Number.isNaN(value)) return NOT_AVAILABLE;
  return `${value.toFixed(decimals)}%`;
}

export function formatNumber(value: number | null | undefined, suffix = '', decimals = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return NOT_AVAILABLE;
  return `${value.toFixed(decimals)}${suffix}`;
}

export function formatBandwidth(mbps: number | null | undefined): string {
  if (mbps === null || mbps === undefined || Number.isNaN(mbps)) return NOT_AVAILABLE;
  return mbps >= 1024 ? `${(mbps / 1024).toFixed(2)} GB/s` : `${mbps.toFixed(0)} MB/s`;
}

export function formatDuration(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || Number.isNaN(ms)) return NOT_AVAILABLE;
  if (ms < 1000) return `${ms.toFixed(0)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

export function formatRelativeTime(iso: string | null | undefined): string {
  if (!iso) return NOT_AVAILABLE;
  const delta = Date.now() - new Date(iso).getTime();
  if (delta < 60_000) return `${Math.max(0, Math.round(delta / 1000))} s fa`;
  if (delta < 3_600_000) return `${Math.round(delta / 60_000)} min fa`;
  if (delta < 86_400_000) return `${Math.round(delta / 3_600_000)} h fa`;
  return new Date(iso).toLocaleDateString();
}

/** true when a value can be drawn as a filled gauge. */
export function hasValue(value: number | null | undefined): value is number {
  return value !== null && value !== undefined && !Number.isNaN(value);
}
