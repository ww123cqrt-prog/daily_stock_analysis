const LOCAL_MLFLOW_HOSTS = new Set(['127.0.0.1', 'localhost', '::1']);

export function resolveMlflowUiUrl(rawUrl: string): string {
  try {
    const url = new URL(rawUrl);
    const hostname = url.hostname.replace(/^\[|\]$/g, '');
    if (
      url.protocol === 'http:'
      && LOCAL_MLFLOW_HOSTS.has(hostname)
      && url.port === '6006'
      && (url.pathname === '' || url.pathname === '/')
      && !url.username
      && !url.password
    ) {
      return '/experiments/';
    }
  } catch {
    return rawUrl;
  }
  return rawUrl;
}

export function buildMlflowRunUiUrl(
  rawUrl: string,
  experimentId?: string | null,
  runId?: string | null,
): string {
  const baseUrl = resolveMlflowUiUrl(rawUrl);
  if (!experimentId || !runId) return baseUrl;

  const route = `#/experiments/${encodeURIComponent(experimentId)}/runs/${encodeURIComponent(runId)}`;
  if (baseUrl.startsWith('/')) {
    return `${baseUrl.replace(/\/?$/, '/')}${route}`;
  }
  try {
    const url = new URL(baseUrl);
    url.hash = route.slice(1);
    return url.toString();
  } catch {
    return `${baseUrl.replace(/\/?$/, '/')}${route}`;
  }
}
