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
