import { describe, expect, it } from 'vitest';
import { buildMlflowRunUiUrl, resolveMlflowUiUrl } from '../mlflowUiUrl';

describe('resolveMlflowUiUrl', () => {
  it('routes local MLflow through the DSA bilingual page', () => {
    expect(resolveMlflowUiUrl('http://127.0.0.1:6006/')).toBe('/experiments/');
    expect(resolveMlflowUiUrl('http://localhost:6006/')).toBe('/experiments/');
    expect(resolveMlflowUiUrl('http://[::1]:6006/')).toBe('/experiments/');
  });

  it('preserves an explicitly configured remote MLflow URL', () => {
    expect(resolveMlflowUiUrl('https://mlflow.example.com/')).toBe('https://mlflow.example.com/');
  });

  it('builds a run-level deep link for the local proxy', () => {
    expect(buildMlflowRunUiUrl('http://127.0.0.1:6006/', '42', 'abc123'))
      .toBe('/experiments/#/experiments/42/runs/abc123');
  });

  it('leaves the experiment landing page when a run identity is incomplete', () => {
    expect(buildMlflowRunUiUrl('http://127.0.0.1:6006/', '42', null)).toBe('/experiments/');
  });
});
