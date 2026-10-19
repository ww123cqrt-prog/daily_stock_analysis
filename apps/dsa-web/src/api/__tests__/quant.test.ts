import { beforeEach, describe, expect, it, vi } from 'vitest';
import { parseQuantJobStreamEvent, quantApi } from '../quant';

const { get, post } = vi.hoisted(() => ({
  get: vi.fn(),
  post: vi.fn(),
}));

vi.mock('../index', () => ({
  default: {
    get,
    post,
    defaults: { baseURL: '' },
  },
}));

describe('quantApi', () => {
  beforeEach(() => {
    get.mockReset();
    post.mockReset();
  });

  it('exposes the governed quant_data readiness endpoint', () => {
    expect(quantApi).toHaveProperty('getDataStatus');
  });

  it('preserves local snapshot status scope and availability without inventing a full audit', async () => {
    get.mockResolvedValueOnce({ data: {
      backend: 'quant_qmt_data', available_datasets: 7,
      validation_scope: 'snapshot_metadata', status: 'ready', checked_at: '2026-10-20T00:00:00Z',
      datasets: [{ dataset_id: 'cn_etf_daily', snapshot_id: 'etf-one', status: 'ready' }],
    } });
    const status = await quantApi.getDataStatus();
    expect(status.backend).toBe('quant_qmt_data');
    expect(status.availableDatasets).toBe(7);
    expect(status.validationScope).toBe('snapshot_metadata');
    expect(status.validatedDatasets).toBeUndefined();
    expect(status.datasets[0].snapshotId).toBe('etf-one');
  });

  it('preserves dynamic quant job type keys when loading status', async () => {
    get.mockResolvedValueOnce({
      data: {
        available_job_types: {
          check_qlib_data: {
            label: '检查 Qlib 数据',
            allowed_params: [],
          },
        },
        api_contract: {
          version: 'quant-workbench-v2',
          backend_build_id: 'backend-test-123',
          backend_started_at: '2026-08-02T02:00:00Z',
        },
        active_count: 0,
        generated_at: '2026-07-05T15:30:00+08:00',
        trainable: true,
        training_readiness: {
          status: 'trainable',
          summary: 'Qlib provider is ready.',
        },
        data_sources: [
          {
            name: 'merged',
            files: 5340,
            rows: 7580087,
            date_range: '2008-01-02 ~ 2026-07-03',
            status: 'available',
          },
        ],
        qlib_provider: {
          path: '/tmp/qlib',
          calendar_days: 4492,
          calendar_range: '2008-01-02 ~ 2026-07-03',
          instruments: 5340,
        },
        latest_training: {
          status: 'completed',
          experiment_name: 'baseline_lgb_alpha158_cleaned',
          run_id: 'run_20260705',
        },
        qlib_predictions: {
          mode: 'read_only_qlib_prediction_summary',
          latest_date: '2026-07-02',
          top: [{ instrument: 'SZ000002', score: 0.25 }],
          bottom: [{ instrument: 'SZ000001', score: -0.4 }],
          summary: { latest_rows: 3 },
        },
        qlib_research: {
          status: 'available',
          available_sections: 1,
          total_sections: 6,
          sections: {
            model_explainability: {
              section: 'model_explainability',
              title: '模型效果深度解释',
              status: 'available',
              generated_at: '2026-07-05T11:00:00+08:00',
            },
          },
        },
        mlflow: {
          url: 'http://127.0.0.1:6006/',
          status: 'linked',
          role: 'experiment_raw_records',
          checked_at: '2026-07-03T13:10:00Z',
        },
        external_services: {
          dsa: {
            url: '/',
            status: 'available',
            role: 'local_web_api',
            checked_at: '2026-07-03T13:10:00Z',
            health_checked: true,
          },
          catdesk: {
            url: 'http://127.0.0.1:8765/',
            status: 'available',
            role: 'external_catalyst_dashboard',
            checked_at: '2026-07-03T13:10:00Z',
            health_checked: true,
          },
        },
        latest_jobs: {
          check_qlib_data: {
            job_id: 'qj_1',
            job_type: 'check_qlib_data',
            label: '检查 Qlib 数据',
            status: 'completed',
            params: {},
            command: [],
            cwd: '/Users/cq/WorkingProjects',
            created_at: '2026-07-04T01:00:00+08:00',
            updated_at: '2026-07-04T01:00:01+08:00',
            can_cancel: true,
            log_path: '/tmp/job.log',
            artifact_paths: [],
          },
        },
        recent_jobs: [],
      },
    });

    const status = await quantApi.getStatus();

    expect(status.availableJobTypes.check_qlib_data.label).toBe('检查 Qlib 数据');
    expect(status.apiContract?.version).toBe('quant-workbench-v2');
    expect(status.apiContract?.backendBuildId).toBe('backend-test-123');
    expect(status.latestJobs.check_qlib_data.jobId).toBe('qj_1');
    expect(status.latestJobs.check_qlib_data.canCancel).toBe(true);
    expect((status.latestJobs.check_qlib_data as unknown as { command?: unknown }).command).toBeUndefined();
    expect((status.latestJobs.check_qlib_data as unknown as { cwd?: unknown }).cwd).toBeUndefined();
    expect((status.latestJobs.check_qlib_data as unknown as { logPath?: unknown }).logPath).toBeUndefined();
    expect(status.latestJobs.checkQlibData).toBeUndefined();
    expect(status.trainable).toBe(true);
    expect(status.trainingReadiness?.status).toBe('trainable');
    expect(status.dataSources[0].dateRange).toBe('2008-01-02 ~ 2026-07-03');
    expect(status.qlibProvider?.calendarDays).toBe(4492);
    expect(status.latestTraining?.runId).toBe('run_20260705');
    expect(status.qlibPredictions?.latestDate).toBe('2026-07-02');
    expect(status.qlibPredictions?.top?.[0].instrument).toBe('SZ000002');
    expect(status.qlibResearch?.sections.model_explainability.generatedAt).toBe('2026-07-05T11:00:00+08:00');
    expect(status.mlflow?.url).toBe('http://127.0.0.1:6006/');
    expect(status.mlflow?.checkedAt).toBe('2026-07-03T13:10:00Z');
    expect(status.externalServices?.dsa.status).toBe('available');
    expect(status.externalServices?.catdesk.url).toBe('http://127.0.0.1:8765/');
    expect(status.externalServices?.catdesk.healthChecked).toBe(true);
  });

  it('loads the static job catalog together with its backend contract', async () => {
    get.mockResolvedValueOnce({
      data: {
        available_job_types: {
          check_qlib_data: { label: '检查 Qlib 数据', allowed_params: [] },
        },
        api_contract: {
          version: 'quant-workbench-v2',
          backend_build_id: 'backend-test-456',
          backend_started_at: '2026-08-02T03:00:00Z',
        },
      },
    });

    const catalog = await quantApi.getJobTypes();

    expect(catalog.availableJobTypes.check_qlib_data.label).toBe('检查 Qlib 数据');
    expect(catalog.apiContract.backendBuildId).toBe('backend-test-456');
  });

  it('requests lightweight quant status without duplicating research-state loading', async () => {
    get.mockResolvedValueOnce({
      data: {
        available_job_types: {},
        active_count: 0,
        latest_jobs: {},
        recent_jobs: [],
      },
    });

    await quantApi.getStatus({ includeResearch: false });

    expect(get).toHaveBeenCalledWith('/api/v1/quant/status', {
      params: { include_research: false },
    });
  });

  it('uses fixed local-service endpoints for status and startup', async () => {
    get.mockResolvedValueOnce({
      data: {
        service: 'catdesk',
        status: 'unavailable',
        url: 'http://127.0.0.1:8765/',
        open_url: 'http://127.0.0.1:8765/',
      },
    });
    post.mockResolvedValueOnce({
      data: {
        service: 'mlflow',
        status: 'starting',
        open_url: '/experiments/',
      },
    });

    const catdesk = await quantApi.getLocalServiceStatus('catdesk');
    const mlflow = await quantApi.startLocalService('mlflow');

    expect(get).toHaveBeenCalledWith('/api/v1/quant/services/catdesk');
    expect(post).toHaveBeenCalledWith('/api/v1/quant/services/mlflow/start');
    expect(catdesk.openUrl).toBe('http://127.0.0.1:8765/');
    expect(mlflow.status).toBe('starting');
  });

  it('parses streamed job snapshots into frontend job objects', () => {
    const jobs = parseQuantJobStreamEvent(JSON.stringify({
      type: 'jobs',
      jobs: [
        {
          job_id: 'qj_stream',
          job_type: 'check_qlib_data',
          label: '检查 Qlib 数据',
          status: 'completed',
          params: {},
          command: [],
          cwd: '/Users/cq/WorkingProjects',
          created_at: '2026-07-04T01:00:00+08:00',
          updated_at: '2026-07-04T01:00:01+08:00',
          log_path: '/tmp/job.log',
          artifact_paths: ['/tmp/report.md'],
        },
      ],
    }));

    expect(jobs).toHaveLength(1);
    expect(jobs[0].jobId).toBe('qj_stream');
    expect(jobs[0].jobType).toBe('check_qlib_data');
    expect(jobs[0].artifactPaths).toEqual(['/tmp/report.md']);
    expect((jobs[0] as unknown as { command?: unknown }).command).toBeUndefined();
    expect((jobs[0] as unknown as { cwd?: unknown }).cwd).toBeUndefined();
    expect((jobs[0] as unknown as { logPath?: unknown }).logPath).toBeUndefined();
  });

  it('warns when a streamed snapshot is malformed instead of silently hiding it', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);

    expect(parseQuantJobStreamEvent('{not-json')).toEqual([]);
    expect(parseQuantJobStreamEvent(JSON.stringify({ type: 'jobs' }))).toEqual([]);

    expect(warn).toHaveBeenCalledTimes(2);
    expect(warn.mock.calls[0][0]).toContain('quant job stream');
    warn.mockRestore();
  });

  it('builds artifact URLs without treating artifact paths as user input', () => {
    expect(quantApi.getJobArtifactUrl('qj_1/unsafe', 2)).toBe('/api/v1/quant/jobs/qj_1%2Funsafe/artifacts/2');
  });

  it('strips internal command fields from cancelled job responses', async () => {
    post.mockResolvedValueOnce({
      data: {
        job_id: 'qj_cancelled',
        job_type: 'check_qlib_data',
        label: '检查 Qlib 数据',
        status: 'cancelled',
        params: {},
        command: ['python', 'internal.py'],
        cwd: '/Users/cq/WorkingProjects',
        created_at: '2026-07-04T01:00:00+08:00',
        updated_at: '2026-07-04T01:00:01+08:00',
        log_path: '/tmp/job.log',
        artifact_paths: [],
      },
    });

    const job = await quantApi.cancelJob('qj_cancelled');

    expect(post).toHaveBeenCalledWith('/api/v1/quant/jobs/qj_cancelled/cancel');
    expect(job.jobId).toBe('qj_cancelled');
    expect((job as unknown as { command?: unknown }).command).toBeUndefined();
    expect((job as unknown as { cwd?: unknown }).cwd).toBeUndefined();
    expect((job as unknown as { logPath?: unknown }).logPath).toBeUndefined();
  });

  it('loads a text artifact by job id and index for inline preview', async () => {
    get.mockResolvedValueOnce({ data: '# Report\n\ninline artifact' });

    const content = await quantApi.getJobArtifactText('qj_1/unsafe', 2);

    expect(get).toHaveBeenCalledWith('/api/v1/quant/jobs/qj_1%2Funsafe/artifacts/2', {
      responseType: 'text',
    });
    expect(content).toBe('# Report\n\ninline artifact');
  });
});
