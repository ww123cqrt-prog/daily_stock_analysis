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

  it('preserves dynamic quant job type keys when loading status', async () => {
    get.mockResolvedValueOnce({
      data: {
        available_job_types: {
          check_qlib_data: {
            label: '检查 Qlib 数据',
            allowed_params: [],
          },
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
        mlflow: {
          url: 'http://127.0.0.1:6006/',
          status: 'linked',
          role: 'experiment_raw_records',
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
            log_path: '/tmp/job.log',
            artifact_paths: [],
          },
        },
        recent_jobs: [],
      },
    });

    const status = await quantApi.getStatus();

    expect(status.availableJobTypes.check_qlib_data.label).toBe('检查 Qlib 数据');
    expect(status.latestJobs.check_qlib_data.jobId).toBe('qj_1');
    expect(status.latestJobs.checkQlibData).toBeUndefined();
    expect(status.trainable).toBe(true);
    expect(status.trainingReadiness?.status).toBe('trainable');
    expect(status.dataSources[0].dateRange).toBe('2008-01-02 ~ 2026-07-03');
    expect(status.qlibProvider?.calendarDays).toBe(4492);
    expect(status.latestTraining?.runId).toBe('run_20260705');
    expect(status.qlibPredictions?.latestDate).toBe('2026-07-02');
    expect(status.qlibPredictions?.top?.[0].instrument).toBe('SZ000002');
    expect(status.mlflow?.url).toBe('http://127.0.0.1:6006/');
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
  });

  it('builds artifact URLs without treating artifact paths as user input', () => {
    expect(quantApi.getJobArtifactUrl('qj_1/unsafe', 2)).toBe('/api/v1/quant/jobs/qj_1%2Funsafe/artifacts/2');
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
