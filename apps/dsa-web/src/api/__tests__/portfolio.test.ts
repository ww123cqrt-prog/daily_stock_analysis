import { beforeEach, describe, expect, it, vi } from 'vitest';
import { portfolioApi } from '../portfolio';

const { deleteRequest, get, post } = vi.hoisted(() => ({
  deleteRequest: vi.fn(),
  get: vi.fn(),
  post: vi.fn(),
}));

vi.mock('../index', () => ({
  default: {
    delete: deleteRequest,
    get,
    post,
    defaults: { baseURL: '' },
  },
}));

describe('portfolioApi', () => {
  beforeEach(() => {
    deleteRequest.mockReset();
    get.mockReset();
    post.mockReset();
  });

  it('deactivates an account through the portfolio account endpoint', async () => {
    deleteRequest.mockResolvedValueOnce({ data: { deleted: 1 } });

    const result = await portfolioApi.deleteAccount(7);

    expect(result).toEqual({ deleted: 1 });
    expect(deleteRequest).toHaveBeenCalledWith('/api/v1/portfolio/accounts/7');
  });
});
