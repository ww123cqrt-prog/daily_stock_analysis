import camelcaseKeys, { type Options as CamelCaseOptions } from 'camelcase-keys';

type ToCamelCaseOptions = Pick<CamelCaseOptions, 'exclude'>;

/**
 * 将 snake_case 对象键转换为 camelCase
 * @param data API 响应数据 (snake_case)
 * @returns 转换后的 camelCase 对象
 */
export function toCamelCase<T>(data: unknown, options: ToCamelCaseOptions = {}): T {
    if (data === null || data === undefined) {
        return data as T;
    }
    return camelcaseKeys(data as Record<string, unknown>, { deep: true, ...options }) as T;
}
