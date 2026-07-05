# Mimo LLM 配置说明和核对手册

更新时间：2026-07-05

## 1. 本次目标

把 DSA 的本地大模型通道切换到 Mimo 的中国区 OpenAI-compatible 接口：

```env
LLM_CHANNELS=mimo
LLM_MIMO_PROTOCOL=openai
LLM_MIMO_BASE_URL=https://token-plan-cn.xiaomimimo.com/v1
LLM_MIMO_API_KEY=<你的 Mimo Token>
LLM_MIMO_MODELS=mimo-v2.5-pro
LITELLM_MODEL=openai/mimo-v2.5-pro
```

注意：真实 token 只能写在本地 `.env` 或系统密钥管理中，不要写入 Git、文档、截图、聊天记录或日志。

## 2. 当前本机应采用的配置

| 配置项 | 当前目标值 | 作用 |
| --- | --- | --- |
| `LLM_CHANNELS` | `mimo` | 启用渠道模式，只让 Mimo 渠道参与本次 LLM 路由。 |
| `LLM_MIMO_PROTOCOL` | `openai` | 告诉 DSA/LiteLLM 这个接口按 OpenAI 兼容格式调用。 |
| `LLM_MIMO_BASE_URL` | `https://token-plan-cn.xiaomimimo.com/v1` | 中国区 Mimo API base URL。 |
| `LLM_MIMO_API_KEY` | 本地 token | 鉴权密钥。手册中必须用占位符，不记录真实值。 |
| `LLM_MIMO_MODELS` | `mimo-v2.5-pro` | 这个渠道下可用的模型短名。 |
| `LITELLM_MODEL` | `openai/mimo-v2.5-pro` | DSA 默认主模型，也是普通分析和 Agent 继承的主模型。 |
| `AGENT_LITELLM_MODEL` | 留空即可 | 留空时问股 Agent 继承 `LITELLM_MODEL`。 |

## 3. DSA 里哪些功能会用这套 LLM

这套配置不是只给某一个页面用，而是 DSA 后端统一配置：

| 功能 | 是否使用当前 LLM | 说明 |
| --- | --- | --- |
| 股票分析报告 | 是 | 后端分析链路会读取 `LITELLM_MODEL` 和渠道模型列表。 |
| 问股 Agent | 是 | `AGENT_LITELLM_MODEL` 留空时继承 `LITELLM_MODEL`。 |
| 多轮工具调用 | 是 | 由 Agent 的 LiteLLM adapter 调用模型并处理工具调用。 |
| 图片识别/视觉模型 | 不一定 | 视觉模型如果配置了 `VISION_MODEL`，会走视觉模型配置；否则按现有逻辑降级。 |
| Qlib 训练/回测 | 否 | Qlib 训练主要是本地量化模型，不依赖 LLM。 |
| MLflow | 否 | MLflow 只是实验记录页面，不负责模型调用。 |

## 4. 配置优先级

DSA 的 LLM 配置优先级可以简单理解为：

1. `LITELLM_CONFIG`：如果配置了 LiteLLM YAML，优先级最高。
2. `LLM_CHANNELS`：本次采用的方式。每个渠道自己携带 base URL、key、模型列表。
3. legacy provider keys：例如 `OPENAI_API_KEY`、`DEEPSEEK_API_KEY`、`ANTHROPIC_API_KEY`。启用 `LLM_CHANNELS` 后，这些外层 key 不参与 Mimo 渠道路由。

本次目标是使用第二种：`LLM_CHANNELS=mimo`。

## 5. 本地核对步骤

在项目根目录 `/Users/cq/WorkingProjects/daily_stock_analysis` 下核对：

```bash
awk -F= '/^(LLM_CHANNELS|LLM_MIMO_PROTOCOL|LLM_MIMO_BASE_URL|LLM_MIMO_API_KEY|LLM_MIMO_MODELS|LITELLM_MODEL|AGENT_LITELLM_MODEL)=/ {
  v=$2
  if ($1 ~ /KEY/) {
    if (length(v)>10) v=substr(v,1,4) "..." substr(v,length(v)-3)
    else v="***"
  }
  print $1 "=" v
}' .env
```

预期输出应类似：

```text
LLM_CHANNELS=mimo
LLM_MIMO_PROTOCOL=openai
LLM_MIMO_BASE_URL=https://token-plan-cn.xiaomimimo.com/v1
LLM_MIMO_API_KEY=***
LLM_MIMO_MODELS=mimo-v2.5-pro
LITELLM_MODEL=openai/mimo-v2.5-pro
```

不要使用 `cat .env` 直接展示完整配置，因为会把 token 打出来。

## 6. Web 端核对步骤

1. 打开 DSA 本地网页，例如当前开发环境常用地址：`http://127.0.0.1:8010/`。
2. 进入系统设置或 LLM 配置页面。
3. 检查启用渠道是否为 `mimo`。
4. 检查 Mimo base URL 是否为 `https://token-plan-cn.xiaomimimo.com/v1`。
5. 检查主模型是否为 `openai/mimo-v2.5-pro`。
6. 如果有“测试连接”按钮，先用一个短问题测试，例如：“请只回复 ok”。
7. 再进入问股/股票分析页面，做一次轻量分析请求，确认后端日志不再出现 `Invalid API Key`。

## 7. 后端日志核对步骤

查看当天 API 日志时，只查模型和错误关键字，不要查完整 token：

```bash
rg -n "Agent LLM|mimo|Invalid API Key|All LLM models failed|openai/mimo" logs
```

正常现象：

- 可以看到模型名 `openai/mimo-v2.5-pro`。
- 不应继续出现 `Invalid API Key`。

如果 DSA 服务是在修改 `.env` 之前启动的，需要重启后端服务让环境变量生效。

## 8. 常见问题排查

| 问题 | 常见原因 | 处理方式 |
| --- | --- | --- |
| 模型配置没有变化 | 后端没有重启，或 `.env` 被 Docker/外部环境覆盖 | 重启 DSA 后端；检查 `docker compose environment` 或 shell export。 |
| `Invalid API Key` | token 错误、token 未开通、base URL 未按本手册设置 | 重新粘贴 token；确认 base URL 是中国区 `/v1`；确认服务已重启。 |
| `model not found` | `LLM_MIMO_MODELS` 与 `LITELLM_MODEL` 不一致 | 保持 `LLM_MIMO_MODELS=mimo-v2.5-pro`，`LITELLM_MODEL=openai/mimo-v2.5-pro`。 |
| 测试连接成功但回复为空 | 推理模型可能把过小的 `max_tokens` 用在 reasoning 上 | 测试时把 `max_tokens` 提到 128 或以上，再判断文本是否返回。 |
| 连接超时 | 网络、代理或服务端不可达 | 先浏览器/终端检查网络；必要时关闭或调整代理。 |
| Web 设置页保存后又变了 | 运行环境有外部变量覆盖 `.env` | 检查启动脚本、Docker compose、launchd、shell profile。 |
| Agent 不跟随主模型 | `AGENT_LITELLM_MODEL` 被单独设置 | 清空或改成同一个模型：`AGENT_LITELLM_MODEL=openai/mimo-v2.5-pro`。 |

## 9. 恢复步骤

如果当前接口不可用，本文只保留恢复方法，不保留其他 Mimo 配置示例。建议使用以下任一方式恢复：

1. 从你自己的 `.env` 备份恢复 LLM 相关字段。
2. 在 Web 设置页切换到另一个已验证渠道。
3. 手动改为其他可用 provider，并保证 `LLM_CHANNELS`、渠道模型列表和 `LITELLM_MODEL` 三者一致。

恢复后同样需要重启 DSA 后端。

## 10. 安全边界

- `.env` 必须保持在 Git ignore 中。
- 文档、测试、截图、Issue、提交信息里只允许出现 `<你的 Mimo Token>` 这类占位符。
- 日志排查时只打印 token 前后几位。
- 如果 token 已经暴露到公共位置，应在服务商后台立即作废并重新签发。
- Web 端永久保存配置前，先确认管理员鉴权已开启，避免本机服务被其他进程误改配置。

## 11. 最小验收清单

- [ ] `.env` 中 `LLM_MIMO_PROTOCOL=openai`。
- [ ] `.env` 中 `LLM_MIMO_BASE_URL=https://token-plan-cn.xiaomimimo.com/v1`。
- [ ] `.env` 中 `LITELLM_MODEL=openai/mimo-v2.5-pro`。
- [ ] token 只保存在 `.env`，没有写入文档。
- [ ] DSA 后端已重启。
- [ ] Web 设置页显示 Mimo 渠道和正确主模型。
- [ ] 测试连接成功。
- [ ] 问股 Agent 不再报 `Invalid API Key`。
- [ ] 股票分析报告能正常生成。
