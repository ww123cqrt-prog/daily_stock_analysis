---
name: "stock_analyzer"
description: "股票智能分析系统，覆盖 A 股、港股、美股。当用户想要分析单个或多个股票、进行市场复盘、配置交易策略、使用 Agent 模式分析、通过 API/Bot/Web 调用分析、配置通知渠道、或了解系统架构时调用。也适用于用户提到股票代码、大盘行情、自选股、交易策略、量化分析、K 线技术分析等场景。"
---

# 股票智能分析系统

本技能基于 `daily_stock_analysis` 项目，提供股票分析、市场复盘、策略配置和多平台接入能力。

## 系统架构

主流程：**抓取数据 → 技术分析/新闻检索 → LLM 分析 → 生成报告 → 通知推送**

关键模块：

| 模块 | 路径 | 职责 |
|------|------|------|
| 分析服务层 | `src/services/analyzer_service.py` | 封装核心分析逻辑，供 CLI/WebUI/Bot 调用 |
| 分析服务（API） | `src/services/analysis_service.py` | API 层分析服务，含诊断上下文 |
| 分析流水线 | `src/core/pipeline.py` | 并发调度、数据获取、AI 分析、通知推送 |
| AI 分析器 | `src/analyzer.py` | LLM 调用、Prompt 构建、结果解析 |
| 大盘复盘 | `src/core/market_review.py` | 多市场（A/港/美）复盘 |
| Agent 编排器 | `src/agent/orchestrator.py` | 多 Agent 协同分析 |
| 数据源 | `data_provider/base.py` | 多数据源适配与 fallback |
| 通知层 | `src/notification.py` | 多渠道推送（企微/飞书/Telegram/邮件/Pushover） |
| 交易策略 | `strategies/*.yaml` | 15+ 内置 YAML 策略，支持自定义 |
| Bot 接入 | `bot/` | 钉钉/飞书/Telegram/微信机器人 |
| API 服务 | `api/` | FastAPI REST API |
| Web 前端 | `apps/dsa-web/` | React + TypeScript |
| 桌面端 | `apps/dsa-desktop/` | Electron 桌面应用 |

## 运行入口

### CLI

```bash
python main.py                    # 正常运行（分析配置的自选股）
python main.py --debug            # 调试模式
python main.py --dry-run          # 仅获取数据不分析
python main.py --stocks 600519,hk00700,AAPL  # 指定股票
python main.py --market-review    # 仅执行大盘复盘
python main.py --schedule         # 定时调度模式
python main.py --serve            # 启动 API + Web 服务
python main.py --serve-only       # 仅启动 API 服务
```

### API 服务

```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

## 核心函数

### 1. 分析单只股票

**函数:** `analyze_stock`（`src/services/analyzer_service.py`）

```python
from src.services.analyzer_service import analyze_stock

result = analyze_stock("600519")                    # A 股
result = analyze_stock("hk00700")                   # 港股
result = analyze_stock("AAPL")                      # 美股
result = analyze_stock("600519", full_report=True)  # 完整报告
```

**参数:**
- `stock_code` (str): 股票代码。A 股纯数字（600519），港股 `hk` 前缀（hk00700），美股直接 ticker（AAPL）
- `config` (Config, 可选): 配置对象，默认从 `.env` 加载
- `full_report` (bool): 是否生成完整报告，默认 `False`
- `notifier` (NotificationService, 可选): 通知服务

**返回:** `Optional[AnalysisResult]`

### 2. 分析多只股票

**函数:** `analyze_stocks`

```python
from src.services.analyzer_service import analyze_stocks

results = analyze_stocks(["600519", "300750", "hk00700"])
for r in results:
    print(f"{r.name}: {r.operation_advice}, 情绪 {r.sentiment_score}")
```

### 3. 大盘复盘

**函数:** `perform_market_review`

```python
from src.services.analyzer_service import perform_market_review

report = perform_market_review()  # 按配置的 market_review_region 执行
```

支持市场区域：`cn`（A 股）、`us`（美股）、`hk`（港股）、`both`（全部）。通过 `MARKET_REVIEW_REGION` 环境变量配置。

## 分析输出结构 (`AnalysisResult`)

`dashboard` 属性包含四个核心部分：

1. **`core_conclusion`**: 一句话总结、信号类型（buy/hold/sell）、时间敏感度、仓位建议
2. **`data_perspective`**: 技术数据——趋势状态、价格位置、量能分析、筹码结构
3. **`intelligence`**: 定性信息——新闻、风险警报、积极催化剂
4. **`battle_plan`**: 可操作策略——狙击点（买/卖目标）、仓位策略、风险控制清单

关键字段：
- `sentiment_score` (int): 情绪评分 0-100
- `operation_advice` (str): 操作建议（买入/持有/卖出）
- `trend_prediction` (str): 趋势预测
- `confidence_level` (str): 置信度

## Agent 模式

系统支持多 Agent 协同分析，通过 `AGENT_MODE=true` 或 API `skills` 参数启用。

### Agent 类型

| Agent | 职责 |
|-------|------|
| `TechnicalAgent` | 技术面分析（均线、MACD、RSI、筹码） |
| `IntelAgent` | 情报搜索（新闻、公告、舆情） |
| `RiskAgent` | 风险评估（利空排查、退市风险） |
| `PortfolioAgent` | 仓位策略（持仓分析、资金管理） |
| `DecisionAgent` | 最终决策（综合各 Agent 意见） |
| `SkillAgent` | 策略执行（加载 YAML 策略） |

### 编排模式

- `standard`: 标准流水线（Technical → Intel → Risk → Decision）
- `specialist`: 策略专家模式（Skill → Decision）

通过 `AGENT_ARCH` 环境变量配置。

## 交易策略

策略文件存放在 `strategies/` 目录，YAML 格式，系统启动时自动加载。

### 内置策略

| 策略 | 文件 | 描述 |
|------|------|------|
| 默认多头趋势 | `bull_trend.yaml` | MA5>MA10>MA20 多头排列 |
| 缠论 | `chan_theory.yaml` | 缠中说禅技术分析 |
| 龙头战法 | `dragon_head.yaml` | 板块龙头选股 |
| 情绪周期 | `emotion_cycle.yaml` | 市场情绪轮动 |
| 事件驱动 | `event_driven.yaml` | 消息面驱动分析 |
| 箱体震荡 | `box_oscillation.yaml` | 箱体区间操作 |
| 缩量回踩 | `shrink_pullback.yaml` | 缩量回踩支撑 |
| 放量突破 | `volume_breakout.yaml` | 放量突破关键位 |
| 波浪理论 | `wave_theory.yaml` | 艾略特波浪 |
| 底部放量 | `bottom_volume.yaml` | 底部放量信号 |
| 均线金叉 | `ma_golden_cross.yaml` | MA 金叉信号 |
| 一阳三阴 | `one_yang_three_yin.yaml` | K 线组合形态 |
| 成长质量 | `growth_quality.yaml` | 基本面成长筛选 |
| 预期重估 | `expectation_repricing.yaml` | 业绩预期差 |
| 热门题材 | `hot_theme.yaml` | 题材热点跟踪 |

### 自定义策略

创建 `.yaml` 文件到 `strategies/` 或通过 `AGENT_SKILL_DIR` 指定自定义目录：

```yaml
name: my_strategy
display_name: 我的策略
description: 策略描述
category: trend  # trend/pattern/reversal/framework
instructions: |
  用自然语言描述判断标准、入场条件、出场条件。
  可引用工具名（get_daily_history, analyze_trend）指导 AI 使用数据。
```

## 通知渠道

通过 `.env` 配置，支持多渠道同时推送：

| 渠道 | 配置项 |
|------|--------|
| 企业微信 | `WECHAT_WEBHOOK_URL` |
| 飞书 | `FEISHU_WEBHOOK_URL` |
| Telegram | `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` |
| 邮件 | `SMTP_SERVER` + `SMTP_USER` + `SMTP_PASSWORD` + `EMAIL_RECIPIENTS` |
| Pushover | `PUSHOVER_USER_KEY` + `PUSHOVER_API_TOKEN` |
| 钉钉 | 通过 Bot 平台接入 |

## 报告类型

| 类型 | 说明 |
|------|------|
| `simple` | 简洁报告（默认） |
| `detailed` | 详细报告 |
| `full` | 完整报告（含全部数据） |
| `brief` | 极简摘要 |

通过 `REPORT_TYPE` 环境变量或 API 参数 `report_type` 设置。

## 多语言

通过 `REPORT_LANGUAGE` 环境变量设置：`zh`（中文，默认）、`en`（英文）。

## 配置

所有配置通过 `.env` 文件管理，关键配置项：

```env
# 自选股列表
STOCK_LIST=600519,300750,002594

# LLM 配置（支持 Gemini/Anthropic/OpenAI/自定义）
GEMINI_API_KEY=your_key
LITELLM_MODEL=gemini/gemini-2.5-flash

# Agent 模式
AGENT_MODE=false
AGENT_SKILLS=all  # 或逗号分隔的策略名

# 通知
WECHAT_WEBHOOK_URL=https://...
REPORT_TYPE=simple
REPORT_LANGUAGE=zh

# 大盘复盘区域
MARKET_REVIEW_REGION=cn  # cn/us/hk/both
```

完整配置参考 `.env.example`（800+ 行详细注释）。

## 数据源

系统支持多数据源自动切换与 fallback：

| 数据源 | 市场 | 用途 |
|--------|------|------|
| AkShare | A 股 | 行情、K 线、板块 |
| Tushare | A 股 | 增强数据（需 Token） |
| TickFlow | A 股 | 大盘复盘指数增强 |
| Longbridge | A/港/美 | 量比、换手率、PE |
| yfinance | 港/美 | 基本面、K 线 |
| Finnhub | 美股 | 增强数据 |
| AlphaVantage | 美股 | 增强数据 |

单一数据源失败不会拖垮整个分析流程。

## 代码参考

| 文件 | 说明 |
|------|------|
| `src/services/analyzer_service.py` | 分析服务入口（`analyze_stock`, `analyze_stocks`, `perform_market_review`） |
| `src/services/analysis_service.py` | API 层分析服务 |
| `src/core/pipeline.py` | `StockAnalysisPipeline` — 完整分析流水线 |
| `src/analyzer.py` | `GeminiAnalyzer` — LLM 分析器 |
| `src/core/market_review.py` | `run_market_review` — 大盘复盘 |
| `src/agent/orchestrator.py` | `AgentOrchestrator` — Agent 编排 |
| `src/agent/executor.py` | `AgentExecutor` — Agent 执行器 |
| `src/config.py` | `Config` — 配置管理 |
| `src/notification.py` | `NotificationService` — 通知服务 |
| `data_provider/base.py` | `DataFetcherManager` — 数据源管理 |
