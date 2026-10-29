# 本地量化工作台安装与复现手册

本文同时面向用户和 agent。用户可以按步骤把本地 DSA 量化工作台跑起来；agent 可以按同一套步骤复现环境、启动服务、验证状态和排查问题。

## QMT 原生数据入口

新 QMT 读取环境要求 Python 3.11 或更新版本；本机验收使用 DSA Python 3.12.7。SDK 解释器沿用券商官方 xtquant 的匹配环境。

本机 Windows QMT 的新主线需要两套已有环境：DSA Python 安装 `quant_qmt_data`，`QMT_PYTHON` 指定的 Python 保留券商官方 xtquant 及其依赖。SDK 环境不必安装 DSA 或本数据包。将 `QMT_PORT` 设为已核实的行情端点，`QMT_DATA_DIR` 可选指定行情库；本机值放私有 `.env`，不写入源码。调用方按环境启动 worker 的绝对文件路径，不要求 cwd 位于仓库。

先在 QMT 中维护所需日线，再使用数据包 `export` 命令保存一个新实验目录，将目录设为 `QMT_DATASET_DIR`。完整输入包含 `bars.parquet` 与 `dataset.json`，训练入口选择 `qmt_native` 和 `baseline_qmt_native.yaml`。之后可离线训练、预测与 DSA 回测，无需旧数据服务、全库清单或发布指针。未收盘数据与中途保存失败的目录不能作为正式输入。

新训练默认使用 `qmt_native` 与 `baseline_qmt_native.yaml`，只读取冻结输入。已训练 Run 可选择“每日预测 TopK”进行 DSA 自定义回测，按下一交易日开盘撮合；初始资金、整手与每侧成交费用均进入结果。它保留完整成交账本，不把标签收益当成撮合收益，也不把预测契约标成已冻结的精确策略。`QMT_DATASET_DIR` 以后改变时，保存 Run 仍绑定其原输入目录。

本节是新 QMT 数据入口的安装方式；下文中的旧数据源账号、canonical/cleaned Provider 构建等内容只适用于独立/历史工作流，不是 QMT 原生模式的依赖。

## 目标架构

| 组件 | 角色 | 安装方式 |
| --- | --- | --- |
| DSA | 主网页、API、任务中心 | clone 本仓库 |
| Qlib | 量化训练、预测、回测引擎 | 在 DSA `.venv` 中通过 PyPI 安装 |
| MLflow | 实验原始记录服务 | 在 DSA `.venv` 中通过 PyPI 安装 |
| `quant_workflow` | 数据、转换、报告、训练脚本层 | 已内置在 DSA 仓库的 `quant_workflow/` |
| `china-astock-quant` | A 股指标和策略参考 | 可选外部参考项目 |
| `quant_data` | 本地行情、Qlib provider、缓存 | 默认放在 DSA 仓库的 `quant_data/`，不入库 |

普通用户不需要单独开发 Qlib。Qlib 作为 Python 依赖被 DSA 和 `quant_workflow` 调用。

## 前置条件

- macOS、Linux 或 Windows WSL。
- Python 3.10 或 3.11，优先使用项目已验证版本。
- Node.js 和 npm，用于构建 `apps/dsa-web`。
- Git。
- 可选：JQData、Tushare、AkShare 等数据源账号或 token。

## 目录约定

推荐目录：

```bash
/path/to/
  daily_stock_analysis/
    quant_workflow/
  quant_qmt_data/
    data/
  china-astock-quant/
```

如果不使用这个根目录，必须在 `.env` 中覆盖：

```bash
DSA_ROOT=/path/to/daily_stock_analysis
WORKING_PROJECTS_ROOT=/path/to
QUANT_WORKFLOW_ROOT=/path/to/daily_stock_analysis/quant_workflow
QUANT_DATA_ROOT=/path/to/quant_qmt_data/data
```

## Mac 本机数据入口

DSA 使用独立的 `quant_qmt_data` Python 包和 `quant_qmt_data/data` 快照目录；它不是 HTTP 数据服务。先在同一 DSA 环境安装本地包：

```bash
/path/to/daily_stock_analysis/.venv/bin/python -m pip install -e '/path/to/quant_qmt_data[ssh]'
```

本分支需要配套轻量包中的 `bar_snapshot`、`inspect_datasets` 及 `read_bars(expected_identity=...)` 接口；不能仅更新 DSA 后仍使用旧版轻量包。普通行情、脚本回测及 Qlib 的行情文件读取委托该包，DSA 仍负责策略、历史成员池与研究清洗规则。`/data` 展示已提交快照的日期、口径及摘要，只核对清单身份，不冒充全库质量体检。回测保存快照 ID、manifest/content/file-index 摘要；重放按保存的身份读取，旧版被清理后明确失败，不静默改读新版。股票成交量按股，ETF 按份；文件中的原始数量不改写。

本机存在或显式配置该快照库时，普通 A 股／ETF 分析的历史日线也从相同快照读取并保留 DataFrame 身份；缺失时提示显式更新，不自动转向供应商取另一版行情。实时行情、新闻及海外市场继续使用现有入口。

网页 `/workflow` 提供独立的“更新股票快照（MiniQMT）”和“更新 ETF 快照（MiniQMT）”，只做行情更新，不附带训练。命令入口保持：

```bash
.venv/bin/python quant_workflow/scripts/quant_data_cli.py update ashare_daily_qfq --through YYYY-MM-DD
.venv/bin/python quant_workflow/scripts/quant_data_cli.py update cn_etf_daily --through YYYY-MM-DD
```

它们调用轻量包既有的有界补缺、重叠核对及快照发布；Mac 实际采集由该包的 OpenSSH 配置决定。每日任务的默认 A 股来源为 `miniqmt`；`none` 显式跳过，其他来源仅更新审计缓存。数据根不可用就报错，不自动选择封存目录或项目版服务。此次接线验收不执行真实供应商采集，也不更新正式数据；本机价格快照的截止日仍以页面实际显示为准。

## 安装 DSA 后端

```bash
cd /path/to
git clone https://github.com/ZhuLinsen/daily_stock_analysis.git daily_stock_analysis
cd daily_stock_analysis

python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果 Qlib、MLflow、可选树模型或数据源包没有被 `requirements.txt` 覆盖，补装：

```bash
pip install pyqlib mlflow xgboost catboost akshare baostock
```

`xgboost` 和 `catboost` 只提供 Qlib 可选选股模型，不会启动新服务或占用新端口。

JQData 的安装方式以聚宽官方说明为准。不要把账号密码写进命令行历史。

## 安装前端

```bash
cd /path/to/daily_stock_analysis/apps/dsa-web
npm ci
npm run build
```

开发调试可使用 Vite；正式本地工作台以 DSA 后端提供的静态页面为主。

## 配置 `.env`

```bash
cd /path/to/daily_stock_analysis
cp .env.example .env
```

至少检查：

```bash
DSA_BIND_HOST=127.0.0.1
UVICORN_HOST=127.0.0.1
UVICORN_PORT=8010
DSA_ROOT=/path/to/daily_stock_analysis
WORKING_PROJECTS_ROOT=/path/to
QUANT_WORKFLOW_ROOT=/path/to/daily_stock_analysis/quant_workflow
# reports/mlruns 的正式位置是仓内 quant_workflow；外部路径仅用于迁移/回滚。
# QUANT_WORKFLOW_REPORTS_ROOT=/path/to/daily_stock_analysis/quant_workflow/reports
# QUANT_WORKFLOW_MLRUNS_DIR=/path/to/daily_stock_analysis/quant_workflow/mlruns
# 默认数据根为 DSA 同级的 quant_qmt_data/data；其它位置须显式配置。
# QUANT_DATA_ROOT=/path/to/quant_qmt_data/data
QUANT_WORKFLOW_PYTHON=/path/to/daily_stock_analysis/.venv/bin/python
MLFLOW_URL=http://127.0.0.1:6006/
```

需要 JQData 时，优先把账号密码放到本机私有文件，不放在项目目录：

```bash
mkdir -p ~/.agents/private/quant-workbench
chmod 700 ~/.agents/private/quant-workbench
cat > ~/.agents/private/quant-workbench/jqdata.env <<'EOF'
JQ_USERNAME=你的聚宽手机号或邮箱
JQ_PASSWORD=你的聚宽密码
EOF
chmod 600 ~/.agents/private/quant-workbench/jqdata.env
```

如需放到其他位置，在 DSA `.env` 里只写路径：

```bash
JQDATA_ENV_FILE=/absolute/path/to/jqdata.env
```

项目 `.env` 是本机配置文件，禁止提交；真实账号、密码和 token 更推荐放在 `~/.agents/private/`。

## 准备相关项目

`quant_workflow` 已经随 DSA 仓库提供，不需要单独 clone。MLflow 双语页面也不需要 clone MLflow 源码；DSA 直接代理已安装的 MLflow 3.14.0 页面。`china-astock-quant` 可以浅克隆，作为可选参考项目；浅克隆节省 Git 历史空间，但源码仍可正常读取和运行。

```bash
cd /path/to
git clone --depth 1 https://github.com/MisakaMikoto128/china-astock-quant.git china-astock-quant
mkdir -p quant_data
```

DSA 通过 `.env` 中的 `QUANT_WORKFLOW_ROOT` 读取量化脚本层。默认值指向仓内 `quant_workflow/`；只有你把脚本层移到外部目录时才需要覆盖。

`quant_workflow/reports/` 和 `quant_workflow/mlruns/` 是 DSA 量化工作台的默认运行态目录，但它们被 `.gitignore` 排除，不会随 GitHub 分发。旧安装如果还保留 DSA 同级的 `quant_workflow/reports` 或 `mlruns`，路径解析器不会再按 marker 自动读取同级旧历史；需要迁移时应复制到仓内目录，需要临时回滚时显式设置 `QUANT_WORKFLOW_REPORTS_ROOT` / `QUANT_WORKFLOW_MLRUNS_DIR`。

## 启动 DSA

推荐用 `screen` 或 `tmux` 保持服务运行：

```bash
cd /path/to/daily_stock_analysis
mkdir -p logs
screen -dmS dsa-8010 zsh -lc '. .venv/bin/activate && python main.py --serve-only --host 127.0.0.1 --port 8010 >> logs/dsa_web_8010.log 2>&1'
```

检查：

```bash
curl -sS --max-time 5 -o /tmp/dsa-home.html -w '%{http_code} %{size_download}\n' http://127.0.0.1:8010/
```

期望返回 `200`。`8010` 是 DSA 日常网页入口；`6006` 是 MLflow 后台数据服务。

## 启动 DSA（跳过前端构建）

当 npm 缓存有权限问题（`EPERM` 错误）导致 `npm install` 卡住时，可设置 `WEBUI_AUTO_BUILD=false` 跳过前端构建，直接复用 `static/` 中已有的产物：

```bash
cd /Users/cq/WorkingProjects/daily_stock_analysis
. .venv/bin/activate
WEBUI_AUTO_BUILD=false python main.py --serve-only --host 127.0.0.1 --port 8014
```

访问地址：
- Web UI：http://127.0.0.1:8014/research
- API 文档：http://127.0.0.1:8014/docs
- 健康检查：http://127.0.0.1:8014/health

npm 缓存权限修复方法：
```bash
sudo chown -R 501:20 "/Users/cq/.npm"
```

## 启动 MLflow

MLflow 3.x 对 file store 有安全限制，本地启动时显式允许：

```bash
cd /path/to/daily_stock_analysis
mkdir -p logs quant_workflow/mlruns
screen -dmS mlflow-6006 zsh -lc 'MLFLOW_STORE="${QUANT_WORKFLOW_MLRUNS_DIR:-$PWD/quant_workflow/mlruns}"; MLFLOW_ALLOW_FILE_STORE=true .venv/bin/python -m mlflow ui --backend-store-uri "file://$MLFLOW_STORE" --host 127.0.0.1 --port 6006 >> logs/mlflow_6006.log 2>&1'
```

检查：

```bash
curl -I --max-time 5 http://127.0.0.1:6006/
```

期望返回 `200`。

## 打开网页

| 页面 | 地址 | 用途 |
| --- | --- | --- |
| 首页 | `http://127.0.0.1:8010/` | 股票分析入口 |
| 量化研究 | `http://127.0.0.1:8010/research` | 数据状态、Qlib 训练和策略研究入口 |
| 回测 | `http://127.0.0.1:8010/backtest` | Python 脚本回测和无代码条件回测 |
| 实验 | `http://127.0.0.1:8010/experiments/` | MLflow 3.14.0 官方页面、中文/English 切换、Run 和 artifact |

`6006` 继续运行，但只供 DSA 后端代理访问。不要再启动 `6007`，也不需要单独运行翻译代理进程。

维护者需要重新生成 MLflow 翻译目录时，只下载官方 3.14.0 的单个 `zh-CN.json`，不要 clone 完整 MLflow 仓库：

```bash
curl -fsSLo /tmp/mlflow-zh-CN-3.14.0.json \
  https://raw.githubusercontent.com/mlflow/mlflow/86cd7f5a1dc25a1387ad07c87811bbc30f62951c/mlflow/server/js/src/lang/zh-CN.json
.venv/bin/python scripts/build_mlflow_i18n_catalog.py \
  --zh-cn /tmp/mlflow-zh-CN-3.14.0.json
```

生成结果是约 90 KB 的英文到中文显示映射，不包含 MLflow 源码、实验数据或模型文件。
生成脚本会核对已安装版本必须是 `3.14.0`，并在产物中记录上游 commit 和翻译源文件 SHA-256；版本不一致时直接停止，不能沿用旧版本标签。

## 验证

快速验证：

```bash
cd /path/to/daily_stock_analysis
scripts/verify-local-quant-workbench.sh quick
```

完整验证：

```bash
scripts/verify-local-quant-workbench.sh full
```

最终闭环：

```bash
scripts/verify-local-quant-workbench.sh complete
```

`complete` 会跑 `full`，然后执行 opencode 只读 review。它不是“下载全量数据”，而是代码、前端、脚本、浏览器 smoke 和安全 review 的完整本地验收。

## 其他用户如何使用 GitHub 版本

GitHub 仓库应只包含可复现的代码、测试、文档和示例配置。其他用户 clone 后：

1. 创建自己的 `.venv`。
2. 安装 DSA 依赖、Qlib、MLflow 和数据源包。
3. 复制 `.env.example` 为 `.env`，填写自己的路径和 token。
4. 准备自己的 `quant_data` 数据目录。
5. 启动 DSA 和 MLflow。
6. 在 DSA 查看数据状态，并在量化研究入口运行训练和回测。

不要把你的 `.env`、股票全量数据、Qlib provider、MLflow `mlruns`、日志和报告推送到 GitHub。它们是本地私有运行态资产。仓库只提交 `quant_workflow` 的脚本、测试和说明文件，不提交其中的 `reports/` 与 `mlruns/` 运行数据。

## 常见问题

### commit 是否等于推送 GitHub

不是。`git commit` 只是在本机创建版本快照；`git push` 才会上传到 GitHub。没有用户明确确认时，agent 不应执行 commit 或 push。

### 为什么不用完整 clone Qlib

普通使用不需要。PyPI 包已经包含运行所需源码和模块；完整 clone Qlib 只适合研究或开发 Qlib 本身。

### DSA 和 MLflow 如何分工

DSA 是主操作台，`/experiments/` 是日常使用的双语实验入口。它代理并翻译本机 MLflow 官方界面；MLflow 仍在后台 `6006` 保存和提供 run、参数与 artifact，不由 DSA 重写实验计算或存储逻辑。

### 为什么网页任务不能输入任意命令

因为本地工作台会接触数据、路径和凭据。网页只能调用后端白名单任务，这样才能控制参数、日志、环境变量和产物路径。
