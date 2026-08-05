# DSA 网页量化训练闭环升级 Goal（2026-08-02）

## 用户最终要得到什么

不了解后台命令的人，可以只使用 DSA 网页完成以下闭环：

1. 验收并选择一份不可变的 Qlib V2 数据快照。
2. 运行测试式选股模型训练，且不会污染正式结果。
3. 运行正式选股模型训练，看到真实进度、日志、产物和失败原因。
4. 对长任务执行“完成当前子任务后暂停”，并在契约一致时从检查点恢复。
5. 区分测试结果、正式训练结果和真正可回测的结果。
6. 从训练结果点击“回测”，自动带入 Run、数据快照、选股模型和交易策略。
7. 在相同配置、数据快照和随机条件下，网页训练与后台脚本得到相同预测和指标。

本 Goal 不负责寻找一套盈利策略，不启动不受控的完整长期训练，不连接券商，也不自动下单。本轮只解决“人类能否可靠地通过网页完成与脚本等价的训练和回测准备”。

## 今天网页实操暴露的问题

| 优先级 | 实际问题 | 风险 | 当前状态 |
| --- | --- | --- | --- |
| P1 | 五模型 dry-run 覆盖正式 `candidate_pipeline_latest` | 测试数字会伪装成最新研究结果 | 已修复，测试产物进入独立目录 |
| P1 | 缺少 Run ID 和预测的 dry-run 仍显示“回测” | 用户跳转后才发现无法运行，且容易误认其为真实结果 | 已修复，改用显式回测资格契约 |
| P1 | 首次真实等价验收把测试期自动延长到 provider 最新日期 | 网页与脚本名义同配置、实际周期不同 | 已修复，验收模式严格冻结训练、验证和测试周期 |
| P1 | 通用数据验收通过后，鲁棒 CSI800 研究仍缺少自己的 provider 契约，但页面允许预览和正式启动 | 用户看到“可训练”，实际任务立即因 PIT CSI800 provider 缺失失败 | 已修复；鲁棒 CSI800 使用独立数据契约、准备任务和启动守卫 |
| P1 | 鲁棒单季度测试与正式研究共用 `latest` 状态 | smoke 可能覆盖正式阶段、检查点和页面结论 | 已修复；正式与 smoke 状态、别名和页面区域彻底分离 |
| P1 | 正式 LightGBM Run 宣称可原样重放，但只登记 `pred.pkl` 和 `label.pkl` 哈希，未登记 Qlib `task` 哈希 | 无法证明重放使用的是训练时冻结的组合回测配置 | 已修复；发布与重放都强制校验三份产物，旧 Run 自动降级为仅自定义回测 |
| P2 | 前端较新、后端较旧时按钮无解释变灰 | 用户无法判断是配置、数据还是服务版本问题 | 已实现 API 契约、构建版本、加载错误和重试提示 |
| P2 | 正式训练被数据门槛阻止时只能跳页，不能形成操作闭环 | 普通用户不知道该生成哪份报告、完成后如何返回 | 已修复并通过 Edge；数据页可验收，训练页显示 snapshot、阻塞原因和返回入口 |
| P2 | 任务卡、顶部计数、日志和准备状态不同步 | 已完成任务看起来仍在运行 | 已修复；SSE、延迟状态发现和运行中轮询共同刷新任务与研究状态 |
| P2 | 新 smoke 已创建，但测试状态仍短暂显示上一 Run | 用户必须手动刷新，或误以为按钮没有生效 | 已修复；运行期间每 3 秒刷新，并在 launcher 结束后继续做延迟状态发现 |
| P2 | 长任务只有强制取消，没有协作式暂停、检查点和恢复 | 训练中断后可能只能重跑，或错误恢复不同配置 | 已通过 Edge；暂停在原子任务结束后落盘，恢复校验模式与运行契约 |
| P2 | 严格鲁棒信号研究只有脚本入口 | 人类无法从网页运行 Goal 1 的 smoke、计划、暂停和恢复 | 已通过 Edge；网页提供准备、预览、smoke、正式启动、暂停和恢复受控入口 |
| P2 | 训练进度只有“运行中”，缺少阶段、数量和耗时 | 用户无法判断是否卡住 | 已增加结构化进度；ETA 仅在可可靠计算时显示 |
| P2 | 旧 DSA 进程残留并继续占用 8010 | 新代码已启动但浏览器仍访问旧服务 | 已清理并通过重启验收；当前页面同时显示前端、后端和接口契约构建身份 |
| P2 | `/experiments/` 启动 MLflow 时解析了 `.venv/bin/python` 的符号链接 | 实际使用 uv 基础 Python，MLflow 子进程立即退出，网页只看到服务不可用 | 已修复；保留虚拟环境解释器路径并检测启动后立即退出，Edge 已显示 MLflow 3.14.0 中文界面 |
| P2 | 进度 JSON 单个计数字段损坏会让状态接口抛出异常 | 页面整体无法读取研究状态 | 已增加损坏字段回归测试，非法计数安全降为 0 |
| P3 | 轮询重渲染会打断复选框操作 | 自动化超时，人工操作也可能不稳定 | 已增加前端状态稳定性回归；最终全量前端回归通过 |
| P3 | provider 实际为 canonical，但下拉框初始显示“默认” | 页面显示与真实输入不一致 | 已修复；当前 Edge 显示“原生 V2（正式）” |
| P3 | 数据质量摘要仍夹杂英文警告 | 中文页面理解成本高 | 已修复展示层中文提示；原始诊断字段保持不变 |

## 已取得的真实等价证据

- 数据快照：`qlib-ashare-v2-20260802T021409-28d2dd81`。
- 有效网页 Run：`e144ddd980514280ad2e07b6eae68dae`。
- 有效脚本 Run：`91951f0af26143d3ab9f4169a2028d7b`。
- 训练期：`2024-01-02 ~ 2024-06-28`。
- 验证期：`2024-07-01 ~ 2024-09-30`。
- 测试期：`2024-10-08 ~ 2024-12-31`。
- 两个 Run 属于专用 MLflow 实验 `dsa_training_equivalence`，Run ID 不同但实验归属一致。
- provider identity、源配置哈希、运行时配置哈希和三个周期一致。
- `pred.pkl` 与 `label.pkl` 的 SHA-256 分别一致。
- 15 项训练及组合回测指标在 `1e-12` 容差内一致。
- 普通可回测目录仍为 9 个正式 Run，普通实验记录仍为 16 条；验收 Run 没有进入两处目录。
- 比较结果见 `quant_workflow/reports/test_runs/training_equivalence/comparison_latest.json`，状态为 `pass`。

首次网页 Run `a709eac9126d448c8f0264e00a30e963` 只保留为“自动扩期缺陷”的失败证据，不能作为等价通过证据。

## 本轮必须完成的升级范围

1. 保持测试训练、等价验收与正式训练的目录、MLflow 实验、latest 指针和回测目录隔离。
2. 回测资格必须由真实 Run ID、可重放预测、数据快照和交易策略配置共同决定。
3. 数据页、训练页和回测页必须使用同一 provider identity，并显示不可变 snapshot ID。
4. 所有运行入口仅允许服务端白名单参数，不开放任意脚本、命令、路径、实验名或输出目录。
5. 暂停与恢复必须校验配置哈希、Git 提交和数据快照/清单哈希；取消不能伪装成可恢复暂停。
6. 任务结束后，任务卡、日志、顶部计数、准备状态和产物必须一起刷新；截断日志必须明示。
7. 页面必须清楚显示当前后端构建、API 契约、runner/PID 和存活状态，能够识别访问旧服务的情况。
8. 补齐中文提示、provider 默认显示和今天剩余的真人操作路径。
9. 更新维护文档，明确网页已支持、脚本专用和仍未实现的能力，不得把计划写成已完成。
10. 将“通用 LightGBM 可训练”和“鲁棒 PIT CSI800 可训练”拆成两个明确契约；计划预览必须能展示数据缺口，测试/正式启动不得在缺少鲁棒 provider 时误导性放行。

## 2026-08-02 17:30 Edge 真人回归证据

- 当前前端构建：`build-20260802-092538Z`。
- 当前后端构建：`localquant-ui-training-20260802-r1`。
- 接口契约：`quant-workbench-v2`。
- 鲁棒数据 provider key：`876aa8a1dd91b230d1a39a5fbc5dc79cb3ae2d1312658128d717bd2888990b55`。
- 鲁棒数据清单：`af07e35a5c3ebb2320e8d2c94332b212e38f432abdf4a1ab3c20cba52584b23a`。
- Edge 启动 smoke 任务：`qj_20260802173040_dded31d6`。
- 页面自动发现 Run：`robust_signal_20260802_173044`，无需手动刷新。
- Edge 请求协作暂停：`qj_20260802173117_0632277a`；页面在当前原子任务完成后显示“已暂停”。
- Edge 从同一检查点恢复：`qj_20260802173230_dae30fee`；请求明确带有 `smoke=true`，最终显示 `1 / 1 · smoke_complete · 已完成`。
- smoke 前后三份正式文件 SHA-256 均未变化：`candidate_pipeline_latest.json=69b8cd76...`、`candidate_pipeline_latest.md=309bda4b...`、`latest_training_result.json=14a2191b...`。
- 不可回测结果只显示“查看测试结果”，同时显示“缺少可重放的预测和标签”；点击后停留在工作流最近任务区域，不会伪装成回测入口。
- 回测页可以从 URL 注入 Run、选股模型、交易策略和 snapshot，并显示冻结参数。当前目录唯一带策略的历史 Run 缺少精确重放契约，因此页面正确禁用“按原条件重放”，只允许明确标记的自定义回测。

### 尚未完成，不能写成可用

- Edge 移动宽度回归、全量测试、lint 和生产构建仍需在最终 diff 上重新执行。
- Codex 子 Agent、OpenCode、Claude Code 尚未对最终同一 diff 完成独立审查。

## 2026-08-03 正式训练与原条件重放验收

- 当前前端构建：`build-20260802-190427Z`。
- 当前后端构建：`localquant-ui-training-20260803-r5`。
- Edge 从 `/workflow` 点击“运行 训练 Qlib LightGBM 选股模型基线”，创建任务 `qj_20260803033245_ac074ebe`。
- 正式 MLflow Run：`4be0a4ea8067402a8fac3d492f4d1fc7`；数据快照：`qlib-ashare-v2-20260802T021409-28d2dd81`。
- 网页产物预览明确显示 `pred.pkl`、`label.pkl` 和 `task` 三份 SHA-256；`task=733f30f18df8b032c545ceee6bfc4a3f5f94cfebd6ce6ae58e8af3a7947c9440`。
- 从训练结果行点击“回测”后，页面自动注入 Run、snapshot、选股模型、原始交易策略、日期、TopK 和成本。
- Edge 点击“按原条件重放”，实际使用“Qlib 原始组合回测”引擎完成 `2024-01-02 ~ 2026-07-23` 共 `618` 个交易日。
- 重放后的扣成本年化超额 `10.890785347485%`、信息比率 `0.904776794557`、超额最大回撤 `-10.892216305774%` 与训练 Run 保存值一致，三项最大绝对误差 `1.33e-15`，低于 `1e-10` 门槛。
- 页面显示非空累计收益图；策略总收益 `75.56%`，沪深300总收益 `37.80%`，策略最大回撤 `-20.46%`。这些是本次基线回测结果，不等于鲁棒策略合格结论。
- 把交易策略切换为“每周低换手”后，页面改为“仅自定义回测”，禁用“按原条件重放”，并解释只有训练时真正执行和冻结的原始策略才能原样重放。
- 修复前 Run `7de3e751dfb64e0591e82affb7c5e6bb` 因缺少 `task` 哈希被准确降级，页面明确显示缺少 `pred.pkl`、`label.pkl` 或 `task` 哈希，不再误报支持原样重放。

## 2026-08-03 最终自动化与响应式验收

### 自动化证据

- 主后端：`3349 passed, 2 skipped`。
- `quant_workflow`：`1094 passed`。
- 前端单元测试：`762 passed, 2 skipped`。
- 前端 lint：通过。
- 前端生产构建：通过。
- 浏览器完整 smoke：第 1 至 22 项通过；取消流程按真实确认弹窗修正后，聚焦用例通过；第 24 至 36 项为 `12 passed, 1 skipped`，唯一跳过项是正式服务不启用假分析入口。
- 独立受控假分析用例再次运行，结果为 `1 passed`。
- `git diff --check`：通过。

### Edge 真人证据

- 桌面视口为 `1714 x 849`，页面无水平溢出。从网页再次点击“按原条件重放”，Run `4be0a4ea8067402a8fac3d492f4d1fc7` 在约 20 秒内完成。
- 桌面重放重新显示 `75.56%` 策略总收益、`37.80%` 沪深300总收益、`10.89%` 扣成本年化超额、`-20.46%` 最大回撤及 `1.33e-15` 最大指标误差；红色策略曲线和蓝色基准曲线均非空。
- 使用 Edge 临时 `390 x 844` 响应式视口重新加载并再次从页面触发同一 Run。实际 CSS 内容宽度为 `384`，无水平溢出、无交互控件越界；结果卡片与净值图可见，最大可见图表 SVG 为 `276 x 288`。
- 桌面与手机两次重放期间均没有新增浏览器控制台 error 或 warning。日志中更早的浏览器扩展 message-channel 错误发生在本轮重放之前，且不是 DSA 页面代码产生，未计入本轮结果。
- 临时移动视口已经 reset，未改变用户 Edge 的原始窗口大小。

### 运行环境

- `8010` 只有一个 DSA 监听进程，后端构建为 `localquant-ui-training-20260803-r5`，API 契约为 `quant-workbench-v2`。
- `6006` 由同一个 MLflow 3.14.0 主进程及其 4 个 Uvicorn worker 共享监听，不是 5 套重复 MLflow 服务。
- 最近正式训练任务状态为 `completed`、`return_code=0`、`runner_alive=false`；没有残留的量化训练脚本或待执行任务。
- CodeGraph 已索引 `932` 个文件，状态正常，无待同步文件提示。

## 完成门槛

以下条件缺一不可：

- 后端、quant_workflow、数据契约、前端单元测试、lint 和生产构建全部通过。
- Edge 中实际完成数据验收、测试训练、正式受控训练、暂停、恢复、结果识别和回测注入路径。
- 网页与脚本真实等价比较保持 `pass`，且正式目录无测试污染。
- 对同一份最终 diff 依次取得三份独立只读审查：Codex 子 Agent、OpenCode CLI、Claude Code。
- 三方都检查功能正确性、训练/数据契约、防前视和结果污染、API 安全、测试缺口及真人网页可用性。
- 任一审查发现 P0、P1、P2 或阻塞问题，都必须修复，并让三方重新审查最终 diff。
- 只有三方最终均明确“无阻塞问题”后才能本地 commit；不 push。

## 2026-08-03 暂停恢复后的安全与性能续审

本节晚于上面的“最终自动化与响应式验收”。后续 diff 又修正了审查发现，
所以上一轮全量测试只能作为历史证据，不能代替当前最终 diff 的重新验收。

Codex 第一轮独立审查发现的七项 P1/P2 已逐项增加红绿回归测试并修正：

1. provider 就绪不再只核对清单字段；正式训练完整读取并校验 SHA-256，API
   每次检查全部路径、文件类型、权限、大小和文件指纹，仅对变化文件重新哈希。
2. Qlib 原条件重放在反序列化前，把 `pred.pkl`、`label.pkl` 和 `task` 的实际
   哈希绑定到训练报告冻结值，任一不一致立即拒绝。
3. 鲁棒 smoke 无条件关闭 MLflow，调用方传入 `mlflow_enabled=true` 也不能污染
   正式实验。
4. durable runner 恢复和取消绑定 launch token、身份文件、PID、进程启动身份、
   runner 命令哈希和任务命令哈希；身份不完整或不匹配时绝不发送信号。
5. 已完成、失败、取消或其他非运行状态不能再申请协作式暂停。
6. 进度文件按 Run ID、batch ID、smoke/formal 和 dry-run 参数匹配，不再只取
   修改时间最新的文件。
7. provider 构建失败只返回有长度上限且已脱敏的错误类型和错误摘要，不暴露
   本机绝对路径。

续审又发现并修正两项轮询性能浪费：终态历史任务不再逐条执行 `ps`，恢复监视
每轮只核验一次 runner 身份；provider 缓存校验由三次目录遍历和二次逐文件
`stat` 合并为一次遍历。真实 33,071 文件 provider 的本机结果为：

- 修正前：首次完整检查 `3.011s`，缓存检查 `2.253s`。
- 修正后：首次完整检查 `1.797s`，缓存检查 `0.500s`。
- 临时 APFS 克隆中的单文件变更在 `0.535s` 内被拒绝，只重新哈希 1 个文件；
  正式 provider 文件哈希前后不变。
- 隔离真实恢复探针 `qj_20260803053230_216f527f` 在启动服务进程退出后，由新
  `QuantJobService` 识别存活 runner，并最终收口为 `completed / return_code=0`。
  探针只使用系统临时目录，不进入正式任务库、训练报告或 MLflow。

当前 8010 仍是这些续审修正之前启动的服务。必须先完成当前 diff 的全量测试，
再重启并重新执行 Edge 真人路径；在此之前不能用旧页面证明新修复已经上线。

## 2026-08-03 市场报告重复加载修复

真人网页测试发现，“市场状态报告”和“现金仓位回测”虽然只需要读取已验收的
A 股前复权日线快照，子进程却没有收到明确的数据目录。脚本因此回退到默认解析，
每次重新核对 5,508 只股票并产生约 16,528 次 Parquet 读取，单次等待约
`46-59s`。这不是训练计算，而是同一份数据被重复验收造成的性能浪费。

当前修复把“已通过完整数据质量验收的快照”作为服务端审批凭据：

- 只接受 `scan_mode=full`、覆盖完整且状态为 `verified` 的质量报告；抽样报告不能
  取得审批资格。
- 依次核对根清单、`LATEST.json`、快照清单、内容身份和文件系统指纹；数据在
  验收后发生变化时立即拒绝旧证据。
- 浏览器只提交受控任务类型，不接收本机路径。后端从固定 `quant_data` 根和已
  审批的 `snapshot_id` 推导 `<snapshot>/bars`，再分别注入两个报告脚本。
- 审批缓存的热调用实测约 `0.17-0.20s`；受控小样本 Playwright 探针实测
  “市场状态报告”约 `2.9s`，“现金仓位回测”约 `6.3s`。这两个数字不是正式
  全市场报告的用户等待时间。

该优化不降低数据门槛，也不复用未经核验的数据。质量报告缺失、仅抽样、清单
不一致或快照已变化时，网页任务会明确失败，不会为了速度绕过验收。

## 2026-08-03 测试门禁收集范围修复

主后端门禁原本声称只运行 DSA 的 `tests/`，实际却从仓库根递归收集了
`quant_workflow/tests/`，导致其脚本专用 `PYTHONPATH` 未设置时出现收集错误，
也让同一批量化测试被重复执行。现在：

- 主后端门禁显式运行 `python -m pytest tests -m "not network"`。
- 本地工作台后端门禁显式运行 `${PYTHON_BIN} -m pytest tests -q`。
- `quant_workflow` 仍在自己的独立步骤中设置脚本 `PYTHONPATH` 并完整运行。
- 契约测试会检查上述三个边界，防止门禁以后再次静默扩大收集范围。

当前续审 diff 的主后端完整结果为 `3364 passed, 2 skipped`；浏览器 smoke、
新服务重启后的 Edge 真人路径和三方独立审查仍须完成，不能据此提前提交。

## 2026-08-03 r6 Edge 真人复验

- 托管假分析浏览器套件在当前 diff 上完成 `36 passed`，耗时约 11 分钟；没有
  跳过或失败。
- `8010` 已从旧 `r5` 重启为 `localquant-ui-training-20260803-r6`，API 契约仍为
  `quant-workbench-v2`，活动量化任务为 0。`6006` 的 MLflow 主进程与四个
  worker PID 在重启前后未变化。
- 数据页显示 canonical 与 cleaned 结构验收均通过；训练页显示当前数据可训练、
  鲁棒 CSI800 数据已就绪，正式训练入口不再被旧服务的空状态错误禁用。
- Edge 从页面启动市场状态报告 `qj_20260803065640_f93a1945` 和现金仓位回测
  `qj_20260803065729_bb5d88d4`。两条实际命令分别显式包含 `--data-dir` 和
  `--market-data-dir`，都指向已审批的同一不可变 qfq 快照；没有再次执行默认
  快照解析或完整质量验收。
- 正式全市场口径从任务启动到子进程落盘分别约 `13s` 和 `40s`。现金仓位回测
  仍需读取一次全市场价格并计算完整日序列，所以不能把小样本的 `6.3s` 当作
  生产口径；本轮消除的是任务启动前额外的整套重复验收。
- Edge 从训练结果表点击“回测”后，Run `4be0a4ea8067402a8fac3d492f4d1fc7`、
  snapshot、选股模型、交易策略、日期、TopK 和成本全部自动注入。再次点击
  “按原条件重放”约 `28.8s` 完成，策略总收益 `75.56%`、沪深300总收益
  `37.80%`、扣成本后年化超额 `10.89%`、最大回撤 `-20.46%`，训练保存指标
  最大绝对误差为 `4.44e-16`。
- Edge 桌面内容宽度为 `1714/1714`，无横向溢出。临时 `390 x 844` 视口下实际
  内容宽度为 `384`，0 个可见交互控件越界；累计收益图为 `276 x 288`，两条
  非空曲线均可见。临时视口已恢复，整轮没有新增浏览器 error 或 warning。

当前仍未完成三方独立审查和最终 commit；以上真人证据不能替代审查门槛。

## Goal 状态

Codex Goal 卡片当前系统状态为 `active`。正式网页训练和原条件重放已有历史真人
证据，但当前续审 diff 仍需重新完成主后端、quant_workflow、前端、构建、
Playwright、Edge 桌面/移动路径及三方独立审查。因此不能宣称本轮升级已经完成，
也不能提前提交。

## 2026-08-03 独立审查六项问题及修复

Planck 独立只读审查曾报告三项 P1 和三项 P2。当前 diff 已分别增加回归测试并
修正，但这些历史发现不等于最终三方审查已经通过：

1. 鲁棒信号研究的 provider 准备和正式启动都绑定不可变完整质量证据；证据必须
   对应同一 provider 指纹、`trainable=true` 且不存在结构阻塞，抽样或质量失败
   的快照不能进入正式训练。
2. 市场报告快照审批使用同一完整质量证据契约，不再只凭“full/verified”文字和
   非空股票池放行。
3. 鲁棒 launcher 的取消会核验 launchd label、launch token、runner 身份和 PID，
   等实际 runner 停止后才报告取消；仅终止包装进程不能伪装成成功取消。
4. 自定义回测明确返回实际执行引擎 `dsa_optimized_strategy`，同时把冻结 Run 的
   来源引擎单独记录为 `source_replay_engine`，不再把二者混为一谈。
5. 鲁棒 Run 的来源身份统一使用版本化字段 `membership_intervals_sha256` 和
   `quality_evidence_sha256`；缺少必需哈希时失败关闭，不静默记录空 provenance。
6. 已存在的 `{run_id}.json` 必须与本次规范化内容完全一致才能复用；碰撞、损坏
   或本地篡改会被拒绝，不能把旧 provenance 关联到新训练输出。

对应测试分布在 `test_robust_signal_data_contract.py`、
`test_prepare_robust_signal_data.py`、`test_research_status_api.py`、
`test_launch_robust_strategy.py`、`test_trained_strategy_backtest_service.py`、
`test_run_robust_signal_research.py` 和 `test_train_baseline_lgb.py`。

## 2026-08-04 状态加载与原条件重放续修

真人网页路径又暴露两项 review 难以发现的运行问题：

- `/workflow` 同时请求完整 `/api/v1/quant/status` 和独立
  `/api/v1/research/status`，同一轮页面加载把完整研究状态组装两次。现在工作流
  摘要和轮询使用 `/api/v1/quant/status?include_research=false`，研究和数据状态
  只由 `/api/v1/research/status` 提供。轻量状态接口本机测量约 `0.007s`；原先
  15 秒超时的假浏览器路径修正后约 `9.5s` 完成。
- 选股模型原条件重放误用了全局 Axios `30s` 上限，超时后又被统一解释为
  “服务端访问外部依赖失败”。日志证明该请求只使用本机不可变 Qlib provider，
  不访问外部行情。r9 冷启动 Edge 实测耗时 `34.184s`，因此旧客户端必然会先
  取消。现在该接口使用独立 `120s` 等待契约；若仍超时，错误类别为
  `local_operation_timeout`，明确说明是本地回测等待超时，不再引导用户检查
  外部网络和代理。两项行为先由 3 个失败断言复现，再修到聚焦测试
  `10 passed`。

r9 真人证据：

- 前端构建：`build-20260803-174815Z`。
- 后端构建：`localquant-ui-training-20260804-r9`；接口契约：
  `quant-workbench-v2`。
- Edge 重放 Run `4be0a4ea8067402a8fac3d492f4d1fc7` 成功，使用 snapshot
  `qlib-ashare-v2-20260802T021409-28d2dd81` 和 Qlib 原始组合回测引擎。
- 策略总收益 `75.56%`、沪深300总收益 `37.80%`、扣成本后年化超额
  `10.89%`、最大回撤 `-20.46%`；重放指标与冻结 Run 的最大绝对误差为
  `7.77e-16`。页面未出现“外部依赖”错误。
- r7 残留进程组 `79587` 已按精确身份终止。关闭 r8 的 screen 后，实际 DSA
  子进程组仍被孤立并继续监听；随后按已核对进程组终止。现在 8010 只有 r9
  一个监听服务，6006 的 MLflow 主进程和四个 worker 未停止、未重启。

### 当前仍未完成

- 最终 diff 的主后端、quant_workflow、前端全量测试、lint、构建和全部浏览器
  套件仍需重新运行。
- `/experiments/`、Edge 临时移动视口和最终控制台检查仍需在 r9 或后续最终构建
  上完成。
- 新 Codex 子 Agent、OpenCode CLI、Claude Code 尚未依次审查同一最终 diff。
- 在三方均无 P0/P1/P2、全量验证通过之前不得本地提交或标记 Goal 完成。

## 2026-08-04 r10 最终候选验收

r9 后的全量门禁暴露了一项此前未稳定复现的状态陈旧问题：完整
`/api/v1/quant/status` 会在请求开始时读取任务状态，同时等待较慢的研究快照。
单独回归用例实测等待 `11.59s` 后仍返回旧 `running`，而子进程早已完成。现在
研究快照和服务探测结束后才读取可变任务状态；确定性异步回归测试在旧实现下以
`active_count=1` 失败，修正后相关 5 项测试全部通过。该改动不改变任务状态
定义，也不降低数据或训练门槛。

当前最终候选服务和自动化证据：

- DSA 后端构建为 `localquant-ui-training-20260804-r10`，API 契约为
  `quant-workbench-v2`；`8010` 只有一个监听进程，活动量化任务为 0。
- MLflow 继续由原 `6006` 服务提供，DSA 重启时未停止或重启 MLflow。
- 主后端门禁：离线集 `3370 passed, 2 deselected`；完整集
  `3370 passed, 2 skipped`。
- `quant_workflow`：`1111 passed`。
- 前端：78 个测试文件中 `765 passed, 2 skipped`；ESLint 和生产构建通过。
- 数据治理 Playwright：`2 passed`；隔离 fake-analysis 工作台：`36 passed`；
  本机 8010 工作台：`35 passed, 1 skipped`，跳过项仅限 fake-analysis 专用
  提交路径；MLflow 双语与本地 Run/筛选/artifact：`3 passed`。
- Edge 在 r10 上重新检查 `/workflow`、`/data` 和带冻结 Run 注入的
  `/backtest`。三页实际视口均为 `390 x 844`，内容宽度均为 `384`，没有横向
  溢出；回测页加载 Run `4be0a4ea8067402a8fac3d492f4d1fc7`、snapshot 和
  “按原条件重放”入口。临时视口随后恢复为 `1720 x 849`。
- `git diff --check` 通过；测试未产生新的待提交数据、报告或浏览器产物。

三方只读审查尚未执行，因此本节仍是“最终候选”，不是已提交版本。后续必须让
Codex 子 Agent、OpenCode CLI 和 Claude Code 依次检查同一工作树；任一 P0、
P1、P2 或阻塞问题都要修正并重新全量验收、重新三方审查。审查通过前不得 commit、
不得标记 Goal 完成；本轮始终不 push。

## 2026-08-04 r11 durable runner 取消与启动握手续修

r10 后的 Codex 独立审查发现两项会影响真实长训练控制的阻塞问题，均已先增加
失败回归测试再修正：

1. launchd 接管鲁棒信号训练后，网页启动任务会先进入 `completed`，旧 API 和
   页面因此不再允许取消仍在运行的后台 runner。现在任务响应增加 `can_cancel`；
   只有配置了服务端白名单取消命令，并且同一 Run 的进度仍为 `starting`、
   `running` 或 `pause_requested` 时，终态启动器才保持可取消。自然完成、取消
   完成和重复取消不会再次调用取消命令；取消命令未确认停止时页面保留错误和
   取消入口，不能假装训练已经停止。
2. API 若恰好在 `runner_identity.json` 原子写入期间重启，旧恢复逻辑可能把合法
   runner 判为失败并释放输出冲突锁。现在启动握手严格绑定 launch token、PID、
   任务命令哈希、进程启动身份和实际进程命令哈希，并做有界等待；重启时身份仍
   待写入会继续保留冲突锁，允许临时记录接纳带签名的子 PID。瞬时 `ps` 失败只
   延迟判断，不能凭 PID 放行；握手超时只停止本次精确匹配的新进程。快速任务在
   wrapper 退出后还会做一次最终严格读取，避免身份刚落盘却被误判。

r11 当前验证证据：

- `8010` 已部署为 `localquant-ui-training-20260804-r11`，只有一个监听进程，
  API 契约为 `quant-workbench-v2`，活动量化任务为 0；`6006` 的 MLflow 未停止、
  未重启。
- 主后端完整测试为 `3377 passed, 2 deselected`；`quant_workflow` 为
  `1111 passed`；前端为 `767 passed, 2 skipped`。ESLint、生产构建、Python
  语法门禁和 `git diff --check` 通过。当前环境未安装 `flake8`，因此没有把
  flake8 记成已通过。
- 隔离 managed Playwright 全路径为 `36 passed`，耗时约 `9.4m`。测试使用临时
  reports 和假分析后端，不覆盖正式训练 latest、候选结果或 MLflow。
- r11 的本机 8010 Playwright 为 `35 passed, 1 skipped`，唯一跳过项是只允许
  fake-analysis 后端执行的假分析提交；MLflow 双语、Run 筛选和 Artifact 回归为
  `3 passed`。测试结束后 API 仍为 `active_count=0`、`trainable=true`，没有
  残留训练进程，正式 provider snapshot 未变化。
- Edge 在正式 8010 页面把 LightGBM 数据版本切换为 cleaned，仅改变表单，没有
  启动任务。随后从训练结果点击“回测”，页面自动带入 Run
  `4be0a4ea8067402a8fac3d492f4d1fc7`、选股模型 `baseline_lgb_alpha158`、交易
  策略“每日 TopK 等权基线”和 snapshot
  `qlib-ashare-v2-20260802T021409-28d2dd81`。
- Edge 另在隔离临时 reports/任务库中复现“启动器已 completed、后台状态仍
  running”。页面显示可用的“立即取消”，点击后出现“这不是可恢复的暂停”的
  二次确认；选择“继续运行”关闭对话框，没有发送取消请求。隔离 18012 服务和
  仓内临时目录随后清理，正式任务库没有新增这条测试记录。

当前仍未完成最终三方独立审查和本地 commit。后续若最终 diff 发生实质变化，
必须重跑受影响测试，并让 Codex 子 Agent、OpenCode CLI、Claude Code 重新审查
同一份最终工作树；三方通过前 Goal 保持 `active`，不 push。

## 2026-08-04 r12 状态一致性与回测目录续修

r11 后的真人页面与实际目录核对又暴露三项只能在运行路径中稳定看见的问题：

1. launchd 已接管训练、启动器任务已经 `completed` 但仍返回 `can_cancel=true` 时，
   后端契约表示后台 runner 仍在运行。旧页面只在取消按钮局部承认这一点，顶部
   运行计数、任务分组、重复启动守卫和鲁棒控制区仍把它当成已结束。现在所有这些
   位置统一使用同一个 active durable runner 判断，并显示“后台仍运行”。
2. `/backtest` 已经挂载后，如果训练页或浏览器历史只改变同一路由的 `runId`、
   `modelId`、`strategyId` 查询参数，旧组件只在首次加载时读取参数。现在查询变化
   会重新注入 Run 和冻结参数，清除上一 Run 的结果和错误；请求版本守卫会丢弃
   已切换 Run 后才返回的旧回测结果。前端单元测试与真实 Playwright
   `history.pushState + popstate` 都覆盖这一行为。
3. 实际回测目录包含 8 个旧候选 Run，它们虽有真实 Run ID，但
   `provider_identity={}`、`strategies=[]` 且 `replay_capability=unavailable`。旧页面
   仍把它们列为可选模型，用户随后只会看到空交易策略和无解释的禁用按钮。现在
   只有 exact 或 custom 回测契约完整的 Run 进入选择框；被排除的历史 Run 数量与
   缺失原因仍在页面可见。当前目录中 3 个可回测 Run 保留，8 个缺契约旧 Run 被
   分离；其中 `custom_only` 且身份/策略完整的旧 Run 仍可自定义回测。

r12 当前服务与自动化证据：

- `8010` 后端构建为 `localquant-ui-training-20260804-r12`，API 契约为
  `quant-workbench-v2`；当前活动量化任务为 0。前端最终构建时间为
  `2026-08-04T07:01:08.765Z`，对应 `build-20260804-070108Z`。`6006` 的 MLflow
  未因本轮验证停止或重启。
- 主后端门禁第一阶段为 `3377 passed, 2 deselected`，第二阶段为
  `3377 passed, 2 skipped`。`test_web_bind` 的 subprocess 上限从 20 秒改为
  60 秒：该测试验证绑定安全行为而非启动性能，隔离运行约 12 秒即可通过，旧上限
  在全套负载下产生过一次假超时；安全断言没有放宽。
- `quant_workflow` 为 `1111 passed`。前端 78 个测试文件为
  `770 passed, 2 skipped`；ESLint 和生产构建通过。
- 数据治理 Playwright 为 `2 passed`；隔离 managed fake-analysis 工作台为
  `37 passed`；本机 8010 工作台为 `36 passed, 1 skipped`，唯一跳过项仍是只在
  fake-analysis 后端开放的假分析提交；MLflow 双语、业务数据保真、Run 筛选和
  Artifact 为 `3 passed`。
- 浏览器 smoke 实际走过数据检查、清洗预检、质量报告、异常报告、任务取消、
  产物预览、回测模式切换和同页 Run 注入。managed 产物使用隔离 reports；本机
  任务均为受控 smoke/dry-run。本轮没有启动正式 Qlib 训练、鲁棒长训练或不受控
  长回测，正式候选 latest、正式 MLflow 实验和 provider snapshot 未被测试替换。
- 数据状态仍为 `trainable=true`；canonical snapshot 为
  `qlib-ashare-v2-20260802T021409-28d2dd81`，cleaned snapshot 为
  `qlib-ashare-cleaned-v2-20260802T021735-1d6989e6`。历史等价验收 Run 继续是
  网页 `e144ddd980514280ad2e07b6eae68dae`、脚本
  `91951f0af26143d3ab9f4169a2028d7b`；正式原条件重放证据 Run 继续是
  `4be0a4ea8067402a8fac3d492f4d1fc7`。

当前最终 diff 尚未完成 Edge 人工桌面/移动复查、三方独立只读审查和本地 commit。
这些项目完成前，本节仍是提交候选证据，Goal 必须保持 `active`；任何审查 P0、
P1、P2 或阻塞问题都必须修复并重新让三方审查同一份最终工作树。本轮不 push。

## 2026-08-04 r13 移动布局与加载语义复验

r12 后继续按人类用户路径检查，补齐两项静态审查不容易发现的页面问题：

1. `/workflow` 的训练结果表最小宽度为 `780px`，外层 CSS Grid 使用隐式自动列时，
   表格会把整个页面撑到约 `810px`。在 `390 x 844` 视口中只能看到左半页。当前
   基础网格轨道改为 `minmax(0,1fr)`，宽表只在自身容器内横向滚动，不再制造页面级
   溢出；Playwright 已固定这一回归。
2. 数据和任务状态请求尚未返回时，旧页面会同时显示“正在读取数据状态”和
   “数据验收：阻塞”，把未知状态误报为失败。当前加载期间只显示中性的
   “数据验收：读取中”；请求完成且 `trainable=false` 后才显示“阻塞”，并有前端
   单元测试覆盖未决 Promise。

r13 当前服务、自动化和 Edge 证据：

- `8010` 后端仍为 `localquant-ui-training-20260804-r12`，API 契约仍为
  `quant-workbench-v2`；本轮没有修改或重启 `6006`。Edge 实际加载的前端为
  `build-20260804-115639Z`。
- 前端单元测试为 `771 passed, 2 skipped`，ESLint 和生产构建通过；本机 8010
  Playwright 为 `37 passed, 1 skipped`，隔离 managed fake-analysis Playwright
  为 `38 passed`。主后端、`quant_workflow`、文档、数据治理和 MLflow 的上一轮
  最终 diff 证据分别保持 `3377 passed, 2 skipped`、`1111 passed`、
  `16 passed`、`2 passed` 和 `3 passed`；文档修改后仍需重新运行文档门禁与
  `git diff --check`。
- Edge 桌面路径核对了 Run `4be0a4ea8067402a8fac3d492f4d1fc7` 的选股模型、
  `baseline_daily` 交易策略和 snapshot
  `qlib-ashare-v2-20260802T021409-28d2dd81` 注入。回测目录显示 3 个可选 Run，
  8 个缺少冻结契约的旧 Run 被排除但原因可见；切换到
  `7de3e751dfb64e0591e82affb7c5e6bb` 后页面正确显示 `custom_only` 并禁用
  原条件重放。
- Edge 临时 `390 x 844` 视口下，`/workflow`、`/data`、`/backtest` 的文档宽度
  均为 `384/384`，没有页面级横向溢出或被裁掉的操作按钮。数据页复选框在轮询时
  保持选中；检查后已恢复原值。
- 第一轮本机 Playwright 的 JQData dry-run 曾在 60 秒上限出现一次超时。任务清理
  后没有残留 PID；相同用例隔离重跑为 `2.4s` 通过，第二次完整套件为 `2.2s`
  通过。因此当前证据是一次瞬时波动，不是稳定复现的产品故障，也没有放宽生产
  超时或数据门槛。

当前工作树仍未取得 Codex 子 Agent、OpenCode CLI、Claude Code 对同一最终版本的
三份干净审查，也尚未本地 commit。三方审查通过前 Goal 保持 `active`；本轮不 push，
不启动正式训练、长回测、券商连接或自动下单。

## 2026-08-04 r14 独立审查三项竞态与数据绑定修复

r13 后的第一份全新 Codex 只读审查发现 3 个 P2，因此该轮审查判定失败，OpenCode
和 Claude 没有继续启动。三项均按红绿回归修复：

1. API 重启后接管的持久 runner 如果超过恢复监控窗口，旧逻辑会把数据库任务改成
   `failed`，但已核对身份的进程可能仍在运行，产物冲突锁随之被释放。现在 runner
   存活或签名身份仍处于可信启动握手时，超时只更新可见告警，任务保持 `running`、
   产物占用继续生效，并重新挂接恢复监控；不能因监控窗口到期伪造进程已经结束。
2. 用户点击取消与任务自然完成可能同时发生。现在取消器在发送 `bootout` 前和确认
   进程消失后检查 Run 终态：取消信号发出前已经完成的 Run 保留真实终态，不改写为
   `cancelled`；`bootout` 已成功后则由用户取消优先，不能把信号造成的退出误报成
   自然失败。Run 状态和 launch 记录使用同一裁决结果。
3. 市场状态和现金仓位报告原本会核对质量报告、catalog、LATEST、manifest 与内容
   哈希身份，但服务提交前没有再次绑定实际快照目录。现在质量验收返回其完成完整
   内容复核时保存的文件系统状态指纹，任务服务在拼接内部 bars 路径前重新计算并
   比较该低成本指纹。快照在批准后发生新增、替换、删除或改写时立即失败关闭；不会
   为此重复读取和哈希全部行情内容。

新增回归覆盖两个恢复超时分支、产物冲突仍被占用、自然完成先于取消、成功取消先于
失败状态以及批准后 bars 被改写。当前受影响后端文件为 `248 passed`，鲁棒启动器
文件为 `21 passed`，`git diff --check` 通过。全量门禁和三方审查必须在 r14 最终
工作树上重新执行；之前任何审查结论都不能复用。

r14 最终自动化与运行路径证据：

- 主后端门禁第一阶段为 `3379 passed, 2 deselected`，第二阶段为
  `3379 passed, 2 skipped`；`quant_workflow` 为 `1114 passed`。前端 78 个测试
  文件为 `771 passed, 2 skipped`，ESLint 和生产构建均通过。
- 数据治理 Playwright 为 `2 passed`；隔离 managed fake-analysis 工作台为
  `38 passed`；本机 8010 工作台为 `37 passed, 1 skipped`，唯一跳过项仍是只在
  fake-analysis 后端开放的假分析提交；MLflow 双语、业务数据保真、Run 筛选和
  Artifact 为 `3 passed`。
- `8010` 已部署为 `localquant-ui-training-20260804-r15`，只有一个监听进程，接口
  契约为 `quant-workbench-v2`，活动量化任务为 0；Edge 实际加载的前端为
  `build-20260804-163244Z`。`6006` 的 MLflow 监听 PID 组未变化，本轮没有停止或
  重启 MLflow。
- Edge 桌面路径从训练结果的“回测”进入，Run
  `4be0a4ea8067402a8fac3d492f4d1fc7`、`baseline_lgb_alpha158` 选股模型、
  `baseline_daily` 交易策略和 snapshot
  `qlib-ashare-v2-20260802T021409-28d2dd81` 均正确注入。数据页在请求未完成时显示
  “读取中”，完成后显示“验收通过”，没有把未知状态误报为阻塞。
- Edge 临时 `390 x 844` 视口下，`/workflow`、`/data`、`/backtest` 的文档宽度
  均为 `384/384`，没有页面级横向溢出。训练结果表只在自身容器内横向滚动，向右
  滚动后“回测”操作可见；检查结束后已恢复 Edge 正常视口。JQData dry-run 本轮在
  `3.6s` 完成，没有重现 r13 的瞬时超时。

三方独立审查和本地 commit 仍未完成。Codex 子 Agent、OpenCode CLI、Claude Code
必须严格按顺序审查上述同一份最终工作树；任一方发现 P0、P1、P2 或阻塞问题，都要
修复并从 Codex 审查重新开始。三方全部通过前 Goal 保持 `active`，本轮不 push。

## 2026-08-05 r16 启动租约、取消裁决和状态请求仲裁修复

r15 后的新 Codex 独立审查发现 1 个 P1 和 3 个 P2，因此该轮审查判定失败，未把
旧审查结论继续传递给 OpenCode 或 Claude。四项均先增加失败回归测试，再完成修复：

1. 持久 runner 在写入签名身份前崩溃时，旧任务可能长期停在 `starting` 并永久
   占用产物锁。现在启动器持久化 `startup_deadline_epoch`；wrapper 在写身份前和
   执行真实命令前都检查启动租约。无 PID、无签名身份且租约已过期的任务会明确
   失败并释放产物锁，迟到的 wrapper 不能再执行训练命令。
2. 取消请求与 runner 自然完成同时发生时，旧服务可能把已经成功完成的运行覆盖为
   `cancelled`。现在受信取消脚本返回显式的 `cancel_result_contract` 和
   `cancel_outcome`；`already_terminal` 保留自然完成状态，缺失或非法的取消输出
   视为取消失败，不能修改任务终态。
3. 网页训练与脚本训练的等价验收此前对证据结构检查不够严格。现在 provider
   identity 必需字段必须完整，配置、manifest、内容、预测和标签哈希必须是合法
   64 位 SHA-256，metrics 必须是非空、有限且真实的数值映射。空对象、普通字符串
   哈希、`NaN` 和字符串形式数值均不能通过验收。
4. `/workflow` 的首次加载、30 秒刷新、3 秒 runner 轮询和 SSE 后刷新可能并发，
   较旧的研究状态响应会覆盖较新状态。现在四条路径共用独立的研究状态请求版本
   仲裁；旧响应和旧错误均不能回滚新状态。

新增和受影响回归分别验证过期 ownerless startup 释放锁、迟到 wrapper 不执行真实
命令、`already_terminal` 保留自然终态、非法取消输出失败关闭、非法等价证据不能
通过，以及旧研究状态响应不能回滚新状态。受影响后端文件为 `145 passed`，相关
`quant_workflow` 文件为 `12 passed`，`WorkflowPage` 为 `79 passed`；Ruff、
TypeScript 和 ESLint 均通过。

r16 最终候选的自动化与真人网页证据：

- 主后端门禁第一阶段为 `3383 passed, 2 deselected`，第二阶段为
  `3383 passed, 2 skipped`；`quant_workflow` 为 `1116 passed`。前端 78 个测试
  文件为 `772 passed, 2 skipped`，ESLint 和生产构建均通过。
- 数据治理 Playwright 为 `2 passed`；隔离 managed fake-analysis 工作台为
  `38 passed`；本机 8010 工作台为 `37 passed, 1 skipped`，唯一跳过项仍是只在
  fake-analysis 后端开放的假分析提交；MLflow 双语、业务数据保真、Run 筛选和
  Artifact 为 `3 passed`。
- `8010` 已部署为 `localquant-ui-training-20260805-r16`，API 契约为
  `quant-workbench-v2`，活动量化任务为 0；Edge 实际加载的前端为
  `build-20260804-195417Z`。旧 r15 进程与一次绑定失败的 r16 进程均按核对身份
  温和结束，当前只有一个 8010 监听服务。`6006` 的 MLflow 监听 PID 组未变化，
  本轮没有停止或重启 MLflow。
- Edge 重新加载后先显示中性的“数据验收：读取中”，随后稳定显示“数据验收：通过”；
  没有出现旧状态回滚或前后端构建不一致告警。训练结果的“回测”入口正确注入 Run
  `4be0a4ea8067402a8fac3d492f4d1fc7`、选股模型
  `baseline_lgb_alpha158`、交易策略 `baseline_daily` 和 snapshot
  `qlib-ashare-v2-20260802T021409-28d2dd81`。
- Edge 临时 `390 x 844` 视口下，`/workflow`、`/data`、`/backtest` 的文档宽度
  均为 `384/384`，没有页面级横向溢出；回测页的 Run、交易策略和加载控件可见。
  检查结束后已恢复为 `1720 x 896`。Edge 控制台没有来自 DSA 页面的错误；记录到
  的警告均来自已安装浏览器扩展。
- 所有浏览器测试结束后 API 仍为 `active_count=0`，没有残留正式训练 runner。
  本轮只执行白名单 smoke、dry-run 和隔离假分析，没有启动正式训练、长回测、券商
  连接或自动下单，也没有替换正式 provider snapshot 或 MLflow 候选结果。

上述证据完成后仍必须让一个全新的 Codex 子 Agent、OpenCode CLI 和 Claude Code
严格按顺序审查同一份最终工作树。任一方发现 P0、P1、P2 或阻塞问题，都必须修复并
从 Codex 审查重新开始。三方全部明确无阻塞问题前不得本地 commit、不得标记 Goal
完成；本轮不 push。

## 2026-08-05 r17 前视、回测来源保护和产物预览竞态修复

r16 后的新 Codex 独立审查发现 1 个 P1 和 2 个 P2，因此该轮审查判定失败，未把
旧审查结论继续传递给 OpenCode 或 Claude。三项均先增加失败回归，再完成修复：

1. 波动率仓位控制原本使用向前一期的 benchmark return 计算历史波动率，导致
   T 日仓位读取了 T+1 收益。现在历史波动率统一从当日及过去收盘价的
   `close.pct_change()` 计算；向前 benchmark return 只保留给收益归因，不能进入
   仓位决策。
2. `/api/v1/backtest/*` 的 POST 请求原本没有纳入严格来源校验。现在与量化任务、
   持仓和实验写接口相同，无本机 `Origin`/`Referer` 的浏览器式写请求返回
   `403 csrf_forbidden`；带本机来源后才进入参数校验和回测服务。
3. 用户快速切换任务日志或产物时，较慢的旧产物预览响应可能覆盖新选择。现在产物
   请求使用独立版本号仲裁；打开日志也会使在途产物请求失效，旧成功和旧错误都不能
   回写当前面板。

r17 候选的自动化、部署和真人路径证据：

- 受影响后端回归为 `72 passed`，候选流程为 `18 passed`，`WorkflowPage` 为
  `80 passed`。主后端门禁第一阶段为 `3385 passed, 2 deselected`，第二阶段为
  `3385 passed, 2 skipped`；`quant_workflow` 为 `1117 passed`；前端为
  `773 passed, 2 skipped`。ESLint、生产构建和 Ruff 严重错误规则
  `E9,F63,F7,F82` 通过。默认全量 Ruff 仍报告 230 个历史存量问题，本轮没有借机
  扩大范围清理。
- 隔离 managed fake-analysis 工作台为 `38 passed`；数据治理为 `2 passed`；本机
  `8010` 工作台顺序运行结果为 `37 passed, 1 skipped`，唯一跳过项仍是只在隔离
  fake-analysis 后端开放的假分析提交；MLflow 双语、业务数据保真、Run 筛选和
  Artifact 为 `3 passed`。
- 四组浏览器套件并发时，昂贵的 `/api/v1/research/status` 扫描曾延长到约 25 秒，
  页面仍处于真实加载状态但 5 秒测试等待产生假超时。顺序请求约为 2.8 至 6.1 秒；
  数据治理断言改为明确的 15 秒等待，产品接口、训练门槛和失败语义没有放宽。所有
  最终浏览器证据均按顺序运行，避免把本机争用误判成产品故障。
- `8010` 当前部署为 `localquant-ui-training-20260805-r17`，前端为
  `build-20260804-210739Z`，API 契约为 `quant-workbench-v2`，只有一个监听进程；
  `6006` 的 MLflow PID 组未变化，本轮没有停止或重启 MLflow。
- Edge 旧标签页在服务重启后仍保留 r16 的内存状态；重新加载后正确显示 r17，
  数据验收由“读取中”稳定变为“通过”，运行按钮恢复可用。人工从训练结果点击
  “回测”后，Run `4be0a4ea8067402a8fac3d492f4d1fc7`、选股模型
  `baseline_lgb_alpha158`、交易策略 `baseline_daily` 和 snapshot
  `qlib-ashare-v2-20260802T021409-28d2dd81` 均自动注入回测页；没有触发重新训练
  或回测计算。Edge 桌面布局无重叠；`390 x 844` 页面级溢出由本轮顺序
  Playwright 回归实际检查通过，Edge 扩展本轮未提供改写真实窗口尺寸的接口，故不把
  移动宽度记作新的 Edge 人工证据。
- 所有浏览器测试结束后没有 Playwright、正式训练或鲁棒研究 runner 残留，`8010`
  仍只有一个监听服务。本轮只执行白名单 smoke、dry-run 和隔离测试，没有启动正式
  训练、长回测、券商连接或自动下单，也没有替换正式 provider snapshot 或正式
  MLflow 候选结果。

r17 仍是未提交候选。必须让一个全新的 Codex 子 Agent、OpenCode CLI 和 Claude
Code 严格按顺序审查上述同一份最终工作树；任一方发现 P0、P1、P2 或阻塞问题，都
必须修复并从 Codex 审查重新开始。三方全部明确无阻塞问题前不得本地 commit、不得
标记 Goal 完成；本轮不 push。

## 2026-08-05 r18 数据、启动、检查点和回测契约修复

r17 后的新 Codex 独立审查发现 4 个 P1、3 个 P2 和 1 个 P3，因此该轮审查判定
失败，OpenCode 与 Claude 没有启动。八项均先建立失败回归，再完成修复：

1. Goal 1 不能再用顶层 canonical `trainable` 代替 cleaned provider 验收。
   `provider_quality.cleaned` 必须存在、可训练、无阻塞，并与数据契约中的四项
   cleaned 身份一致。
2. 任务提交与启动改成共享 SQLite 上的原子 `pending -> running` 认领。耐久
   wrapper 先写签名身份并等待一次性启动许可；取消先赢时不会执行真实命令，授权
   先赢时 PID 和进程组已经登记，可准确终止。
3. Goal 1 恢复不再信任可变绝对路径或空文件。检查点绑定 Run 内相对路径、哈希、
   大小、任务身份、时间切分、行数和 schema；预测与 diagnostics 会安全解析，
   模型只核对路径、哈希和模型契约，不为校验而反序列化 pickle。
4. 训练 artifact 的哈希从所选模型条目读取，并覆盖 baseline exact/custom 与
   candidate custom。服务读取每个 pickle 一次，验证同一份字节后再从内存反序列化，
   消除“验证路径后重新打开”的替换窗口。
5. 候选 DSA 交易策略统一标为 `custom_only` / “重建回测”。历史引擎没有冻结时，
   不再提供或声称“按原条件重放”。
6. 候选子训练持久化 `candidate_child` scope 和 MLflow tag。baseline 目录跳过
   子记录，候选报告用真实模型 ID 只贡献一条记录，消除重复 Run 和冲突身份。
7. 回测深链只有完全没有 Run/模型身份时才默认第一项。明确指定但不存在或不匹配的
   Run、模型、交易策略会清空选择、旧结果和在途请求，并显示错误，不再静默切换。
8. Goal 1 的预览、单季度测试、正式启动、协作暂停和恢复按钮都遵从后端任务目录；
   目录未加载、后端版本不兼容或任务未启用时显示具体不可用状态并禁止提交。

本轮还固定了审查边界：外部端口被封、上游数据源拒绝连接或本机网络不可达是环境
状态，本身不算 DSA Bug，也不能作为最终审查阻塞项；DSA 误报成功、损坏数据、无限
等待或无法恢复仍属于软件缺陷。不得为了外部连接失败降低数据门槛。

r18 提交候选的当前自动化证据：

- 跨进程任务服务 `148 passed`；Goal 1 数据与检查点相关文件 `53 passed`；候选
  回测契约 `76 passed`；研究状态 API `107 passed`；前端关键页 `93 passed`。
- 主后端门禁第一阶段为 `3394 passed, 2 deselected`，第二阶段为
  `3394 passed, 2 skipped`；`quant_workflow` 为 `1133 passed`。
- 前端 78 个测试文件为 `776 passed, 2 skipped`；ESLint 和生产构建通过。
- 最终部署与浏览器证据写入后，文档门禁为 `16 passed`，Ruff
  `E9/F63/F7/F82` 和 `git diff --check` 均通过。

r18 部署与浏览器顺序验收证据：

- `8010` 已从 r17 温和重启为 `localquant-ui-training-20260805-r18`，前端构建为
  `build-20260805-061613Z`，接口契约为 `quant-workbench-v2`，只有一个监听进程且
  `active_count=0`。重启前后 `6006` 的 MLflow 监听 PID 组
  `95362/95368/95369/95370/95371` 完全不变。
- 数据治理浏览器回归为 `2 passed`。当前 `DATA_MANIFEST` 与上次数据盘点不一致，
  页面正确显示“已过期”和“请重新扫描”；E2E 不再把动态环境硬编码为“可用”，但在
  已过期状态下必须看到重扫提示，不能掩盖数据状态。
- 隔离 managed fake-analysis 工作台为 `38 passed`；正式 `8010` 工作台为
  `37 passed, 1 skipped`，唯一跳过的是只允许隔离假后端运行的假分析提交；
  MLflow 双语、业务数据保真、Run 筛选和 Artifact 为 `3 passed`。四组套件严格
  串行执行，避免并发状态扫描造成假超时。
- Edge 真人路径显示 canonical 与 cleaned 结构验收通过、当前基线数据可训练、前后端
  构建一致。普通 LightGBM 训练入口可用；严格鲁棒信号研究因 provider 绑定旧数据
  清单而禁用，并明确要求重新验收准备，没有把普通基线资格误当作 Goal 1 资格。
- Edge 从训练结果点击“回测”后，Run `4be0a4ea8067402a8fac3d492f4d1fc7`、选股模型
  `baseline_lgb_alpha158`、交易策略 `baseline_daily` 和 snapshot
  `qlib-ashare-v2-20260802T021409-28d2dd81` 全部注入；本轮没有点击执行重放。
  改为不存在的 Run 深链后，页面显示明确错误、两个选择框复位、加载按钮禁用，旧的
  历史指标和结果不再残留，也没有静默回退到第一条 Run。
- Edge 临时 `390 x 844` 视口下页面宽度为 `384/384`，无页面级横向溢出。训练结果
  表的回测按钮位于自身 `326 -> 780` 的横向滚动容器内，这是可操作的表格滚动而非
  控件越界。临时视口已经恢复；控制台没有来自 `127.0.0.1:8010` 的 error/warn，
  记录到的警告全部来自已安装浏览器扩展。
- 本轮浏览器验收只执行白名单 smoke、dry-run、隔离报告和表单/深链检查，没有启动
  正式训练、鲁棒长任务、长回测、券商连接或自动下单，也没有替换正式 provider、
  MLflow 候选或正式 `latest` 结果。

r18 仍未完成 Codex 子 Agent、OpenCode、Claude Code 三方独立审查和本地 commit。
三方全部通过前 Goal 保持 `active`；本轮不 push。

## 2026-08-05 r19 训练成本、恢复语义与启动失败修复

r18 后的 Codex 独立只读审查发现 5 个 P2。五项均已增加能够在修复前失败的回归
断言，再完成实现修正：

1. 优化交易策略不再只按持仓集合变化估算换手。成本以调仓前后真实组合权重差
   计算，覆盖部分减仓、清仓、现金重新入场以及持仓与权重同时变化。
2. Goal 1 从检查点恢复时复用已冻结的运行上下文，`smoke` 运行不会被恢复成正式
   研究，也不会写入正式研究状态。
3. 任务服务恢复 `smoke` 参数时统一使用布尔解析，字符串 `"true"` 与布尔值
   `true` 的行为一致。
4. 两个耐久启动器在 warm-up 或实际子进程失败时向任务服务返回非零退出码，网页
   不再把训练失败收口为“任务完成”。
5. 候选模型 pickle 只读取一次；同一份内存字节依次完成 512 MB 大小限制、
   SHA-256 与来源校验和反序列化，消除校验后重新打开文件的替换窗口。

新增或更新的八个回归断言均已完成红绿验证。r19 最终候选的验收证据为：

- 相关模块 `272 passed`。
- 主后端离线门禁 `3394 passed, 2 deselected`，完整套件
  `3394 passed, 2 skipped`。
- `quant_workflow` 为 `1142 passed`。
- 前端为 `776 passed, 2 skipped`；ESLint 和生产构建通过。
- 数据治理浏览器为 `2 passed`；隔离 managed 工作台为 `38 passed`；本机
  `8010` 工作台为 `37 passed, 1 skipped`；MLflow 浏览器为 `3 passed`。
- Ruff 严重错误规则 `E9/F63/F7/F82` 和 `git diff --check` 通过。

`8010` 当前部署为 `localquant-ui-training-20260805-r19`，前端构建为
`build-20260805-113236Z`，接口契约为 `quant-workbench-v2`，只有一个监听进程，
`active_count=0`。`6006` 的 MLflow 没有重启或修改。Edge 真人检查确认普通
canonical/cleaned 快照训练入口可用，严格鲁棒 CSI800 研究仍因绑定旧数据清单而
正确阻止；本轮没有启动正式长训练。

## 2026-08-05 OpenCode 最终独立审查

OpenCode 对 r19 同一工作树完成独立只读审查，结论为 `PASS`，没有 P0、P1、P2
或其他阻塞问题。审查遵守以下边界：端口被封、上游数据源拒绝连接或本机网络不可达
是环境事件，本身不算 DSA Bug；只有 DSA 误报成功、发布部分或损坏数据、无限挂起、
重试/恢复失效或降低质量门槛时才算软件缺陷。

OpenCode 记录了六项非阻塞 P3/残余风险，等待与最终 Codex 子 Agent 结论合并后
统一决定是否修改：

1. 未加载 Run 时，回测日期、TopK 和成本输入框仍可编辑；执行按钮本身已禁用。
2. MLflow 链接只进入实验首页，没有直接打开指定 Run 的深链。
3. SSE 遇到畸形 JSON 或缺少 `jobs` 数组时静默返回空列表，缺少诊断警告。
4. 研究状态刷新失败时保留旧 `researchStatus`，页面没有明确标记其已过期。
5. 任务列表逐行计算进度、runner 身份和日志时间，任务很多时存在额外
   `ps`、glob 和 `stat` 开销。
6. 通用 `_update_job` 依赖当前单进程与 `RLock` 保证状态一致；若未来改为多 worker，
   关键更新需要增加 SQL 状态守卫。

## 最终审查政策变更

用户在 OpenCode 审查完成后明确取消后续 OpenCode 和 Claude Code 审查，只保留一个
最终 Codex 子 Agent 审查。因此，本文此前要求“三方全部复审”的段落保留为历史
过程记录，但不再是当前完成门槛。当前门槛为：等待正在运行的 Codex 子 Agent 对
r19 同一工作树给出独立结论；若发现 P0、P1、P2 或阻塞问题，完成修复和复审；若
明确无阻塞问题，则完成最终代码卫生检查和本地 commit。仍然不 push。

## 2026-08-05 Codex 最终审查与 r20 恢复完整性修复

最终 Codex 子 Agent 第一轮只读审查发现 1 个 P2：Goal 1 恢复只比较
`context.json` 自报的 `context_hash` 字符串，没有根据文件实际内容重新计算。攻击者
或损坏文件可以保留旧哈希并替换 `provider_uri`，使同一 Run 的后续任务读取另一套
provider。

r20 先增加“替换 provider 但保留旧哈希”的失败回归，再完成以下窄修复：

1. 新建 Run 与恢复统一使用 `_immutable_context_hash`，按相同 canonical JSON 规则
   计算不可变上下文哈希。
2. `context_hash` 自身以及 `run_mode`、`smoke_quarter`、`smoke_signal_id` 是运行控制
   字段，不参与不可变上下文哈希；后三项仍由独立契约严格校验。
3. 恢复时先按 `context.json` 实际内容重算并校验 stored hash，再与刚重新完成数据
   验收的 fresh context hash 比较。保留旧哈希篡改内容时报
   `resume_context_integrity_mismatch`；篡改内容并重算哈希时仍会报
   `resume_context_mismatch`。

修复后的聚焦回归为：Goal 1 研究文件 `41 passed`、信号研究 launcher
`7 passed`、网页恢复接口聚焦用例通过；完整 `quant_workflow` 为
`1143 passed`，`git diff --check` 通过。

同一 Codex 子 Agent 随后完成只读复审，明确确认原 P2 已关闭，最终结论为
`PASS`：P0、P1、P2 均为零。没有把端口、上游连接或本机网络事件列为 DSA 缺陷，
也没有建议降低数据质量门槛。

## 非阻塞后续清单

OpenCode 与 Codex 子 Agent 合计记录 8 项 P3。它们不影响当前单机网页训练闭环的
正确性，也没有误报成功、数据污染或恢复绕过，因此不在最终提交前继续扩大改动：

1. 未加载 Run 时禁用回测日期、TopK 和成本输入框，减少无效编辑。
2. MLflow 链接增加指定实验与 Run 的深链。
3. SSE 畸形事件增加诊断警告，不再静默返回空任务列表。
4. 研究状态刷新失败时清除旧状态或明确标记为过期。
5. 任务列表的 `ps`、glob 和 `stat` 增加短 TTL 缓存或延迟计算。
6. 只有未来支持多 worker 时，才为通用 `_update_job` 增加 SQL 状态守卫。
7. 研究状态接口进一步隐藏本机 provider 绝对路径；当前接口已受本机和同源门禁。
8. 将同步训练结果回测升级为可取消、有限并发的持久任务；当前 120 秒客户端超时
   不会终止已经开始的服务端计算。

当前最终审查门槛已满足：OpenCode `PASS`，Codex 子 Agent 复审 `PASS`；用户已取消
Claude Code 审查。完成最终文档、Ruff、差异格式和 Git 卫生检查后允许本地 commit，
仍然不 push。
