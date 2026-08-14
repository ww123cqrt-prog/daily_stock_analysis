/**
 * Preset strategy scripts shown in the research workbench dropdown.
 *
 * Every preset is a self-contained DSA Python strategy with detailed
 * Chinese comments, runnable against any universe pool.
 */

export interface PresetScript {
  id: string;
  nameZh: string;
  nameEn: string;
  script: string;
}

export const SAMPLE_SCRIPT = `# ============================================================
# 动量轮动策略（预置脚本）
# ------------------------------------------------------------
# 思路：每个交易日收盘后，在股票池内按过去 N 日涨幅排名，
#       持有排名靠前的标的，并要求收盘价站上自身均线；
#       都不满足时空仓。信号在下一交易日开盘执行。
#
# 可调参数（在 initialize 中修改）：
#   lookback : 动量回看天数（默认 20 个交易日）
#   ma_days  : 均线周期（默认 20 个交易日）
#   top_n    : 持有排名前几名（默认 1）
#
# 常用 API：
#   initialize(context)      回测开始时执行一次，设置参数
#   on_bar(context, data)    每个交易日收盘后执行一次
#   data.rank_by_return(n)   按过去 n 日涨幅给当日池内可交易
#                            成员排名，返回代码列表（降序）
#   data.sma(symbol, n)      某标的过去 n 日收盘价均值
#   data.close(symbol)       某标的当日收盘价
#   data.universe()          当日可交易的池内成员列表
#   context.target_equal([]) 目标仓位：等权持有列表内标的
#   context.target_cash()    目标仓位：全部空仓
# ============================================================

def initialize(context):
    context.lookback = 20
    context.ma_days = 20
    context.top_n = 1


def on_bar(context, data):
    # 1. 按过去 lookback 个交易日的涨幅，对当日池内成员排名
    ranked = data.rank_by_return(context.lookback)
    if not ranked:
        # 池内没有可排名的标的（历史数据不足），空仓
        context.target_cash()
        return

    # 2. 逐个检查排名靠前的标的：收盘价必须站上自身均线
    kept = []
    for symbol in ranked[:context.top_n]:
        average = data.sma(symbol, context.ma_days)
        if average is not None and data.close(symbol) > average:
            kept.append(symbol)

    # 3. 有满足条件的标的就等权持有，否则空仓
    if kept:
        context.target_equal(kept)
    else:
        context.target_cash()
`;

const DUAL_MA_TIMING = `# ============================================================
# 双均线择时策略（预置脚本）
# ------------------------------------------------------------
# 思路：对股票池中第一只标的，比较短期均线与长期均线：
#       短期均线在长期均线上方时满仓持有，否则空仓。
#       适合配合「自选股票」池（1 只标的）使用。
#
# 可调参数（在 initialize 中修改）：
#   fast : 短期均线天数（默认 5 个交易日）
#   slow : 长期均线天数（默认 20 个交易日）
#
# 常用 API：
#   data.universe()             当日可交易的池内成员列表
#   data.sma(symbol, n)         某标的过去 n 日收盘价均值
#   context.target_percent(s, w) 目标仓位：用 w（0~1）的资金持
#                                有标的 s，1.0 表示满仓
#   context.target_cash()        目标仓位：全部空仓
# ============================================================

def initialize(context):
    context.fast = 5
    context.slow = 20


def on_bar(context, data):
    # 1. 取当日池内第一只标的（配合自选股票池使用）
    universe = data.universe()
    if not universe:
        context.target_cash()
        return
    symbol = universe[0]

    # 2. 计算短期与长期均线（数据不足时为 None）
    fast = data.sma(symbol, context.fast)
    slow = data.sma(symbol, context.slow)

    # 3. 短期均线在长期均线上方 → 满仓持有；否则空仓
    if fast is not None and slow is not None and fast > slow:
        context.target_percent(symbol, 1.0)
    else:
        context.target_cash()
`;

const EQUAL_WEIGHT_BUY_HOLD = `# ============================================================
# 等权买入持有策略（预置脚本）
# ------------------------------------------------------------
# 思路：在回测区间第一个交易日开盘，将资金等权买入
#       股票池内全部可交易标的，此后一直持有不动。
#       通常用作其他策略的对照基准。
#
# 常用 API：
#   data.universe()          当日可交易的池内成员列表
#   context.target_equal([]) 目标仓位：等权持有列表内标的
#
# 注意：context.bought 是脚本自定义的状态变量，在
#       initialize 中初始化一次，之后跨交易日保持。
# ============================================================

def initialize(context):
    context.bought = False


def on_bar(context, data):
    # 只买一次：买入后不再改变目标仓位
    if context.bought:
        return
    universe = data.universe()
    if universe:
        context.target_equal(universe)
        context.bought = True
`;

const MA_GOLDEN_CROSS = `# ============================================================
# 均线金叉择时（预置脚本）
# ------------------------------------------------------------
# 思路：对股票池中第一只标的：
#   - MA5 由下向上穿越 MA10（金叉）→ 满仓买入；
#   - MA5 由上向下穿越 MA10（死叉）→ 清仓。
#   适合配合「自选股票」池（1 只标的）使用。
#
# 可调参数：
#   fast : 短期均线（默认 5 日）
#   slow : 长期均线（默认 10 日）
#
# 实现说明：用 data.history(symbol, n) 取最近 n 个收盘价，
#   自行计算今日与昨日的均线值来判定是否发生交叉。
# ============================================================

def initialize(context):
    context.fast = 5
    context.slow = 10


def on_bar(context, data):
    universe = data.universe()
    if not universe:
        context.target_cash()
        return
    symbol = universe[0]

    # 需要 slow+1 个收盘价才能同时算出今日与昨日的两条均线
    history = data.history(symbol, context.slow + 1, 'close')
    if len(history) < context.slow + 1:
        return  # 历史数据不足，保持当前仓位不动

    def ma(values):
        return sum(values) / len(values)

    today_fast = ma(history[-context.fast:])
    yesterday_fast = ma(history[-context.fast - 1:-1])
    today_slow = ma(history[-context.slow:])
    yesterday_slow = ma(history[-context.slow - 1:-1])

    # 金叉：昨短期 <= 昨长期，且今短期 > 今长期
    if yesterday_fast <= yesterday_slow and today_fast > today_slow:
        context.target_percent(symbol, 1.0)
    # 死叉：昨短期 >= 昨长期，且今短期 < 今长期
    elif yesterday_fast >= yesterday_slow and today_fast < today_slow:
        context.target_cash()
    # 其他情况保持当前仓位不动（不调用 target 函数即可）
`;

const RSI_REVERSAL = `# ============================================================
# RSI 超卖反转（预置脚本）
# ------------------------------------------------------------
# 思路：对股票池中第一只标的：
#   - RSI(14) 低于 30（超卖）→ 满仓买入；
#   - RSI(14) 高于 70（超买）→ 清仓。
#   属于均值回归类策略：在极端情绪时反向操作。
#
# 可调参数：
#   rsi_days : RSI 周期（默认 14 日）
#   oversold : 买入阈值（默认 30）
#   overbought : 卖出阈值（默认 70）
#
# 实现说明：RSI 用最近 rsi_days+1 个收盘价自行计算：
#   涨幅均值 / 跌幅均值 → 相对强弱指标。
# ============================================================

def initialize(context):
    context.rsi_days = 14
    context.oversold = 30
    context.overbought = 70


def on_bar(context, data):
    universe = data.universe()
    if not universe:
        context.target_cash()
        return
    symbol = universe[0]

    history = data.history(symbol, context.rsi_days + 1, 'close')
    if len(history) < context.rsi_days + 1:
        return

    gains = []
    losses = []
    for i in range(1, len(history)):
        change = history[i] - history[i - 1]
        if change >= 0:
            gains.append(change)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(-change)

    avg_gain = sum(gains) / len(gains)
    avg_loss = sum(losses) / len(losses)
    if avg_loss == 0:
        rsi = 100.0
    else:
        rsi = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    if rsi < context.oversold:
        context.target_percent(symbol, 1.0)
    elif rsi > context.overbought:
        context.target_cash()
    # 中间区间保持当前仓位不动
`;

const BULL_ALIGNMENT = `# ============================================================
# 多头排列趋势跟随（预置脚本）
# ------------------------------------------------------------
# 思路：对股票池中第一只标的：
#   - MA5 > MA10 > MA20 且收盘价在 MA20 之上（多头排列）
#     → 满仓持有；
#   - 收盘价跌破 MA20 → 清仓。
#   属于趋势跟随类策略：只在趋势明确向上时持仓。
#
# 可调参数：
#   short_days : 短期均线（默认 5 日）
#   mid_days   : 中期均线（默认 10 日）
#   long_days  : 长期均线（默认 20 日）
# ============================================================

def initialize(context):
    context.short_days = 5
    context.mid_days = 10
    context.long_days = 20


def on_bar(context, data):
    universe = data.universe()
    if not universe:
        context.target_cash()
        return
    symbol = universe[0]

    short_ma = data.sma(symbol, context.short_days)
    mid_ma = data.sma(symbol, context.mid_days)
    long_ma = data.sma(symbol, context.long_days)
    if short_ma is None or mid_ma is None or long_ma is None:
        return  # 数据不足，保持当前仓位

    close = data.close(symbol)
    # 多头排列且价格在长期均线上方 → 满仓
    if short_ma > mid_ma and mid_ma > long_ma and close > long_ma:
        context.target_percent(symbol, 1.0)
    # 价格跌破长期均线 → 清仓
    elif close < long_ma:
        context.target_cash()
    # 其他情况保持当前仓位不动
`;

const NEW_HIGH_BREAKOUT = `# ============================================================
# N 日新高突破（预置脚本）
# ------------------------------------------------------------
# 思路：对股票池中第一只标的：
#   - 收盘价创过去 N 个交易日新高 → 满仓买入；
#   - 持仓后收盘价跌破短期均线 → 清仓离场。
#   属于趋势突破类策略：强者恒强，破位离场。
#
# 可调参数：
#   breakout_days : 新高回看天数（默认 60 个交易日）
#   exit_ma_days  : 离场均线周期（默认 10 日）
#
# 实现说明：用 data.history 取最近 breakout_days 个收盘价，
#   与最高价比较判定是否突破；用 context.in_position
#   记录持仓状态（跨交易日保持）。
# ============================================================

def initialize(context):
    context.breakout_days = 60
    context.exit_ma_days = 10
    context.in_position = False


def on_bar(context, data):
    universe = data.universe()
    if not universe:
        context.target_cash()
        return
    symbol = universe[0]

    history = data.history(symbol, context.breakout_days, 'close')
    if len(history) < context.breakout_days:
        return

    close = history[-1]
    previous_high = max(history[:-1])  # 不含当日的前 N-1 日最高价

    if not context.in_position:
        # 空仓：收盘价突破前高 → 买入
        if close > previous_high:
            context.target_percent(symbol, 1.0)
            context.in_position = True
    else:
        # 持仓：跌破短期均线 → 卖出
        exit_ma = data.sma(symbol, context.exit_ma_days)
        if exit_ma is not None and close < exit_ma:
            context.target_cash()
            context.in_position = False
        # 否则继续持有（不调用 target 函数）
`;

const BOLLINGER_REVERSION = `# ============================================================
# 布林带回归（预置脚本）
# ------------------------------------------------------------
# 思路：对股票池中第一只标的：
#   - 收盘价跌破布林带下轨（超跌）→ 满仓买入；
#   - 收盘价回到中轨上方（回归完成）→ 清仓。
#   属于均值回归类策略：价格极端偏离后向均值回归。
#
# 可调参数：
#   window : 布林带窗口（默认 20 日）
#   mult   : 标准差倍数（默认 2.0）
#
# 实现说明：布林带 = 窗口均线 ± mult × 窗口标准差。
#   用 data.history 取收盘序列自行计算。
# ============================================================

def initialize(context):
    context.window = 20
    context.mult = 2.0
    context.in_position = False


def on_bar(context, data):
    universe = data.universe()
    if not universe:
        context.target_cash()
        return
    symbol = universe[0]

    history = data.history(symbol, context.window, 'close')
    if len(history) < context.window:
        return

    mid = sum(history) / len(history)
    variance = sum((value - mid) ** 2 for value in history) / len(history)
    std = variance ** 0.5
    lower = mid - context.mult * std

    close = history[-1]

    if not context.in_position:
        # 空仓：跌破下轨 → 买入
        if close < lower:
            context.target_percent(symbol, 1.0)
            context.in_position = True
    else:
        # 持仓：回到中轨上方 → 卖出
        if close > mid:
            context.target_cash()
            context.in_position = False
        # 否则继续持有
`;

const MOMENTUM_TOP_N = `# ============================================================
# 动量 Top-N 排名（预置脚本）
# ------------------------------------------------------------
# 思路：每天收盘后按过去 N 日涨幅对池内全部可交易成员排名，
#       等权持有排名前 Top-N 的标的；不附加任何过滤条件。
#       可与「动量轮动」对比：去掉均线过滤后的纯排名效果。
#
# 可调参数：
#   lookback : 动量回看天数（默认 20 个交易日）
#   top_n    : 持有只数（默认 5）
# ============================================================

def initialize(context):
    context.lookback = 20
    context.top_n = 5


def on_bar(context, data):
    ranked = data.rank_by_return(context.lookback)
    if not ranked:
        context.target_cash()
        return
    context.target_equal(ranked[:context.top_n])
`;

const MULTI_MOMENTUM_ROTATION = `# ============================================================
# 多股动量轮动（预置脚本）
# ------------------------------------------------------------
# 思路：每个交易日收盘后，按过去 N 日涨幅对池内成员排名，
#       在排名靠前的候选中，筛选收盘价站上自身均线的标的，
#       等权持有最多 Top-N 只；全部不满足时空仓。
#       相比单股轮动，分散到多只以降低单押风险。
#
# 可调参数（在 initialize 中修改）：
#   lookback  : 动量回看天数（默认 20 个交易日）
#   ma_days   : 均线过滤周期（默认 20 个交易日）
#   top_n     : 最多持有只数（默认 5）
#   candidate : 候选范围，在排名前 candidate 名内筛选
#               （默认 10，建议 ≥ top_n）
#
# 常用 API：
#   data.rank_by_return(n)  池内按过去 n 日涨幅排名（降序）
#   data.sma(symbol, n)     某标的过去 n 日收盘价均值
#   data.close(symbol)      某标的当日收盘价
#   context.target_equal([]) 目标仓位：等权持有列表内标的
#   context.target_cash()    目标仓位：全部空仓
# ============================================================

def initialize(context):
    context.lookback = 20
    context.ma_days = 20
    context.top_n = 5
    context.candidate = 10


def on_bar(context, data):
    # 1. 按动量对当日池内成员排名
    ranked = data.rank_by_return(context.lookback)
    if not ranked:
        context.target_cash()
        return

    # 2. 在排名靠前的候选中筛选：收盘价必须站上自身均线
    kept = []
    for symbol in ranked[:context.candidate]:
        average = data.sma(symbol, context.ma_days)
        if average is not None and data.close(symbol) > average:
            kept.append(symbol)
        if len(kept) >= context.top_n:
            break

    # 3. 有满足条件的就等权持有，否则空仓
    if kept:
        context.target_equal(kept)
    else:
        context.target_cash()
`;

export const PRESET_SCRIPTS: PresetScript[] = [
  {
    id: 'momentum_rotation',
    nameZh: '动量轮动',
    nameEn: 'Momentum rotation',
    script: SAMPLE_SCRIPT,
  },
  {
    id: 'multi_momentum_rotation',
    nameZh: '多股动量轮动',
    nameEn: 'Multi-asset momentum rotation',
    script: MULTI_MOMENTUM_ROTATION,
  },
  {
    id: 'momentum_top_n',
    nameZh: '动量 Top-N 排名',
    nameEn: 'Momentum Top-N',
    script: MOMENTUM_TOP_N,
  },
  {
    id: 'dual_ma_timing',
    nameZh: '双均线择时',
    nameEn: 'Dual moving-average timing',
    script: DUAL_MA_TIMING,
  },
  {
    id: 'ma_golden_cross',
    nameZh: '均线金叉',
    nameEn: 'MA golden cross',
    script: MA_GOLDEN_CROSS,
  },
  {
    id: 'bull_alignment',
    nameZh: '多头排列',
    nameEn: 'Bullish alignment',
    script: BULL_ALIGNMENT,
  },
  {
    id: 'new_high_breakout',
    nameZh: '60 日新高突破',
    nameEn: '60-day high breakout',
    script: NEW_HIGH_BREAKOUT,
  },
  {
    id: 'rsi_reversal',
    nameZh: 'RSI 超卖反转',
    nameEn: 'RSI oversold reversal',
    script: RSI_REVERSAL,
  },
  {
    id: 'bollinger_reversion',
    nameZh: '布林带回归',
    nameEn: 'Bollinger reversion',
    script: BOLLINGER_REVERSION,
  },
  {
    id: 'equal_weight_buy_hold',
    nameZh: '等权买入持有',
    nameEn: 'Equal-weight buy and hold',
    script: EQUAL_WEIGHT_BUY_HOLD,
  },
];
