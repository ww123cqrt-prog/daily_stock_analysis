"""BVR-20 short-suspension proposal; saved assets require separate approval."""

def initialize(context):
    context.stock_pool = ("SH601208", "SZ300054", "SZ002709", "SH603986", "SH688008", "SH600584", "SZ301566", "SH603019", "SZ002409", "SH688313", "SH688017", "SZ300408", "SZ002138", "SH600183", "SZ002463", "SH600378", "SZ002886", "SH688183", "SH688498", "SH601869", "SH600487", "SZ002484", "SZ002859", "SH688019", "SZ300666", "SZ300346", "SH688268", "SH688146", "SH688072", "SH688012", "SH688120", "SZ300604", "SZ300373", "SH603290", "SH603938", "SH603688", "SZ002371", "SH688082", "SH688652")
    context.states = {}
    context.pending_entries = {}
    context.pending_exits = set()
    context.pending_adds = {}
    context.observations = {}
    context.market_bar = 0

def _mean(values):
    return sum(values) / len(values) if values else None

def _std(values):
    if len(values) < 2:
        return None
    mean = _mean(values)
    return (sum((value - mean) * (value - mean) for value in values) / (len(values) - 1)) ** 0.5

def _b_values(data, symbol, field="close", ago=0):
    """Twenty valid observations, bounded to 25 market dates at that date."""
    newer = len(data.history(symbol, ago, field)) if ago else 0
    if len(data.history(symbol, ago + 1, field)) == newer:
        return ()
    for span in range(25 + ago, 19 + ago, -1):
        values = data.history(symbol, span, field)
        if values:
            values = values[:len(values) - newer]
            return values[-20:] if len(values) >= 20 else ()
    return ()

def _bands(data, symbol, b_mode=False):
    closes = _b_values(data, symbol) if b_mode else data.history(symbol, 20, "close")
    volumes = _b_values(data, symbol, "volume") if b_mode else data.history(symbol, 20, "volume")
    if len(closes) < 20 or len(volumes) < 20:
        return None
    middle = _mean(closes)
    deviation = _std(closes)
    volume_mean = _mean(volumes)
    if middle is None or deviation is None or volume_mean is None:
        return None
    return {"close": closes[-1], "middle": middle, "upper": middle + 2.0 * deviation, "lower": middle - 2.0 * deviation, "volume": volumes[-1], "volume_mean": volume_mean}

def _strong_regime(data):
    values = data.index_history(20)
    if len(values) < 20:
        return None
    return values[-1] >= _mean(values)

def _strong_streak(data):
    values = data.index_history(40)
    if len(values) < 20:
        return 0
    streak = 0
    for index in range(len(values) - 1, 18, -1):
        window = values[index - 19:index + 1]
        if values[index] < _mean(window):
            break
        streak += 1
        if streak >= 3:
            break
    return streak

def _mode_b_pullback(data, symbol, bands):
    if _strong_streak(data) < 3 or bands["close"] < bands["middle"] or bands["close"] >= bands["upper"]:
        return None
    closes = data.history(symbol, 25, "close")
    if len(closes) < 21:
        return None
    for ago in range(1, 6):
        window = _b_values(data, symbol, ago=ago)
        if len(window) < 20:
            continue
        deviation = _std(window)
        if deviation is not None and window[-1] >= _mean(window) + 2.0 * deviation:
            return (window[-1], ago)
    return None

def _apply_targets(context, targets):
    context.target_cash()
    for symbol in sorted(targets):
        if targets[symbol] > 0:
            context.target_percent(symbol, targets[symbol])

def on_bar(context, data):
    context.market_bar += 1
    if set(context.symbols) != set(context.stock_pool):
        context.target_cash()
        return
    strong = _strong_regime(data)
    if strong is None:
        context.target_cash()
        return
    current = dict(context.current_weights)
    targets = dict(current)
    # Entry/exit state is maintained by actual fill callbacks, never requested weights.
    for symbol in list(context.states):
        if symbol not in current:
            continue
        state = context.states[symbol]
        bands = _bands(data, symbol, b_mode=state["mode"] == "B")
        if bands is None:
            continue
        close = bands["close"]
        if state["mode"] == "A":
            if close >= bands["middle"] or close <= state["first_entry_price"] * 0.70:
                targets.pop(symbol, None)
                context.pending_exits.add(symbol)
            elif state["add_count"] < 2 and close <= state["last_buy_price"] * 0.90:
                context.pending_adds[symbol] = str(context.current_date)
                targets[symbol] = min(0.75, targets.get(symbol, 0.50) + 0.10)
        else:
            if close >= bands["upper"] and bands["volume"] >= bands["volume_mean"] * 1.30:
                targets.pop(symbol, None)
                context.pending_exits.add(symbol)
            elif close < bands["middle"] * 0.95:
                targets.pop(symbol, None)
                context.pending_exits.add(symbol)

    for symbol in list(context.observations):
        observation = context.observations[symbol]
        if (symbol in context.states or not strong
                or context.market_bar - observation["touch_bar"] > 5
                or len(data.history(symbol, 25, "close")) < 21):
            context.observations.pop(symbol, None)
            continue
        bands = _bands(data, symbol, b_mode=True)
        if bands is None or bands["close"] < bands["middle"]:
            context.observations.pop(symbol, None)
            continue
        if bands["close"] >= observation["touch_close"] * 1.03:
            targets[symbol] = 0.50
            context.pending_entries[symbol] = "B"
            context.observations.pop(symbol, None)
        else:
            observation["days_left"] -= 1
            if observation["days_left"] <= 0:
                context.observations.pop(symbol, None)
    slots = max(0, 2 - len(context.states))
    candidates = []
    for symbol in context.stock_pool:
        if symbol in context.states:
            continue
        bands = _bands(data, symbol, b_mode=strong)
        if bands is None:
            continue
        if not strong and bands["close"] < bands["lower"] and bands["volume"] >= bands["volume_mean"] * 1.40:
            candidates.append((symbol, "A"))
        elif strong:
            touch = _mode_b_pullback(data, symbol, bands)
            if touch is not None:
                context.observations[symbol] = {"touch_close": touch[0], "days_left": 3,
                                                "touch_bar": context.market_bar - touch[1]}
    selected = candidates[:slots]
    if selected:
        remaining = max(0.0, 0.75 - sum(targets.values()))
        weight = min(0.50, remaining / len(selected))
        for symbol, mode in selected:
            if weight > 0:
                targets[symbol] = weight
                context.pending_entries[symbol] = mode
    if targets != current:
        if sum(targets.values()) > 0.75:
            scale = 0.75 / sum(targets.values())
            targets = {symbol: value * scale for symbol, value in targets.items()}
        _apply_targets(context, targets)


def on_fill(context, trade):
    """Only actual buys update entry/add anchors; partial sales retain them."""
    symbol = trade["symbol"]
    if trade["action"] == "buy":
        if symbol not in context.states:
            mode = context.pending_entries.pop(symbol)
            context.states[symbol] = {"mode": mode, "first_entry_price": trade["price"],
                "last_buy_price": trade["price"], "add_count": 0}
        elif context.pending_adds.get(symbol) == trade["signal_date"]:
            state = context.states[symbol]
            state["add_count"] += 1
            state["last_buy_price"] = trade["price"]
            context.pending_adds.pop(symbol, None)
    elif context.positions.get(symbol, 0) == 0:
        context.states.pop(symbol, None)
        context.pending_exits.discard(symbol)
        context.pending_adds.pop(symbol, None)
        context.pending_entries.pop(symbol, None)
    context.log((symbol, trade["action"], trade["shares"], trade["price"], dict(context.states.get(symbol, {}))))
