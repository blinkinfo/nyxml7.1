"""Message formatters — every output shows UTC timeslots with emojis."""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Live notifications (sent by scheduler)
# ---------------------------------------------------------------------------

def format_signal(
    side: str,
    entry_price: float,
    slot_start_str: str,
    slot_end_str: str,
    autotrade: bool,
    adx_direction: str | None = None,
    adx_flipped: bool = False,
    adx_value: float | None = None,
) -> str:
    side_emoji = "\U0001f4c8" if side == "Up" else "\U0001f4c9"
    at_line = "\U0001f916 AutoTrade: ON \u2192 Order Placed" if autotrade else "\U0001f916 AutoTrade: OFF"

    # ADX info line
    adx_line = ""
    if adx_direction is not None and adx_value is not None:
        adx_emoji = "\U0001f4c8" if adx_direction == "rising" else "\U0001f4c9"
        flip_note = " (FLIPPED)" if adx_flipped else ""
        adx_line = f"\u2502 {adx_emoji} ADX(14): {adx_value:.2f} {adx_direction}{flip_note}\n"

    return (
        "\U0001f4e1 <b>Signal Fired!</b>\n"
        "\u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"\u2502 \u23f0 Slot: {slot_start_str}-{slot_end_str} UTC\n"
        f"\u2502 {side_emoji} Side: {side}\n"
        f"\u2502 \U0001f4b2 Ask Price: ${entry_price:.2f}\n"
        f"{adx_line}"
        f"\u2502 {at_line}\n"
        "\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
    )


def format_skip(
    slot_start_str: str,
    slot_end_str: str,
    up_price: float,
    down_price: float,
    adx_direction: str | None = None,
    adx_value: float | None = None,
) -> str:
    adx_line = ""
    if adx_direction is not None and adx_value is not None:
        adx_emoji = "\U0001f4c8" if adx_direction == "rising" else "\U0001f4c9"
        adx_line = f"\u2502 {adx_emoji} ADX(14): {adx_value:.2f} {adx_direction}\n"

    return (
        "\u23ed\ufe0f <b>No Signal</b>\n"
        "\u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"\u2502 \u23f0 Slot: {slot_start_str}-{slot_end_str} UTC\n"
        f"\u2502 \U0001f4c8 Up Ask: ${up_price:.2f}  |  \U0001f4c9 Down Ask: ${down_price:.2f}\n"
        f"{adx_line}"
        "\u2502 Neither side \u2265 $0.51 \u2014 skipping\n"
        "\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
    )


def format_resolution(
    is_win: bool,
    side: str,
    entry_price: float,
    slot_start_str: str,
    slot_end_str: str,
    pnl: float | None = None,
) -> str:
    result_price = 1.00 if is_win else 0.00
    icon = "\u2705" if is_win else "\u274c"
    label = "WIN" if is_win else "LOSS"
    side_emoji = "\U0001f4c8" if side == "Up" else "\U0001f4c9"

    lines = [
        f"{icon} <b>Signal Result \u2014 {label}</b>",
        "\u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
        f"\u2502 \u23f0 Slot: {slot_start_str}-{slot_end_str} UTC",
        f"\u2502 {side_emoji} Side: {side}",
        f"\u2502 \U0001f4b2 Entry: ${entry_price:.2f} \u2192 Result: ${result_price:.2f}",
    ]
    if pnl is not None:
        sign = "+" if pnl >= 0 else ""
        lines.append(f"\u2502 \U0001f4b0 P&L: {sign}${pnl:.2f}")
    lines.append("\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dashboards (requested via bot commands)
# ---------------------------------------------------------------------------

def format_signal_stats(stats: dict[str, Any], label: str = "All Time") -> str:
    streak_str = "0"
    if stats.get("current_streak") and stats.get("current_streak_type"):
        streak_str = f"{stats['current_streak']}{stats['current_streak_type']}"

    SEP = "\u2501" * 20
    lines = [
        f"\U0001f4ca <b>Signal Performance ({label})</b>",
        SEP,
        f"\U0001f4e1 Total Signals: {stats['total_signals']}",
        f"\u2705 Wins: {stats['wins']}  |  \u274c Losses: {stats['losses']}",
        f"\U0001f4c8 Win Rate: {stats['win_pct']}%",
        SEP,
        f"\U0001f525 Current Streak: {streak_str}",
        f"\U0001f3c6 Best Win Streak: {stats['best_win_streak']}",
        f"\U0001f480 Worst Loss Streak: {stats['worst_loss_streak']}",
        SEP,
        f"\u23ed\ufe0f Skipped (No Signal): {stats['skip_count']}",
    ]
    return "\n".join(lines)


def format_trade_stats(stats: dict[str, Any], label: str = "All Time") -> str:
    streak_str = "0"
    if stats.get("current_streak") and stats.get("current_streak_type"):
        streak_str = f"{stats['current_streak']}{stats['current_streak_type']}"

    sign = "+" if stats["net_pnl"] >= 0 else ""
    roi_sign = "+" if stats["roi_pct"] >= 0 else ""

    SEP = "\u2501" * 20
    lines = [
        f"\U0001f4b0 <b>Trade Performance ({label})</b>",
        SEP,
        f"\U0001f4ca Total Trades: {stats['total_trades']}",
        f"\u2705 Wins: {stats['wins']}  |  \u274c Losses: {stats['losses']}",
        f"\U0001f4c8 Win Rate: {stats['win_pct']}%",
        SEP,
        f"\U0001f4b5 Total Deployed: ${stats['total_deployed']:.2f}",
        f"\U0001f4b0 Total Returned: ${stats['total_returned']:.2f}",
        f"\U0001f4c8 Net P&L: {sign}${stats['net_pnl']:.2f}",
        f"\U0001f4ca ROI: {roi_sign}{stats['roi_pct']}%",
        SEP,
        f"\U0001f525 Current Streak: {streak_str}",
        f"\U0001f3c6 Best Win Streak: {stats['best_win_streak']}",
    ]
    return "\n".join(lines)


def format_status(
    connected: bool,
    balance: float | None,
    autotrade: bool,
    trade_amount: float,
    open_positions: int,
    uptime_str: str,
    last_signal: str | None,
) -> str:
    conn_icon = "\U0001f7e2" if connected else "\U0001f534"
    conn_text = "Connected" if connected else "Disconnected"
    at_text = "ON" if autotrade else "OFF"
    bal_text = f"{balance:.2f} USDC" if balance is not None else "N/A"
    sig_text = last_signal or "None"

    SEP = "\u2501" * 20
    lines = [
        "\U0001f916 <b>AutoPoly Status</b>",
        SEP,
        f"{conn_icon} Bot: Running",
        f"\U0001f517 Polymarket: {conn_text}",
        f"\U0001f4b0 Balance: {bal_text}",
        SEP,
        f"\U0001f916 AutoTrade: {at_text}",
        f"\U0001f4b5 Trade Amount: ${trade_amount:.2f}",
        f"\U0001f4ca Open Positions: {open_positions}",
        SEP,
        f"\u23f0 Uptime: {uptime_str}",
        f"\U0001f4e1 Last Signal: {sig_text}",
    ]
    return "\n".join(lines)


def format_recent_signals(signals: list[dict[str, Any]]) -> str:
    if not signals:
        return "\nNo signals recorded yet."
    lines = ["\n\U0001f4cb <b>Recent Signals:</b>"]
    for s in signals:
        ss = s["slot_start"].split(" ")[-1] if " " in s["slot_start"] else s["slot_start"]
        se = s["slot_end"].split(" ")[-1] if " " in s["slot_end"] else s["slot_end"]
        if s["skipped"]:
            lines.append(f"\u23ed\ufe0f {ss}-{se} UTC \u2014 skipped")
        else:
            icon = "\u2705" if s.get("is_win") == 1 else ("\u274c" if s.get("is_win") == 0 else "\u23f3")
            lines.append(f"{icon} {ss}-{se} UTC  {s['side']}  ${s.get('entry_price', 0):.2f}")
    return "\n".join(lines)


def format_recent_trades(trades: list[dict[str, Any]]) -> str:
    if not trades:
        return "\nNo trades recorded yet."
    lines = ["\n\U0001f4cb <b>Recent Trades:</b>"]
    for t in trades:
        ss = t["slot_start"].split(" ")[-1] if " " in t["slot_start"] else t["slot_start"]
        se = t["slot_end"].split(" ")[-1] if " " in t["slot_end"] else t["slot_end"]
        icon = "\u2705" if t.get("is_win") == 1 else ("\u274c" if t.get("is_win") == 0 else "\u23f3")
        pnl_str = ""
        if t.get("pnl") is not None:
            sign = "+" if t["pnl"] >= 0 else ""
            pnl_str = f"  {sign}${t['pnl']:.2f}"
        lines.append(f"{icon} {ss}-{se} UTC  {t['side']}  ${t['amount_usdc']:.2f}{pnl_str}")
    return "\n".join(lines)


def format_help() -> str:
    return (
        "\u2753 <b>AutoPoly Commands</b>\n\n"
        "/start \u2014 Main menu\n"
        "/status \u2014 Bot status & balance\n"
        "/signals \u2014 Signal performance stats\n"
        "/trades \u2014 Trade P&L dashboard\n"
        "/settings \u2014 Toggle autotrade, set amount\n"
        "/help \u2014 This help message\n\n"
        "<b>How it works:</b>\n"
        "Every 5 minutes the bot checks the NEXT slot's BTC up/down "
        "prices 85 seconds before the current slot ends. If either "
        "side \u2265 $0.51, a signal fires and trades that slot.\n\n"
        "<b>ADX Filter:</b>\n"
        "The bot computes ADX(14) from Coinbase BTC-USD 5-minute candles. "
        "If ADX is <b>rising</b> (strengthening trend), the signal is "
        "<b>flipped</b> (Up\u2194Down) to fade the consensus. If ADX is "
        "falling or flat, the original signal is kept. "
        "With AutoTrade ON, a FOK market order is placed automatically."
    )
