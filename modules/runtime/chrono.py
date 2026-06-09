import sys
import time

# now a list of {"title": str, "timestamp": float} entries
records = []

def restart():
    """
    Start (or restart) the timer and reset all lap records.
    """
    global chrono_start_time, records
    records = []                    # clear out old entries
    chrono_start_time = time.time()
    return chrono_start_time

def lap(title=None):
    """
    Record a lap: append {"title": title, "timestamp": elapsed_since_start}.
    """
    assert 'chrono_start_time' in globals(), "restart() must be called before lap()"
    now     = time.time()
    elapsed = now - chrono_start_time

    entry = {
        "title": title or None,
        "timestamp": elapsed
    }
    records.append(entry)
    return f"{entry['title']}: {elapsed:.6f}s" if entry['title'] else f"{elapsed:.6f}s"

def elapsed_time() -> float:
    """
    Return the total time elapsed since the last call to restart(),
    without printing anything.
    """
    assert 'chrono_start_time' in globals(), "restart() must be called before elapsed_time()"
    return time.time() - chrono_start_time

def _fmt(t: float) -> str:
    """Format a duration as ms or s depending on magnitude."""
    if t < 1.0:
        return f"{t * 1000:.1f} ms"
    return f"{t:.3f} s"

def summary(steps=None):
    """
    Print a lap summary, flushing immediately.

    Parameters
    ----------
    steps : None | int | iterable of int
        Which 1-based lap indices to display.
        - None or omitted  → all laps (default)
        - int N            → laps 1 .. N
        - list / range     → the specified indices, e.g. [1, 3, 5] or range(2, 6)

    Examples
    --------
    chrono.summary()           # all laps
    chrono.summary(4)          # first 4 laps
    chrono.summary(range(2,6)) # laps 2, 3, 4, 5
    chrono.summary([1, 3, 5])  # laps 1, 3, 5
    """
    assert 'chrono_start_time' in globals(), "restart() must be called before summary()"
    now   = time.time()
    total = now - chrono_start_time

    if not records:
        print("Chrono summary: (no laps recorded)")
        return

    # Resolve which 1-based indices to show
    if steps is None:
        visible = set(range(1, len(records) + 1))
    elif isinstance(steps, int):
        visible = set(range(1, steps + 1))
    else:
        visible = set(steps)

    shown = [(i, records[i - 1]) for i in range(1, len(records) + 1) if i in visible]

    w = max(len(e["title"] or "") for e in records)
    header = f"  {'#':>3}   {'label':<{w}}   {'t':>9}   {'ΔT':>9}"
    rule   = f"  {'─'*3}   {'─'*w}   {'─'*9}   {'─'*9}"

    lines = ["Chrono summary:", header, rule]
    for i, entry in shown:
        ts    = entry["timestamp"]
        prev  = records[i - 2]["timestamp"] if i > 1 else 0.0
        delta = ts - prev
        title = entry["title"] or f"lap {i}"
        lines.append(f"  {i:>3}.  {title:<{w}}   {_fmt(ts):>9}   {_fmt(delta):>9}")

    lines.append(rule)
    lines.append(f"  {'':>3}   {'total':<{w}}   {_fmt(total):>9}")

    print("\n".join(lines), flush=True)
    sys.stdout.flush()

# initialize on import
restart()