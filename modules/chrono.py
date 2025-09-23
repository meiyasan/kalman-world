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

def summary():
    """
    Print a detailed summary of all recorded laps and total elapsed time.
    """
    assert 'chrono_start_time' in globals(), "restart() must be called before summary()"
    now   = time.time()
    total = now - chrono_start_time

    print("Chrono summary:")
    prev = 0.0
    for i, entry in enumerate(records, start=1):
        ts    = entry["timestamp"]
        delta = ts - prev
        print(f"{i:2d}. {entry['title']}: {ts:.6f}s (ΔT={delta:.6f}s)")
        prev = ts

    print(f"\n==> Total elapsed: {total:.6f}s\n")

# initialize on import
restart()