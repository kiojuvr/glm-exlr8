"""Model-free M1 experiment, not a runtime/cache API. See docs/m1-storage-decision.md.
Run with PYTHONPATH pointing at a version-specific build of m1_probe.
"""
import argparse
import gc
import importlib.metadata
import json
import os
import platform
import subprocess
import time

import mlx.core as mx
import numpy as np
from m1_probe import backing

P, WIDTHS = 256, (512, 64, 128)
PAGE_BYTES = P * sum(WIDTHS) * 2
LIMIT = 128 * 2**20
STREAM = mx.default_stream(mx.gpu)
START = time.perf_counter()
RSS_START = 0


def rss():
    # Current RSS, not ru_maxrss (which is a lifetime high-water mark).
    return int(subprocess.check_output(['ps', '-o', 'rss=', '-p', str(os.getpid())])) * 1024


def sample(label, live=0, **extra):
    row = dict(label=label, seconds=time.perf_counter() - START,
               logical_page_bytes=live, active=mx.get_active_memory(),
               cached=mx.get_cache_memory(), peak=mx.get_peak_memory(), rss=rss(), **extra)
    if row['peak'] > LIMIT or row['rss'] - RSS_START > 256 * 2**20:
        raise RuntimeError(f'M1 safety budget exceeded: {row}')
    return row


def drain():
    mx.synchronize(STREAM)
    gc.collect()


def expected(positions, width, component):
    return (np.asarray(positions)[:, None] % 31 +
            np.arange(width)[None, :] % 7 + component * 40).astype(np.float32)


def incoming(start, count, width, component):
    # Independent device formula; the oracle above runs on the host.
    return ((mx.arange(start, start + count)[:, None] % 31 +
             mx.arange(width)[None, :] % 7 + component * 40)
            .astype(mx.bfloat16).reshape(1, 1, count, width))


def complete(mode, changed, output):
    begin = time.perf_counter()
    if mode == 'output_only':
        mx.eval(output)  # Deliberately NOT an all-state commit.
    elif mode in ('sync_all', 'bounded_cow'):
        mx.eval(*changed, output)
    else:
        mx.async_eval(*changed, output)
        if mode == 'ds41f_all':
            # Independent control evaluation cannot substitute for the fence.
            control = mx.array([1], dtype=mx.int32)
            mx.eval(control)
        mx.synchronize(STREAM)
    return time.perf_counter() - begin


def gather(pages, positions, component):
    # Only compact selected rows are assembled, never full-cache concatenation.
    out = mx.zeros((len(positions), WIDTHS[component]), dtype=mx.bfloat16)
    for page_id in sorted(set(p // P for p in positions)):
        slots = [i for i, p in enumerate(positions) if p // P == page_id]
        rows = [positions[i] % P for i in slots]
        out[mx.array(slots)] = pages[page_id][component][0, 0, mx.array(rows)]
    return out


def verify_gather(pages, frontier):
    positions = sorted(set([0, frontier - 1, frontier // 2] +
                           list(range(0, frontier, 193))))
    for component, width in enumerate(WIDTHS):
        compact = gather(pages, positions, component)
        mx.eval(compact)
        np.testing.assert_array_equal(np.asarray(compact.astype(mx.float32)),
                                      expected(positions, width, component))


def cycle(mode, cycle_id):
    pages, seals, seal_backings = [], [], []
    tail_replacements = [0, 0, None if mode == 'output_only' else 0]
    sealed_replacements = 0
    elapsed = 0.0
    snapshots = []
    frontier = 0
    target = 8 * P + 73
    while frontier < target:
        offset = frontier % P
        if not offset:
            pages.append([mx.zeros((1, 1, P, w), dtype=mx.bfloat16) for w in WIDTHS])
            mx.eval(*pages[-1])
        tail = pages[-1]
        # Probe only completed state. output_only intentionally leaves indexer lazy.
        components = range(2) if mode == 'output_only' else range(3)
        before = {c: backing(tail[c]) for c in components}
        if mode == 'bounded_cow':
            # Force one old active-tail backing per component to survive this
            # transaction only. This is a bounded COW control, not another API.
            aliases = [a[:, :, :1, :] for a in tail]
            mx.eval(*aliases)
        count = min(17, P - offset, target - frontier)
        for c, width in enumerate(WIDTHS):
            tail[c][:, :, offset:offset + count, :] = incoming(frontier, count, width, c)
        output = mx.sum(tail[0].astype(mx.float32)) + mx.sum(tail[1].astype(mx.float32))
        elapsed += complete(mode, tail, output)
        del output
        if mode == 'bounded_cow':
            mx.synchronize(STREAM)
            del aliases
        for c in components:
            tail_replacements[c] += backing(tail[c]) != before[c]
        for page_id, handles in zip(seals, seal_backings):
            for c in range(3):
                sealed_replacements += backing(pages[page_id][c]) != handles[c]
        assert sealed_replacements == 0
        frontier += count
        # No retained arrays/graphs in telemetry.
        if frontier % P == 0:
            mx.eval(*tail)
            mx.synchronize(STREAM)
            seals.append(len(pages) - 1)
            seal_backings.append([backing(a) for a in tail])
            snapshots.append(sample('seal', len(pages) * PAGE_BYTES,
                                    frontier=frontier, page_count=len(pages)))
            if mode != 'output_only':
                verify_gather(pages, frontier)
    snapshots.append(sample('partial_before_drain', len(pages) * PAGE_BYTES,
                            frontier=frontier, page_count=len(pages)))
    mx.eval(*tail)
    drain()
    snapshots.append(sample('partial_completed', len(pages) * PAGE_BYTES,
                            frontier=frontier, page_count=len(pages)))
    verify_gather(pages, frontier)
    # Verify every row, including that unwritten capacity is still zero.
    for page_id, page in enumerate(pages):
        valid = min(P, frontier - page_id * P)
        for c, a in enumerate(page):
            actual = np.asarray(a.astype(mx.float32))[0, 0]
            np.testing.assert_array_equal(actual[:valid],
                expected(range(page_id * P, page_id * P + valid), WIDTHS[c], c))
            assert np.all(actual[valid:] == 0)
    del actual, a, page, tail, pages
    drain()
    closed = sample('closed', mode=mode, cycle=cycle_id)
    assert closed['active'] == 0, closed
    assert closed['peak'] < 8 * 2**20, closed
    if mode in ('sync_all', 'async_all', 'ds41f_all'):
        assert tail_replacements == [0, 0, 0], tail_replacements
    elif mode == 'bounded_cow':
        assert tail_replacements == [133, 133, 133], tail_replacements
    assert snapshots[-1]['active'] == (len(seals) + 1) * PAGE_BYTES
    return dict(mode=mode, cycle=cycle_id, sealed_pages=len(seals),
                tail_component_backing_replacements=tail_replacements,
                sealed_component_backing_replacements=sealed_replacements,
                completion_seconds=elapsed, samples=snapshots, closed=closed)


def retention(kind):
    """Bounded adverse case: hold 32 old tails and/or lazy 1 MiB scratch nodes."""
    mx.clear_cache()
    mx.reset_peak_memory()
    baseline = sample('retention_baseline')
    held = []
    tail = mx.zeros((1, 1, P, 512), dtype=mx.bfloat16)
    mx.eval(tail)
    handles, replacements = set(), 0
    for i in range(32):
        previous = backing(tail)
        if kind == 'view':
            view = tail[:, :, :1, :]
            mx.eval(view)
            assert backing(view)[0] == previous[0], 'view must really alias'
            held.append(view)
            del view
        else:
            scratch = mx.full((512, 512), i, dtype=mx.float32)
            mx.eval(scratch)
            # Pending output keeps BOTH old tail and evaluated scratch reachable.
            held.append(mx.sum(tail.astype(mx.float32)) + mx.sum(scratch))
            del scratch
        handles.add(previous[0])
        tail[:, :, i:i + 1, :] = mx.full((1, 1, 1, 512), i + 1, dtype=mx.bfloat16)
        mx.eval(tail)
        mx.synchronize(STREAM)
        replacements += backing(tail) != previous
        sample('retention_step')
    with_owner = sample('held_with_owner', tail.nbytes, count=len(held))
    actual = np.asarray(tail.astype(mx.float32))[0, 0]
    np.testing.assert_array_equal(actual[:32], np.broadcast_to(np.arange(1, 33)[:, None], (32, 512)))
    assert np.all(actual[32:] == 0)
    del actual, tail
    drain()
    mx.clear_cache()
    retained = sample('owner_dropped_sync_cache_cleared', count=len(held))
    # Confirm snapshots refer to pre-update values, not a mutated shared buffer.
    if kind != 'lazy_drop':
        for i, item in enumerate(held):
            if kind == 'view':
                np.testing.assert_array_equal(np.asarray(item.astype(mx.float32)), 0 if i == 0 else 1)
            else:
                want = 512 * sum(range(1, i + 1)) + 512 * 512 * i
                assert item.item() == want, (i, item.item(), want)
        del item
    drain()
    evaluated = sample('pending_unevaluated' if kind == 'lazy_drop' else 'retained_outputs_evaluated')
    held.clear()
    drain()
    released = sample('references_removed_and_synchronized')
    mx.clear_cache()
    cleared = sample('released_cache_cleared')
    assert cleared['active'] == 0
    assert cleared['peak'] < 64 * 2**20
    assert retained['active'] > baseline['active'] + 4 * 2**20
    return dict(kind=kind, backing_replacements=replacements,
                distinct_old_backings=len(handles), samples=[baseline, with_owner, retained,
                                                            evaluated, released, cleared])


def main():
    global RSS_START
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    assert mx.metal.is_available()
    mx.set_memory_limit(LIMIT)
    mx.set_cache_limit(16 * 2**20)
    RSS_START = rss()
    mx.eval(mx.zeros((1,)))
    drain()
    mx.clear_cache()
    mx.reset_peak_memory()
    records = []
    for mode in ('sync_all', 'async_all', 'ds41f_all', 'output_only', 'bounded_cow'):
        for cycle_id in range(4):
            records.append(cycle(mode, cycle_id))
            print(f'{mode} cycle {cycle_id}: closed active={records[-1]["closed"]["active"]}', flush=True)
    results = dict(mlx=importlib.metadata.version('mlx'),
                   mlx_metal=importlib.metadata.version('mlx-metal'),
                   nanobind=importlib.metadata.version('nanobind'),
                   python=platform.python_version(), platform=platform.platform(),
                   device=mx.device_info(), page_bytes=PAGE_BYTES,
                   memory_limit=LIMIT, cache_limit=16 * 2**20,
                   cycles=records, retention=[retention('view'), retention('lazy'), retention('lazy_drop')])
    with open(args.output, 'w') as f:
        json.dump(results, f, separators=(',', ':'))
        f.write('\n')
    print(f'PASS: {args.output}', flush=True)


if __name__ == '__main__':
    main()
