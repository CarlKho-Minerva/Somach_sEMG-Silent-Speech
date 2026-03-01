#!/usr/bin/env python3
"""
MASTER RUNNER — Runs all 5 experiments sequentially.
Each experiment outputs its own log file.
Reduced to EPOCHS=50 and uses known baseline (62.0%) to avoid double training.
"""
import subprocess, os, time, sys

BASE = os.path.dirname(os.path.abspath(__file__))

experiments = [
    ('exp1_specaugment', 'SpecAugment'),
    ('exp2_feature_importance', 'Feature Importance'),
    ('exp3_permutation_integrity', 'Permutation Integrity'),
    ('exp4_raw_signal_cnn', 'Raw Signal CNN'),
    ('exp5_phase_weighted', 'Phase-Weighted Curriculum'),
]

results = {}

for folder, name in experiments:
    print(f'\n{"="*60}', flush=True)
    print(f'  STARTING: {name}', flush=True)
    print(f'{"="*60}', flush=True)
    t0 = time.time()

    log_path = os.path.join(BASE, folder, 'log.txt')
    run_path = os.path.join(BASE, folder, 'run.py')

    with open(log_path, 'w') as log_f:
        result = subprocess.run(
            [sys.executable, '-u', run_path],
            cwd=os.path.join(BASE, folder),
            stdout=log_f, stderr=subprocess.STDOUT,
            timeout=600  # 10 min max per experiment
        )

    elapsed = time.time() - t0
    status = '✅' if result.returncode == 0 else '❌'
    results[name] = {'code': result.returncode, 'time': elapsed}
    print(f'  {status} {name}: exit={result.returncode}, time={elapsed:.0f}s', flush=True)

print(f'\n{"="*60}', flush=True)
print(f'  ALL EXPERIMENTS COMPLETE', flush=True)
for name, r in results.items():
    s = '✅' if r['code'] == 0 else '❌'
    print(f'  {s} {name}: {r["time"]:.0f}s', flush=True)
print(f'{"="*60}', flush=True)
