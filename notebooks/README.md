# notebooks/ — legacy notebooks

These notebooks are the earlier thin wrappers around the 1,000-node,
same-direction experiment. They default to `RUN_MODE='none'` and are retained
for reference.

The current authoritative 5,000-node reverse-influence experiment is run via:

```bash
python experiments/exp_20261124_5000_reversed/run_experiment_5000.py --mode plan
python experiments/exp_20261124_5000_reversed/run_experiment_5000.py --mode smoke
python experiments/exp_20261124_5000_reversed/run_experiment_5000.py --mode full --confirm-full --run-id run_20261124_5000_reversed
python experiments/exp_20261124_5000_reversed/analyze_5000.py
```

See `experiments/exp_20261124_5000_reversed/reports/FINAL_REPORT.md`.
