"""Generate the project notebooks.

Run from the project root:

    python tools/build_notebooks.py

The generated notebooks are thin wrappers around the tested modules.  They have
real line breaks (no literal ``\\n`` artefacts), a safe default that runs
nothing, and an explicit confirmation for a full experiment.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
KERNELSPEC = {
    "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3",
    },
}


def notebook(cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = dict(KERNELSPEC)
    return nb


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


def setup_notebook():
    cells = [
        md(
            "# Setup — Shared Synthetic Signed Network\n\n"
            "This notebook validates the shared **1,000-node / 7,500-directed-arc** signed network. "
            "The network is source data: do **not** regenerate or overwrite it unless the file is missing. "
            "Generation seed is 42 and the network is reused by every B1–B4 experiment."
        ),
        code(
            "from pathlib import Path\n"
            "import json\n"
            "import sys\n"
            "\n"
            "ROOT = Path.cwd()\n"
            "if ROOT.name == 'notebooks':\n"
            "    ROOT = ROOT.parent\n"
            "for p in [ROOT, ROOT / 'common', ROOT / 'cpp_engine']:\n"
            "    if str(p) not in sys.path:\n"
            "        sys.path.insert(0, str(p))\n"
            "\n"
            "from network import NETWORK_CSV, NETWORK_METADATA_V2, load_network, validate_network, verify_fingerprint, write_metadata_v2\n"
            "\n"
            "print('Project root:', ROOT)\n"
            "print('Network file:', NETWORK_CSV)\n"
        ),
        code(
            "df, G, nodes = load_network(NETWORK_CSV)\n"
            "report = validate_network(df)\n"
            "print(f'nodes={len(nodes)}  directed_arcs={len(df)}  weak_components={report[\"n_weak_components\"]}')\n"
            "print(json.dumps(report, indent=2))\n"
        ),
        md(
            "## Fingerprint and metadata\n\n"
            "The v2 metadata stores the file SHA-256 and a semantic edge-table SHA-256. "
            "All scenario notebooks verify this fingerprint before running."
        ),
        code(
            "if not NETWORK_METADATA_V2.exists():\n"
            "    write_metadata_v2(df, NETWORK_CSV, NETWORK_METADATA_V2)\n"
            "    print('Wrote', NETWORK_METADATA_V2)\n"
            "check = verify_fingerprint(NETWORK_CSV, NETWORK_METADATA_V2)\n"
            "print(json.dumps(check, indent=2))\n"
            "assert check['ok'], 'Network fingerprint mismatch: stop and investigate.'\n"
        ),
        md(
            "## Optional seed-42 regeneration check\n\n"
            "This is off by default. It regenerates the network in memory and compares it to the stored file; "
            "it never overwrites the stored network."
        ),
        code(
            "REGENERATE_CHECK = False\n"
            "if REGENERATE_CHECK:\n"
            "    import pandas as pd\n"
            "    from network import regenerate_network_df\n"
            "    regenerated = regenerate_network_df()\n"
            "    pd.testing.assert_frame_equal(regenerated.reset_index(drop=True), df.reset_index(drop=True))\n"
            "    print('Seed-42 regeneration matches the stored network.')\n"
            "else:\n"
            "    print('Regeneration check skipped (set REGENERATE_CHECK=True to verify).')\n"
        ),
        md(
            "## C++ engine status\n\n"
            "Build the extension once from a terminal in the project root (no installation is performed):\n\n"
            "```bash\n"
            "cd cpp_engine && python setup.py build_ext --inplace\n"
            "```"
        ),
        code(
            "from cpp_adapter import cpp_available, openmp_available, cpp_import_error\n"
            "print('C++ available:', cpp_available())\n"
            "print('OpenMP:', openmp_available())\n"
            "if not cpp_available():\n"
            "    print('Import error:', cpp_import_error())\n"
            "    print('Build manually: cd cpp_engine && python setup.py build_ext --inplace')\n"
        ),
        md(
            "## Next steps\n\n"
            "Run `B1_Random_opponent.ipynb`, `B2_Degree-based_opponent.ipynb`, "
            "`B3_Positive_PageRank_opponent.ipynb`, and `B4_Community-concentrated_opponent.ipynb` "
            "in any order, then `05_Analysis_Charts.ipynb`. For robust overnight execution, prefer the CLI:\n\n"
            "```bash\n"
            "python common/run_experiment.py --print-plan\n"
            "python common/run_experiment.py --run-id run_YYYYMMDD_HHMMSS\n"
            "```"
        ),
    ]
    return notebook(cells)


def scenario_notebook(scenario_id, title, description):
    cells = [
        md(
            f"# {scenario_id} Experiment — {title}\n\n"
            f"{description}\n\n"
            "Target: 5 K values × 6 objectives × 2 modes = **60 configurations**. "
            "SSA is run once per (K, objective) and its saved seed set is shared by SSA-only and SSA+Swap. "
            "Results are checkpointed and written under `results/<run_id>/<scenario>/`."
        ),
        code(
            "from pathlib import Path\n"
            "import json\n"
            "import sys\n"
            "\n"
            "ROOT = Path.cwd()\n"
            "if ROOT.name == 'notebooks':\n"
            "    ROOT = ROOT.parent\n"
            "for p in [ROOT, ROOT / 'common', ROOT / 'cpp_engine']:\n"
            "    if str(p) not in sys.path:\n"
            "        sys.path.insert(0, str(p))\n"
            "\n"
            "from network import NETWORK_CSV, NETWORK_METADATA_V2, load_network, verify_fingerprint\n"
            "from experiment import ExperimentConfig, ExperimentRunner, MODE_SSA, MODE_SWAP, default_run_id\n"
            "from objectives import VARIANTS\n"
            "\n"
            f"SCENARIO = '{scenario_id}'\n"
        ),
        md("## Shared-network fingerprint check"),
        code(
            "check = verify_fingerprint(NETWORK_CSV, NETWORK_METADATA_V2)\n"
            "print(json.dumps(check, indent=2))\n"
            "assert check['ok'], 'Network fingerprint mismatch: stop before running experiments.'\n"
            "df, G, nodes = load_network(NETWORK_CSV)\n"
            "print(f'Shared network: {len(nodes)} nodes, {G.number_of_edges()} directed arcs')\n"
        ),
        md(
            "## Configuration\n\n"
            "`RUN_MODE` defaults to `'none'`, so **Run All does not launch an experiment**. "
            "Set it to `'smoke'` for a tiny pipeline check or `'full'` for the 60-configuration scenario. "
            "A full run additionally requires `CONFIRM_FULL_RUN = True`."
        ),
        code(
            "RUN_MODE = 'none'          # 'none', 'smoke', or 'full'\n"
            "CONFIRM_FULL_RUN = False   # must be True for RUN_MODE == 'full'\n"
            "RUN_ID = None              # set a name to resume a specific run\n"
            "\n"
            "K_VALUES = (10, 20, 30, 40, 50)\n"
            "OBJECTIVE_IDS = tuple(VARIANTS.keys())\n"
            "MODES = (MODE_SSA, MODE_SWAP)\n"
            "SSA_POPULATION = 12\n"
            "SSA_ITERATIONS = 10\n"
            "SSA_MUTATION_RATE = 0.20\n"
            "OPTIMIZATION_MC_RUNS = 15\n"
            "FINAL_MC_RUNS = 100\n"
            "SWAP_MAX_TESTS = 100\n"
            "THREADS = 0                # 0 = OpenMP runtime default\n"
            "RESUME = True\n"
            "\n"
            "print('Planned configurations:', len(K_VALUES) * len(OBJECTIVE_IDS) * len(MODES))\n"
        ),
        md("## Safety confirmation"),
        code(
            "if RUN_MODE == 'full' and not CONFIRM_FULL_RUN:\n"
            "    raise RuntimeError('Full run not confirmed. Set CONFIRM_FULL_RUN=True after reviewing the plan.')\n"
            "if RUN_MODE == 'none':\n"
            "    print('Nothing will run. Set RUN_MODE to smoke or full.')\n"
        ),
        md("## Run (checkpointed and resumable)"),
        code(
            "if RUN_MODE != 'none':\n"
            "    run_kind = 'SMOKE_TEST' if RUN_MODE == 'smoke' else 'full'\n"
            "    run_id = RUN_ID or default_run_id(run_kind)\n"
            "    if RUN_MODE == 'smoke':\n"
            "        config = ExperimentConfig(\n"
            "            run_id=run_id, run_kind='SMOKE_TEST', scenarios=(SCENARIO,),\n"
            "            k_values=(10,), objective_ids=('O2',), modes=MODES,\n"
            "            ssa_population=4, ssa_iterations=2, optimization_mc_runs=5,\n"
            "            final_mc_runs=10, swap_max_tests=10, n_threads=THREADS,\n"
            "            network_path=NETWORK_CSV, results_root=ROOT / 'results' / 'smoke',\n"
            "            resume=RESUME,\n"
            "        )\n"
            "    else:\n"
            "        config = ExperimentConfig(\n"
            "            run_id=run_id, run_kind='full', scenarios=(SCENARIO,),\n"
            "            k_values=K_VALUES, objective_ids=OBJECTIVE_IDS, modes=MODES,\n"
            "            ssa_population=SSA_POPULATION, ssa_iterations=SSA_ITERATIONS,\n"
            "            ssa_mutation_rate=SSA_MUTATION_RATE,\n"
            "            optimization_mc_runs=OPTIMIZATION_MC_RUNS, final_mc_runs=FINAL_MC_RUNS,\n"
            "            swap_max_tests=SWAP_MAX_TESTS, n_threads=THREADS,\n"
            "            network_path=NETWORK_CSV, results_root=ROOT / 'results',\n"
            "            resume=RESUME,\n"
            "        )\n"
            "    print('Run directory:', config.run_dir)\n"
            "    runner = ExperimentRunner(config)\n"
            "    manifest = runner.run_all()\n"
            "    print(json.dumps(manifest, indent=2))\n"
        ),
        md("## Results table"),
        code(
            "if RUN_MODE != 'none':\n"
            "    import pandas as pd\n"
            "    results = pd.read_csv(config.scenario_dir(SCENARIO) / 'experiment_results.csv')\n"
            "    columns = ['K', 'objective_id', 'mode', 'spread_A', 'spread_B', 'net_advantage', 'blocked_B', 'objective_validity']\n"
            "    print(results[columns].to_string(index=False))\n"
        ),
        md("## Quick view"),
        code(
            "if RUN_MODE != 'none':\n"
            "    import matplotlib\n"
            "    matplotlib.use('Agg')\n"
            "    import matplotlib.pyplot as plt\n"
            "    fig, ax = plt.subplots(figsize=(9, 5))\n"
            "    for oid in results['objective_id'].unique():\n"
            "        for mode in results['mode'].unique():\n"
            "            s = results[(results['objective_id'] == oid) & (results['mode'] == mode)].sort_values('K')\n"
            "            if not s.empty:\n"
            "                ax.plot(s['K'], s['net_advantage'], marker='o', label=f'{oid} {mode}')\n"
            "    ax.axhline(0, color='black', linewidth=0.8)\n"
            "    ax.set_xlabel('K')\n"
            "    ax.set_ylabel('Net Advantage')\n"
            "    ax.set_title(f'{SCENARIO} quick view')\n"
            "    ax.legend(fontsize=7)\n"
            "    plt.show()\n"
        ),
        md(
            "## CLI alternative (recommended for overnight runs)\n\n"
            "```bash\n"
            f"python common/run_experiment.py --scenarios {scenario_id} --run-id run_YYYYMMDD_HHMMSS\n"
            "```\n\n"
            "Re-running the same `--run-id` resumes without duplicating completed configurations."
        ),
    ]
    return notebook(cells)


def analysis_notebook():
    cells = [
        md(
            "# Analysis — Charts and Tables\n\n"
            "**Read-only.** This notebook never launches simulations. It loads stored scenario CSVs and per-seed "
            "Monte Carlo files, validates them, and writes tables/figures under `analysis/`."
        ),
        code(
            "from pathlib import Path\n"
            "import json\n"
            "import sys\n"
            "\n"
            "ROOT = Path.cwd()\n"
            "if ROOT.name == 'notebooks':\n"
            "    ROOT = ROOT.parent\n"
            "for p in [ROOT, ROOT / 'common', ROOT / 'cpp_engine']:\n"
            "    if str(p) not in sys.path:\n"
            "        sys.path.insert(0, str(p))\n"
            "\n"
            "from analysis import discover_runs, select_run, run_analysis\n"
            "\n"
            "RUN_ID = None          # set explicitly to choose a run\n"
            "INCLUDE_SMOKE = False  # keep smoke tests separate from real results\n"
        ),
        md("## Available runs"),
        code(
            "runs = discover_runs(ROOT / 'results', include_smoke=INCLUDE_SMOKE)\n"
            "print(json.dumps(runs, indent=2)[:6000])\n"
            "if not runs:\n"
            "    print('No runs found yet. Run a B1-B4 notebook or the CLI first.')\n"
        ),
        md("## Select and validate one run"),
        code(
            "run_dir = None\n"
            "try:\n"
            "    run_dir = select_run(RUN_ID, results_root=ROOT / 'results', include_smoke=INCLUDE_SMOKE)\n"
            "    print('Selected run:', run_dir)\n"
            "except FileNotFoundError as exc:\n"
            "    print('No run available:', exc)\n"
        ),
        md("## Build tables and figures"),
        code(
            "result = None\n"
            "if run_dir is not None:\n"
            "    result = run_analysis(run_dir=run_dir, make_figures=True, n_boot=2000)\n"
            "    print('Validation:', result['report']['ok'])\n"
            "    print('Issues:', result['report']['issues'])\n"
            "    print('Report:', result['report_path'])\n"
            "    print('Figures:', result['figure_dir'])\n"
        ),
        md("## Tables"),
        code(
            "from IPython.display import display\n"
            "if result is not None:\n"
            "    for name in ['overall', 'objective_ranking', 'mode_comparison', 'best_config', 'runtime', 'robustness']:\n"
            "        print('##', name)\n"
            "        display(result['tables'][name])\n"
        ),
        md("## Figures"),
        code(
            "from IPython.display import Image, display\n"
            "if result is not None:\n"
            "    for path in result['figure_paths']:\n"
            "        print(path)\n"
            "        display(Image(filename=path))\n"
        ),
        md(
            "## Interpretation guardrails\n\n"
            "- Rank objectives by **final Net Advantage**, never by optimizer fitness.\n"
            "- O2 (Competition-aware) is the only fully validated objective.\n"
            "- O1/O3/O4/O5/O6 use the provisional paired B-only `Blocked_B` protocol; their results are exploratory.\n"
            "- Monte Carlo replicates are common random numbers, not independent optimizer runs. "
            "Do not treat MC replicates as between-run replication.\n"
            "- Swap vs SSA comparisons are paired by MC seed within the same scenario/K/objective."
        ),
    ]
    return notebook(cells)


def main():
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    notebooks = {
        "00_Setup_Synthetic_Network.ipynb": setup_notebook(),
        "B1_Random_opponent.ipynb": scenario_notebook(
            "B1", "Random opponent",
            "B1 samples K opponent seeds uniformly at random from all nodes (fixed order, nested across K).",
        ),
        "B2_Degree-based_opponent.ipynb": scenario_notebook(
            "B2", "Degree-based opponent",
            "B2 selects the highest out-degree nodes, with ties broken by ascending node id.",
        ),
        "B3_Positive_PageRank_opponent.ipynb": scenario_notebook(
            "B3", "Positive PageRank opponent",
            "B3 ranks nodes by PageRank on the positive-only directed subgraph, weighted by transmission probability.",
        ),
        "B4_Community-concentrated_opponent.ipynb": scenario_notebook(
            "B4", "Community-concentrated opponent",
            "B4 selects the largest Louvain community on the undirected projection, then ranks by out-degree.",
        ),
        "05_Analysis_Charts.ipynb": analysis_notebook(),
    }
    for name, nb in notebooks.items():
        path = NOTEBOOK_DIR / name
        nbf.write(nb, str(path))
        print("wrote", path)


if __name__ == "__main__":
    main()
