# IM Synthetic Network Experiment (Starter Build)

This project separates the six-variant SSA experiment into four scenario notebooks and a fifth analysis notebook.

## Folder contents
- `data/`: generated shared synthetic signed network and metadata
- `cpp_engine/`: C++ competitive signed Maki–Thompson diffusion module (pybind11)
- `common/`: Python experiment logic, SSA, objectives, candidate-pool generation, scenario runner, adapter
- `notebooks/00_Setup_Synthetic_Network.ipynb`: create the shared 1000-node / 7500-edge network
- `notebooks/B1_Random_opponent.ipynb`: random B strategy
- `notebooks/B2_Degree-based_opponent.ipynb`: degree-based B strategy
- `notebooks/B3_Positive_PageRank_opponent.ipynb`: positive PageRank B strategy
- `notebooks/B4_Community-concentrated_opponent.ipynb`: community-concentrated B strategy
- `notebooks/05_Analysis_Charts.ipynb`: reads all scenario CSVs and plots comparisons
- `results/`: per-scenario result and checkpoint folders

## Network assumptions in this starter
- 1,000 nodes with IDs 0–999
- 7,500 unique directed edges
- A Barabási–Albert-style preferential attachment backbone, randomly oriented, then supplemented with random directed edges to reach exactly 7,500
- Each edge gets a random sign with 50% positive probability
- Trust magnitude is sampled uniformly from integers 1–10; rating is signed by the edge sign
- Rating-to-transmission mapping: magnitude 1–3 → 0.1; 4–7 → 0.5; 8–10 → 0.9
- Generation seed 42; one shared graph file used by B1–B4

These are experiment-design choices, not properties of a real dataset. They should be documented and justified in the research methodology.


## Local setup
Use a local Jupyter environment (preferably a dedicated virtual environment). From the project root:


```bash
python -m pip install numpy pandas networkx matplotlib jupyter pybind11 setuptools
```

Compile the extension once:

```bash
cd cpp_engine
python setup.py build_ext --inplace
cd ..
```

Open Jupyter from the project root:

```bash
jupyter notebook
```

Run `00_Setup_Synthetic_Network.ipynb` first, then B1–B4 in any order, then `05_Analysis_Charts.ipynb`.

## Research validation warning
This is a **starter implementation**. The C++ diffusion port uses a C++ random-number generator and is not expected to reproduce NumPy random streams seed-for-seed. Before a full research run:
1. Create tiny hand-checkable directed signed graphs.
2. Confirm edge orientation, one-contact-per-neighbor behavior, stifling, negative-edge rumor flipping, conflict resolution, and final spread counts against the original Python simulator.
3. Compare Monte Carlo distributions/means on fixed small graphs, not only one stochastic run.
4. Benchmark Python and C++ engines.
5. Review all six objectives and the bounded sampled Swap code for correctness.

The current C++ implementation is single-threaded; OpenMP batch parallelization is a next optimization after correctness validation. No experiment CSVs are populated yet.
