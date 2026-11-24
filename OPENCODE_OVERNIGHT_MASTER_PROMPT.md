# OpenCode Overnight Master Prompt — Signed Competitive Influence Maximization Research

You are an autonomous coding agent working on an academic research project. Your goal is to audit, repair, implement, validate, and prepare/run the experiment pipeline for Competitive Influence Maximization on signed networks using a competitive rumor-spreading model, Salp Swarm Algorithm (SSA), C++ diffusion engine, Python orchestration, and result analysis.


## 0. STRICT SAFETY AND SCOPE BOUNDARIES

### Workspace-only access
- Your ONLY permitted filesystem workspace is the directory opened in OpenCode.
- Read, create, edit, rename, and delete files only inside that workspace.
- Do not traverse upward (`..`) or use absolute paths outside the workspace.
- Do not inspect the user's home directory, system directories, unrelated projects, Desktop, Downloads, Documents, browser data, credentials, SSH keys, environment secrets, hidden configuration, or any other files outside the opened workspace.
- Do not create temporary files outside the workspace.
- If you discover that a task would require access outside the workspace, stop that task and explain why.

### System safety
- Never use `sudo`, administrator privileges, package managers, installers, system configuration changes, driver changes, service changes, or shell/profile edits.
- Never install, uninstall, upgrade, or downgrade any software or dependencies.
- Never modify global Python environments, PATH, environment variables permanently, shell configuration, compiler configuration, or OS settings.
- If a dependency is missing, report the exact dependency and a suggested manual command, but do not run it.
- No internet/network access, downloads, external services, remote execution, telemetry, uploads, or sending any project code/data/results anywhere.
- Do not modify Git configuration, initialize a repo, commit, push, pull, reset, rebase, or perform any Git operation unless explicitly authorized later.

### File safety
- You may create new files and directories inside this project as needed without asking, provided they are safe and relevant to this project.
- Preserve original research notebooks and existing results. Never overwrite, delete, or destructively alter the original source notebooks, source datasets, existing result files, or prior generated project artifacts.
- When modifying existing files, create a backup inside the workspace first (e.g., a `backups/` folder with a timestamped copy). Avoid unnecessary backups of newly generated disposable files.
- Prefer new versioned files when there is any risk of damaging a source or existing working file.
- Do not delete files to “clean up.” If obsolete files are found, list them in your report instead.
- Avoid broad rewrite/refactor operations. Make targeted, reviewable changes.

### Execution/resource safety
- You may run local code and commands that are necessary for this project, provided they are bounded and safe, stay inside the workspace, and do not install dependencies or change the system.
- You may run small smoke tests and validation tests automatically.
- You may run the experiment pipeline overnight after correctness gates pass, but only using the resource limits below and with checkpoints. Do not launch unbounded processes, multiple competing full experiment jobs, or duplicate jobs.
- Do not execute notebooks wholesale if they may trigger long experiments without first extracting and reviewing the cells and controlling execution.
- Use timeouts on potentially hanging test commands where practical. Avoid uncontrolled memory consumption.
- If tests fail, diagnose and fix project-local code where safe. Do not hide, suppress, or bypass test failures.
- If a correctness-critical issue remains unresolved, do not run full experiments. Continue with safe implementation/tests and leave the full run unstarted.
- If the existing environment lacks required tools/libraries, do not install anything. Continue with tasks that can be completed safely, then provide setup instructions for me.

### Autonomy
- Do not wait for user confirmation between the phases listed below. Work through phases autonomously while respecting all safety boundaries.
- You are allowed to create project files, tests, source modules, notebooks, documentation, backups, and local experiment outputs needed for this work.
- You are allowed to compile project-local C++ extensions using the already-installed compiler and dependencies, as long as compilation does not install anything or modify system configuration.
- You are allowed to run bounded smoke tests, benchmark tests, and—only after validation gates pass—the full experiment.
- Do not ask questions during the overnight run. When a scientific ambiguity blocks correctness, make no unsupported guess: document the ambiguity, use a conservative safe choice only if it does not change the research definition, otherwise stop the full experiment and continue other non-dependent tasks.
- Keep an ongoing `OVERNIGHT_LOG.md` with timestamps, phases, commands (excluding sensitive data), outcomes, issues, decisions, and files changed. Do not include secrets.


## 1. RESEARCH CONTEXT

### Research goal
Study competitive influence maximization for Rumor A on a signed social network, against an opponent Rumor B. The main evaluation metric is:

**Net Advantage = Spread_A − Spread_B**

Higher Net Advantage is better. Do not confuse this evaluation metric with an objective function optimized by SSA.

### Network plan
A synthetic signed network with:
- 1,000 nodes
- 7,500 edges (verify and explicitly document whether edge count means directed arcs or undirected links; do not silently reinterpret)
- Barabási–Albert-inspired topology
- Directed edges/orientation
- Randomized orientation if an undirected BA graph is oriented
- Positive and negative signs
- Trust rating magnitudes 1–10
- Fixed network-generation seed 42

Generate the network once and use the exact same network for all B scenarios and experiments. Store network metadata and a fingerprint/hash so consistency can be checked. Explain the topology construction, edge orientation, edge-count handling, duplicate/self-loop policy, and distributions.

### Edge trust/probability mapping (experimental choice)
- Rating magnitude 1–3: probability 0.1
- Rating magnitude 4–7: probability 0.5
- Rating magnitude 8–10: probability 0.9

Positive edges transmit the same rumor. Negative edges transmit the opposite rumor, subject to the actual model rules. This mapping is a project assumption, not a canonical model parameter. Preserve it unless the source notebook proves a different implementation is intended; report discrepancies.

### Diffusion model
Competitive Maki–Thompson rumor spreading on a signed graph with Rumor A and Rumor B. The exact Python source notebook is the authority for node states, contact rules, stifling, propagation, conflict handling, synchronous/asynchronous updates, stopping criteria, and metrics. Do not invent or silently simplify model behavior. If source notebooks disagree, document differences and use the most clearly designated latest/reference notebook; do not merge conflicting scientific definitions without explaining the choice.

### Opponent B scenarios
- B1: random seeds
- B2: degree-based seeds
- B3: positive PageRank-based seeds
- B4: community-concentrated seeds

Inspect the actual source implementation to determine exact ranking, graph, tie-breaking, and community-selection logic. Document it.

### Seed budgets
K = [10, 20, 30, 40, 50]

### Candidate pool
Candidate pool size = 10*K, with intended 70% polarity-aware ranked candidates and 30% random exploration. Candidate ranking must not wrongly reward negative relationships as positive influence. Check exact candidate scoring against source code. Handle small networks, duplicates, unavailable nodes, and opponent seed exclusions robustly. Do not silently change research scoring.

### SSA
Use Salp Swarm Algorithm (SSA), preserving the source implementation and equations. Validate representation, leader/follower update, fitness evaluation, boundary/repair logic, duplicates, exactly K unique seeds, valid candidate restrictions, random seed control, and best-so-far preservation. Do not replace SSA with another optimizer.

### Six objective variants (verify source definitions and implement exact formulas)
1. Original: `Spread_A + 1.5 * Blocked_B`
2. Competition-aware: `Spread_A - Spread_B`
3. Explicit blocking: `Blocked_B`
4. Combined influence/blocking: `Spread_A + 2 * Blocked_B`
5. Gain_A + blocking: `(Spread_A - Spread_B) + 2 * Blocked_B`, where `Gain_A = Spread_A - Spread_B`
6. Budget-aware: `alpha(K) * Spread_A + beta(K) * Blocked_B`

Budget-aware coefficients:
- K=10 or 20: alpha=0.75, beta=2.25
- K=30 or 40: alpha=1.00, beta=1.75
- K=50: alpha=1.25, beta=1.50

Objectives 4 and 5 are different. Never conflate them. Write unit tests with hand-calculated values. If a source implementation differs, document the difference and resolve only when the research intent is clear.

### Blocked_B metric
Do not assume `Blocked_B = baseline_B - Spread_B` unless the reference implementation supports it. Audit baseline definition, seed-set handling, randomness/paired evaluation, caching, and interpretation. If it is unresolved, do not invent a definition or run objective variants that depend on it as if validated. Continue other independent work, and mark the experiment blocked for that portion.

### Modes
1. SSA-only: retain the SSA solution.
2. SSA + Swap refinement: start from the exact same SSA solution and apply bounded one-for-one replacements.

Run SSA once per experimental configuration and use its saved seed set for both modes. Do not rerun SSA independently for the Swap mode. Swap must preserve exactly K unique seeds, candidate restrictions, log attempts/accepted swaps, and keep the best solution if the source/approved design is elitist. The proposed maximum swap tests is 100, configurable.

### Configuration count
Per scenario: 5 K values × 6 objectives × 2 modes = 60 configurations. Across 4 scenarios = 240 configurations, assuming all objectives are scientifically implementable and all gates pass.

### Research questions/results desired
- Objective variants compared by actual final-evaluation Net Advantage
- Performance across K=10..50
- SSA vs SSA+Swap, using paired initial SSA solutions
- Spread_A vs Spread_B trade-off
- Blocking metric only if valid
- Robustness across B1..B4
- Variability/uncertainty and statistical significance, appropriately designed
- Runtime/throughput and practical cost
- Reproducibility and correctness evidence
Do not claim an objective is best because it has a high internal fitness value. Rank by the real final evaluation metric(s). Never fabricate results.


## 2. REQUIRED PROJECT ARCHITECTURE

Keep Python responsible for research orchestration and optimization:
- network loading/validation
- candidate pool generation
- opponent seed strategies
- SSA
- objective calculations
- Swap refinement
- Monte Carlo orchestration
- checkpoints/recovery
- CSV output
- statistics/plots

Keep C++ responsible for the computationally expensive per-run diffusion engine:
- graph representation/read-only adjacency
- rumor-state transitions
- signed edge behavior
- simulation loop
- per-run metrics

Use pybind11 for Python/C++ integration. Avoid excessive graph copying and avoid changing the model for speed. First establish correctness; parallelization is later.


## 3. REQUIRED PROJECT STRUCTURE (adjust if justified, keep separation)

Maintain a clean structure similar to:

- `README.md`
- `CONFIG.md`
- `CHANGELOG.md`
- `OVERNIGHT_LOG.md`
- `data/` (network and metadata; never overwrite prior network without backup)
- `cpp_engine/` (diffusion source, headers, pybind11 binding, build scripts, README)
- `common/` (Python modules for generator, validation, objectives, candidate pool, opponents, SSA, swap, C++ adapter, Monte Carlo, runner, utilities)
- `tests/` (network, diffusion, C++/Python behavior, objectives, candidate pool, opponent strategies, SSA, Swap)
- `notebooks/` (setup notebook, four scenario notebooks, combined analysis notebook)
- `results/B1/`, `results/B2/`, `results/B3/`, `results/B4/`, `results/combined/`
- `logs/`
- `analysis/figures/`

Do not place all research logic in one giant Python file or notebook. Keep modules testable and notebooks thin.


## 4. PHASED WORK — EXECUTE AUTONOMOUSLY IN ORDER

Maintain a checklist in `OVERNIGHT_LOG.md`. Do not pause for approval between phases unless a scientific decision is genuinely required or a safety boundary would be crossed.


### Phase 0 — Inventory and source-of-truth audit
1. Inventory files in the opened workspace only.
2. Identify the current starter project, notebooks, C++ files, Python files, data, and prior results.
3. Find and inspect the original/reference notebooks that are available inside the workspace. Prioritize the original/reference competitive four-B Net Advantage notebook and the six-objective correct Gain_A/Blocked_B notebook, if present. Also inspect the optimized/shared-cache variant if present.
4. Never assume filenames guarantee content; inspect notebooks programmatically, including code cells and markdown.
5. Extract the exact Python diffusion model, SSA, candidate pool, opponent strategy, objective, Blocked_B, Swap, MC, and logging definitions.
6. Create `AUDIT_REPORT.md` listing source-of-truth files, verified definitions, discrepancies, scientific risks, current defects, and a prioritized repair plan.
7. Preserve original notebooks and files unchanged.

### Phase 1 — Repair notebook formatting and project safety
1. Find notebook code cells with malformed literal escaped newline characters (e.g. text that contains `\n` where actual line breaks should be).
2. Do NOT globally replace escape sequences; preserve valid string literals.
3. Back up any notebook before modifying it.
4. Repair only confirmed malformed cells and validate notebook JSON plus Python syntax where feasible.
5. Preserve markdown, metadata, cell order, and original outputs when possible.
6. Repair project-local path/import handling, including ensuring the C++ adapter adds its module directory to `sys.path` BEFORE attempting to import the extension.
7. Document changes. Do not run notebooks wholesale.

### Phase 2 — Synthetic network generator
1. Audit the existing generator. Correct issues, but preserve the agreed network design.
2. Explicitly define directed/undirected edge-count semantics. If source materials or current design are ambiguous, choose a transparent documented interpretation that does not conflict with the intended 7,500 count; do not silently change it.
3. Generate exactly 1,000 nodes and exactly 7,500 directed arcs under the declared convention, unless the design explicitly defines the count differently and documents it.
4. Ensure node IDs are contiguous/consistent or clearly mapped.
5. Validate duplicate arcs, self-loops, sign, trust, probability, and topology properties.
6. Generate with seed 42. Repeated generation with seed 42 must produce the same network and fingerprint.
7. Report positive/negative counts, degree summary, component/connectivity summary as appropriate, and rating/probability distributions.
8. Do not overwrite existing network data. If the current network file exists, validate it and create a versioned new file only if needed.
9. Write tests for small networks and the generated network. Do not begin full experiments.

### Phase 3 — C++ diffusion engine and Python binding
1. Write a precise model specification based on the reference source code and include it in `cpp_engine/README.md` or `MODEL_SPEC.md`.
2. Implement C++ using an efficient graph structure appropriate for repeated simulations (e.g., adjacency lists, edge probability/polarity stored compactly).
3. Keep the graph read-only between simulations. Avoid re-building or copying it per Monte Carlo replicate if unnecessary.
4. Implement the exact model behavior from the reference Python implementation. Do not simplify, approximate, or change state-transition semantics for performance.
5. Expose the interface using pybind11. Return a structured result with at least Spread_A, Spread_B, ignorant count, rounds, and any source-required metrics.
6. Add clear parameter validation and errors for malformed inputs.
7. Build using only already-installed compiler/dependencies. Do not install anything.
8. If Python development headers or pybind11 are missing, record the dependency and manual install command but do not install it. Previously observed Fedora error was missing `Python.h`; do not attempt to fix system packages yourself.
9. Create deterministic tiny graph tests: no edges, one positive edge, one negative edge, each probability class, A-only reachability, B-only reachability, conflict targeting, stifling, termination, fully informed case, overlap case if allowed.
10. If a reference Python implementation is available, compare behavior. RNG implementations may differ, so use deterministic tests and distribution-level comparisons rather than demanding identical stochastic paths from different RNGs.
11. Build/import/smoke test the C++ module. Report what actually passed.
12. Do not start full experiment runs in this phase.

### Phase 4 — Performance baseline and safe optimization
1. Benchmark small bounded workloads only. Report simulations/second, runtime, and workload details.
2. Identify real bottlenecks before optimizing.
3. Do not introduce OpenMP until serial correctness is validated.
4. Propose and then, autonomously within this prompt's permission, implement project-local OpenMP MC parallelization only if the toolchain already supports it and it can be done safely without changing system settings or installing dependencies.
5. If OpenMP support is unavailable, leave a serial implementation and document how I can enable it manually; do not install or change compiler configuration.
6. Use independent RNG streams and deterministic seed derivation. Ensure outputs do not depend on thread scheduling. Avoid shared mutable state/races.
7. Validate serial vs parallel results with fixed seed design and tests; report speedup. Do not launch concurrent full runs.

### Phase 5 — Python optimizer and experiment logic
1. Audit and fix candidate pool, SSA, opponent strategy functions, objective formulas, Blocked_B, Swap, MC evaluation, and caching.
2. Preserve source algorithm/equations. Fix bugs without silently changing methodology.
3. Use tests for every objective with hand-calculated examples.
4. Ensure candidate pool size is `10*K` when possible; handle limits when fewer eligible nodes exist. 70/30 composition should be rounded and documented consistently.
5. Ensure polarity-aware scoring does not reward negative edges as positive influence.
6. Ensure every solution has exactly K unique valid seeds and respects any B seed exclusions in the source method.
7. Test B1–B4 strategy details and deterministic tie handling.
8. Resolve Blocked_B exactly. If scientifically ambiguous, do not guess. Clearly mark dependent objective variants as blocked and continue with valid objectives only; do not label incomplete results as full six-variant results.
9. Ensure SSA is run once per configuration and its selected seed set is saved. The SSA-only and SSA+Swap modes must share that seed set.
10. Swap should be one-for-one and bounded, preserve K, log attempts/acceptances, and maintain the best valid solution according to the intended acceptance policy.
11. Avoid caching stochastic objective values incorrectly. Cache keys must include all relevant variables, graph fingerprint, scenario, seed sets, objective settings, and random/evaluation protocol. If objective evaluations use common random numbers, document it; otherwise use a sound reproducible design.
12. Keep all Python code in maintainable modules and add tests. Do not run full experiment until validation gates are passed.

### Phase 6 — Experiment infrastructure and notebooks
Create or repair these notebooks:
- `notebooks/00_Setup_Synthetic_Network.ipynb`
- `notebooks/B1_Random_opponent.ipynb`
- `notebooks/B2_Degree-based_opponent.ipynb`
- `notebooks/B3_Positive_PageRank_opponent.ipynb`
- `notebooks/B4_Community-concentrated_opponent.ipynb`
- `notebooks/05_Analysis_Charts.ipynb`

Notebook code must have genuine line breaks, not literal escaped `\n` inserted between statements.

Each B notebook imports shared code, loads the shared network, validates its fingerprint, defines configurable experiment parameters, runs that scenario only, and writes its own timestamped results/checkpoints.

Scenario run target (if all six objectives are valid): 5 K × 6 objectives × 2 modes = 60 configurations per scenario.

Each notebook should support safe restart/resume without duplicating completed configurations. Persist seed sets, configs, progress, results, and checkpoint metadata. Never silently overwrite earlier runs. If there are multiple runs, use unique run directories or explicit run IDs.

Use a CSV schema containing at least:
- run_id, scenario, K, objective_id/name, mode
- master seed and derived seeds/protocol
- optimization/final MC counts
- SSA population and iterations
- candidate pool size
- initial/final optimizer objective score
- Spread_A, Spread_B, Net_Advantage, Blocked_B (only valid), Gain_A
- ignorant_nodes, average_rounds
- SSA, Swap, total runtimes
- Swap tests/accepted swaps
- A seed set and B seed set (serialized safely)
- network fingerprint
- summary statistics and uncertainty fields when correctly calculated

Use a warning/confirmation cell for a notebook's full experiment, but also make command-line runner scripts available for robust overnight execution. Do not rely solely on Jupyter cell state.

### Phase 7 — Experiment-run decision and smoke run
Before full experiments, create a validation gate report (`VALIDATION_GATE.md`). The full run is allowed only if:
- network tests pass
- C++ compilation/import passes (or clearly documented fallback if C++ is not available; do not run large Python fallback silently)
- diffusion tests pass
- Python/C++ model comparison is acceptable and documented
- objective tests pass
- candidate-pool tests pass
- opponent strategy tests pass
- SSA tests pass
- Swap tests pass
- checkpoint/resume tests pass
- no unresolved scientific ambiguity affects the chosen experiment
- estimated workload is based on an actual bounded benchmark

Run a small smoke experiment (e.g., a single scenario, small K, very small population/iterations/MC) and validate the output schema, seed pairing, checkpoint/resume, and plotting ingestion. Keep it clearly separate from final experiment results and label it `SMOKE_TEST`.

If the validation gate fails, DO NOT start full experiments. Continue with independent coding, tests, documentation, notebooks, and analysis pipeline, and leave a precise report of the blockers.

### Phase 8 — Full overnight experiment (only if validation gates pass)
If all gates pass, execute the full experiment in an orderly, checkpointed manner, using resource limits and a single run coordinator. Do not launch four competing jobs at once unless the measured resource budget demonstrates that safe parallel scenario execution is better; default to sequential scenarios to prevent resource exhaustion.

Run four scenarios, each 60 configurations (or clearly mark any scientifically blocked configurations). Use the shared network and paired SSA/Swap starting sets. Use a single master run ID and separate scenario result folders. Store per-configuration Monte Carlo raw outcomes where feasible, plus aggregated records. Checkpoint frequently and resume after interruptions.

Avoid excessive resource use: use measured benchmark data to choose reasonable worker/thread counts, do not oversubscribe CPU, and keep memory bounded. If runtime is too long for the available overnight window, complete a safe subset in a declared order, checkpoint it, and report incomplete configurations rather than running uncontrolled processes.

Never fake, fill in, or infer missing results. If execution is interrupted, report exactly what finished, what remains, and how to resume.

### Phase 9 — Analysis and visualization notebook
Create `notebooks/05_Analysis_Charts.ipynb`. It must only read stored results; it must never launch simulations.

Validate required columns, scenario labels, K, objectives, modes, duplicates, incomplete configurations, network fingerprints, and parameter consistency. If multiple run folders exist, require/allow explicit run selection and never silently mix configurations.

Create publication-quality charts and tables, including:
1. Net Advantage vs K by objective, scenario, and mode
2. SSA vs SSA+Swap paired differences (absolute and percentage only where denominator is meaningful)
3. Spread_A vs Spread_B trade-off
4. Blocked_B, only if valid
5. Runtime vs K/objective/scenario/mode
6. Objective rankings using final Net Advantage, not optimizer fitness
7. Robustness across B1–B4
8. Variability and uncertainty using valid Monte Carlo and/or independent optimizer-run data

Create tables: overall, per-scenario, per-K, Swap comparison, objective rankings, runtime, variability, best configuration per scenario/K. Use appropriate paired tests only if data structure supports them; distinguish MC variability from between-optimizer-run variability and avoid pseudoreplication. Do not claim significance based on invalid units of analysis. Save figures and tables within the workspace, with unique names to avoid overwriting.

### Phase 10 — Final research and implementation audit
Perform a read-only final review. Create `FINAL_AUDIT.md` with:
- final project structure
- implementation/source file inventory
- build/test/run instructions
- notebook order
- configuration used
- exact result files generated
- validation results and test commands
- research correctness findings
- C++/Python parity findings
- benchmark/speedup results if measured
- full experiment completion status
- result summary only from actual data
- missing configurations and why
- limitations and next steps
Classify each item as PASS, FAIL, NOT TESTED, or RESEARCHER DECISION REQUIRED. Do not claim full scientific validation unless tests and experiments actually support it.


## 5. EXPERIMENT OUTPUT AND FAIRNESS RULES

- Every configuration should be reproducible from saved configuration and seed information.
- Use consistent network fingerprint across scenarios.
- Opponent seed sets should be saved.
- SSA selected seed sets must be saved.
- Swap initial and final seed sets must be saved.
- SSA-only and SSA+Swap evaluation must be paired appropriately.
- Report raw metrics and means/variability, not only one score.
- Separate optimizer objective from final evaluation objective.
- Keep smoke-test results separate from real experiment results.
- Never silently omit failures or missing configurations.
- Keep the original/reference implementation untouched.
- Write checkpoints atomically where possible (write temp file within workspace then replace target) to reduce corruption risk; never use temp files outside workspace.


## 6. AUTONOMOUS DECISION RULES

Proceed autonomously for safe, reversible, project-local tasks. If a scientific choice is unresolved:
- search the supplied local source notebooks and project documentation first
- record the evidence and discrepancy
- do not guess a scientifically meaningful formula
- continue independent work
- if the full experiment depends on the unresolved choice, stop before full execution and clearly report the blocker

Do not wait for a response during the overnight run. Instead, maximize safe progress and produce a complete status report.


## 7. FINAL MESSAGE TO ME

At the end, provide a concise but complete summary:
- what you accomplished
- whether C++ compiled/imported
- tests run and passed/failed
- whether the full experiment was launched
- how many configurations completed out of the planned 240 (or adjusted valid total)
- exact result/figure/report file paths within the project
- what remains
- any commands I must run manually (especially dependency installation, but never install them yourself)
- the single best next action for me

Remember: safety, scientific correctness, reproducibility, and preserving my original files take priority over completing all tasks overnight.
