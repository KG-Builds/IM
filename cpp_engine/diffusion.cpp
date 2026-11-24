#include "diffusion.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

std::string seed_error(const char* label, int id) {
    return std::string("Unknown ") + label + " seed: " + std::to_string(id);
}

}  // namespace

DiffusionEngine::DiffusionEngine(
    const std::vector<int>& node_ids,
    const std::vector<std::vector<int>>& adj,
    const std::vector<std::vector<double>>& probs,
    const std::vector<std::vector<int>>& polarity,
    const std::vector<std::vector<int>>& strength)
    : n_(static_cast<int>(node_ids.size())), node_ids_(node_ids) {
    if (adj.size() != static_cast<size_t>(n_) ||
        probs.size() != static_cast<size_t>(n_) ||
        polarity.size() != static_cast<size_t>(n_) ||
        strength.size() != static_cast<size_t>(n_)) {
        throw std::invalid_argument("All adjacency arrays must have length n");
    }

    idx_.reserve(static_cast<size_t>(n_) * 2);
    for (int i = 0; i < n_; ++i) {
        if (idx_.count(node_ids_[i])) {
            throw std::invalid_argument("Duplicate node id: " + std::to_string(node_ids_[i]));
        }
        idx_[node_ids_[i]] = i;
    }

    offsets_.assign(static_cast<size_t>(n_) + 1, 0);
    size_t total_edges = 0;
    for (int u = 0; u < n_; ++u) {
        const size_t deg = adj[u].size();
        if (probs[u].size() != deg || polarity[u].size() != deg || strength[u].size() != deg) {
            throw std::invalid_argument("Per-node adjacency arrays have inconsistent lengths");
        }
        total_edges += deg;
    }
    targets_.reserve(total_edges);
    probs_.reserve(total_edges);
    polarity_.reserve(total_edges);
    strength_.reserve(total_edges);

    for (int u = 0; u < n_; ++u) {
        const size_t deg = adj[u].size();
        for (size_t j = 0; j < deg; ++j) {
            const int v = adj[u][j];
            if (v < 0 || v >= n_) {
                throw std::invalid_argument("Edge target index out of range");
            }
            const double p = probs[u][j];
            if (!std::isfinite(p) || p < 0.0 || p > 1.0) {
                throw std::invalid_argument("Edge probability outside [0,1]");
            }
            const int pol = polarity[u][j];
            if (pol != 1 && pol != -1) {
                throw std::invalid_argument("Edge polarity must be +1 or -1");
            }
            const int s = strength[u][j];
            if (s < 1 || s > 10) {
                throw std::invalid_argument("Edge strength must be in 1..10");
            }
            targets_.push_back(v);
            probs_.push_back(p);
            polarity_.push_back(static_cast<signed char>(pol));
            strength_.push_back(s);
        }
        offsets_[static_cast<size_t>(u) + 1] = static_cast<int>(targets_.size());
    }
    m_ = static_cast<int>(targets_.size());
}

void DiffusionEngine::validate_seeds(const std::vector<int>& seeds_A,
                                     const std::vector<int>& seeds_B) const {
    std::unordered_set<int> a;
    a.reserve(seeds_A.size() * 2 + 1);
    for (int id : seeds_A) {
        if (!idx_.count(id)) throw std::invalid_argument(seed_error("A", id));
        a.insert(id);
    }
    for (int id : seeds_B) {
        if (!idx_.count(id)) throw std::invalid_argument(seed_error("B", id));
        if (a.count(id)) throw std::invalid_argument("A/B seed overlap");
    }
}

SimResult DiffusionEngine::run(const std::vector<int>& seeds_A,
                               const std::vector<int>& seeds_B,
                               uint64_t seed,
                               Workspace& ws) const {
    std::fill(ws.state.begin(), ws.state.end(), 0);
    std::fill(ws.contacted.begin(), ws.contacted.end(), 0);
    std::fill(ws.best_strength.begin(), ws.best_strength.end(), -1);
    std::fill(ws.rumor_mask.begin(), ws.rumor_mask.end(), 0);
    ws.stiflers.clear();

    for (int id : seeds_A) {
        auto it = idx_.find(id);
        if (it == idx_.end()) throw std::invalid_argument(seed_error("A", id));
        ws.state[it->second] = 1;
    }
    for (int id : seeds_B) {
        auto it = idx_.find(id);
        if (it == idx_.end()) throw std::invalid_argument(seed_error("B", id));
        const int i = it->second;
        if (ws.state[i] != 0) throw std::invalid_argument("A/B seed overlap");
        ws.state[i] = 2;
    }

    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<double> real(0.0, 1.0);
    int round_no = 0;

    while (true) {
        ws.old = ws.state;
        ws.next = ws.state;
        bool any_spreader = false;
        ws.stiflers.clear();
        std::fill(ws.best_strength.begin(), ws.best_strength.end(), -1);
        std::fill(ws.rumor_mask.begin(), ws.rumor_mask.end(), 0);

        for (int u = 0; u < n_; ++u) {
            const int su = ws.old[u];
            if (su != 1 && su != 2) continue;
            any_spreader = true;

            const int begin = offsets_[u];
            const int end = offsets_[u + 1];
            int available = 0;
            for (int e = begin; e < end; ++e) {
                if (!ws.contacted[e]) ++available;
            }
            if (available == 0) {
                ws.stiflers.push_back(u);
                continue;
            }

            // Same RNG call and selection rule as the starter implementation.
            std::uniform_int_distribution<int> pick(0, available - 1);
            const int chosen = pick(gen);
            int e = begin;
            int seen = 0;
            for (; e < end; ++e) {
                if (!ws.contacted[e]) {
                    if (seen == chosen) break;
                    ++seen;
                }
            }
            ws.contacted[e] = 1;
            const int v = targets_[e];
            if (ws.old[v] != 0) {
                ws.stiflers.push_back(u);
                continue;
            }
            if (real(gen) < probs_[e]) {
                int rumor = (su == 1) ? 1 : 2;
                if (polarity_[e] < 0) rumor = (rumor == 1) ? 2 : 1;
                const int s = strength_[e];
                if (s > ws.best_strength[v]) {
                    ws.best_strength[v] = s;
                    ws.rumor_mask[v] = static_cast<unsigned char>((rumor == 1) ? 1 : 2);
                } else if (s == ws.best_strength[v]) {
                    ws.rumor_mask[v] = static_cast<unsigned char>(
                        ws.rumor_mask[v] | ((rumor == 1) ? 1 : 2));
                }
            }
        }

        if (!any_spreader) break;

        // Resolve proposals in ascending target order, consuming one RNG draw
        // per contested target (matching the starter's stream exactly).
        for (int v = 0; v < n_; ++v) {
            const unsigned char mask = ws.rumor_mask[v];
            if (mask == 0) continue;
            int contenders[2];
            int count = 0;
            if (mask & 1) contenders[count++] = 1;
            if (mask & 2) contenders[count++] = 2;
            std::uniform_int_distribution<int> pick(0, count - 1);
            ws.next[v] = contenders[pick(gen)];
        }

        for (int u : ws.stiflers) {
            ws.next[u] = (ws.old[u] == 1) ? 3 : 4;
        }

        ws.state.swap(ws.next);
        ++round_no;
        if (round_no > 7501) {
            throw std::runtime_error("Simulation exceeded finite-contact safety limit");
        }
    }

    int A = 0;
    int B = 0;
    int I = 0;
    for (int s : ws.state) {
        if (s == 1 || s == 3) {
            ++A;
        } else if (s == 2 || s == 4) {
            ++B;
        } else {
            ++I;
        }
    }
    // Reference-notebook semantics: the returned round counter includes the
    // terminal empty round in which no spreader remained (executed_rounds + 1).
    return SimResult{A, B, I, round_no + 1};
}

SimResult DiffusionEngine::simulate(const std::vector<int>& seeds_A,
                                    const std::vector<int>& seeds_B,
                                    uint64_t seed) const {
    validate_seeds(seeds_A, seeds_B);
    Workspace ws(n_, m_);
    return run(seeds_A, seeds_B, seed, ws);
}

std::vector<SimResult> DiffusionEngine::simulate_batch(
    const std::vector<std::vector<int>>& seeds_A,
    const std::vector<std::vector<int>>& seeds_B,
    const std::vector<uint64_t>& seeds,
    int n_threads) const {
    const size_t nruns = seeds.size();
    if (seeds_A.size() != nruns || seeds_B.size() != nruns) {
        throw std::invalid_argument("simulate_batch: seeds_A, seeds_B and seeds must have equal length");
    }
    if (nruns == 0) return {};

    // Validate serially so exceptions never escape an OpenMP region.
    for (size_t r = 0; r < nruns; ++r) {
        validate_seeds(seeds_A[r], seeds_B[r]);
    }

    std::vector<SimResult> out(nruns);
    int threads = n_threads;
#ifdef _OPENMP
    if (threads <= 0) threads = omp_get_max_threads();
#else
    threads = 1;
    (void)threads;
#endif

#ifdef _OPENMP
#pragma omp parallel num_threads(threads) if (threads > 1 && nruns > 1)
#endif
    {
        Workspace ws(n_, m_);
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 4)
#endif
        for (long long r = 0; r < static_cast<long long>(nruns); ++r) {
            out[static_cast<size_t>(r)] =
                run(seeds_A[static_cast<size_t>(r)],
                    seeds_B[static_cast<size_t>(r)],
                    seeds[static_cast<size_t>(r)],
                    ws);
        }
    }
    return out;
}

std::vector<SimResult> DiffusionEngine::simulate_many(
    const std::vector<int>& seeds_A,
    const std::vector<int>& seeds_B,
    const std::vector<uint64_t>& seeds,
    int n_threads) const {
    std::vector<std::vector<int>> all_A(seeds.size(), seeds_A);
    std::vector<std::vector<int>> all_B(seeds.size(), seeds_B);
    return simulate_batch(all_A, all_B, seeds, n_threads);
}

SimResult simulate(const std::vector<int>& node_ids,
                   const std::vector<std::vector<int>>& adj,
                   const std::vector<std::vector<double>>& probs,
                   const std::vector<std::vector<int>>& polarity,
                   const std::vector<std::vector<int>>& strength,
                   const std::vector<int>& seeds_A,
                   const std::vector<int>& seeds_B,
                   uint64_t seed) {
    DiffusionEngine engine(node_ids, adj, probs, polarity, strength);
    return engine.simulate(seeds_A, seeds_B, seed);
}

bool openmp_enabled() {
#ifdef _OPENMP
    return true;
#else
    return false;
#endif
}
