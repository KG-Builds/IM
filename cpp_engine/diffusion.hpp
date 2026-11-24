#pragma once
#include <cstdint>
#include <unordered_map>
#include <vector>

struct SimResult {
    int spread_A;
    int spread_B;
    int ignorant;
    int rounds;
};

// Persistent, read-only graph engine for repeated Monte Carlo simulations.
// The graph is built once and reused; every run has its own RNG stream, so
// results are independent of thread scheduling.
class DiffusionEngine {
public:
    DiffusionEngine(const std::vector<int>& node_ids,
                    const std::vector<std::vector<int>>& adj,
                    const std::vector<std::vector<double>>& probs,
                    const std::vector<std::vector<int>>& polarity,
                    const std::vector<std::vector<int>>& strength);

    SimResult simulate(const std::vector<int>& seeds_A,
                       const std::vector<int>& seeds_B,
                       uint64_t seed) const;

    std::vector<SimResult> simulate_many(const std::vector<int>& seeds_A,
                                         const std::vector<int>& seeds_B,
                                         const std::vector<uint64_t>& seeds,
                                         int n_threads = 0) const;

    std::vector<SimResult> simulate_batch(const std::vector<std::vector<int>>& seeds_A,
                                          const std::vector<std::vector<int>>& seeds_B,
                                          const std::vector<uint64_t>& seeds,
                                          int n_threads = 0) const;

    int num_nodes() const { return n_; }
    int num_edges() const { return m_; }
    const std::vector<int>& node_ids() const { return node_ids_; }

private:
    struct Workspace {
        std::vector<int> state;
        std::vector<int> old;
        std::vector<int> next;
        std::vector<unsigned char> contacted;
        std::vector<int> best_strength;
        std::vector<unsigned char> rumor_mask;
        std::vector<int> stiflers;

        Workspace(int n, int m)
            : state(n, 0), old(n, 0), next(n, 0), contacted(m, 0),
              best_strength(n, -1), rumor_mask(n, 0) {
            stiflers.reserve(static_cast<size_t>(n));
        }
    };

    void validate_seeds(const std::vector<int>& seeds_A,
                        const std::vector<int>& seeds_B) const;
    SimResult run(const std::vector<int>& seeds_A,
                  const std::vector<int>& seeds_B,
                  uint64_t seed,
                  Workspace& ws) const;

    int n_ = 0;
    int m_ = 0;
    std::vector<int> node_ids_;
    std::unordered_map<int, int> idx_;
    std::vector<int> offsets_;           // size n+1
    std::vector<int> targets_;           // size m
    std::vector<double> probs_;          // size m
    std::vector<signed char> polarity_;  // size m, -1 or +1
    std::vector<int> strength_;          // size m, 1..10
};

// Backward-compatible functional API with the starter's exact semantics.
SimResult simulate(const std::vector<int>& node_ids,
                   const std::vector<std::vector<int>>& adj,
                   const std::vector<std::vector<double>>& probs,
                   const std::vector<std::vector<int>>& polarity,
                   const std::vector<std::vector<int>>& strength,
                   const std::vector<int>& seeds_A,
                   const std::vector<int>& seeds_B,
                   uint64_t seed);

bool openmp_enabled();
