#pragma once
#include <vector>
#include <cstdint>

struct SimResult {
    int spread_A;
    int spread_B;
    int ignorant;
    int rounds;
};

SimResult simulate(
    const std::vector<int>& node_ids,
    const std::vector<std::vector<int>>& adj,
    const std::vector<std::vector<double>>& probs,
    const std::vector<std::vector<int>>& polarity,
    const std::vector<std::vector<int>>& strength,
    const std::vector<int>& seeds_A,
    const std::vector<int>& seeds_B,
    uint64_t seed
);
