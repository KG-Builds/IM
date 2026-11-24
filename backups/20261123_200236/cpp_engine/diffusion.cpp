#include "diffusion.hpp"
#include <random>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <stdexcept>

SimResult simulate(
    const std::vector<int>& node_ids,
    const std::vector<std::vector<int>>& adj,
    const std::vector<std::vector<double>>& probs,
    const std::vector<std::vector<int>>& polarity,
    const std::vector<std::vector<int>>& strength,
    const std::vector<int>& seeds_A,
    const std::vector<int>& seeds_B,
    uint64_t seed
) {
    const int n = (int)node_ids.size();
    std::unordered_map<int,int> idx;
    idx.reserve(n*2);
    for(int i=0;i<n;i++) idx[node_ids[i]]=i;
    std::vector<int> state(n,0); // 0 ignorant, 1 A spreader, 2 B spreader, 3 A stifler, 4 B stifler
    for(int id: seeds_A) {
        if(!idx.count(id)) throw std::invalid_argument("Unknown A seed");
        state[idx[id]]=1;
    }
    for(int id: seeds_B) {
        if(!idx.count(id)) throw std::invalid_argument("Unknown B seed");
        if(state[idx[id]]!=0) throw std::invalid_argument("A/B seed overlap");
        state[idx[id]]=2;
    }
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<double> real(0.0,1.0);
    std::vector<std::vector<unsigned char>> contacted(n);
    for(int u=0;u<n;u++) contacted[u].assign(adj[u].size(),0);
    const int max_rounds = 1;
    int round_no=0;
    // Each spreader contacts one random, not-yet-contacted outgoing neighbor per round.
    while(true) {
        std::vector<int> old=state, next=state;
        bool any_spreader=false;
        std::vector<int> stiflers;
        struct Proposal { int rumor; int strength; };
        std::vector<std::vector<Proposal>> proposals(n);
        for(int u=0;u<n;u++) if(old[u]==1 || old[u]==2) {
            any_spreader=true;
            std::vector<int> available;
            for(int j=0;j<(int)adj[u].size();j++) if(!contacted[u][j]) available.push_back(j);
            if(available.empty()) { stiflers.push_back(u); continue; }
            std::uniform_int_distribution<int> pick(0,(int)available.size()-1);
            int j=available[pick(gen)];
            contacted[u][j]=1;
            int v=adj[u][j];
            if(old[v]!=0) { stiflers.push_back(u); continue; }
            if(real(gen)<probs[u][j]) {
                int rumor=(old[u]==1)?1:2;
                if(polarity[u][j]<0) rumor=(rumor==1)?2:1;
                proposals[v].push_back({rumor,strength[u][j]});
            }
        }
        if(!any_spreader) break;
        for(int v=0;v<n;v++) if(!proposals[v].empty()) {
            int max_s=0; for(auto p:proposals[v]) max_s=std::max(max_s,p.strength);
            std::vector<int> contenders;
            for(auto p:proposals[v]) if(p.strength==max_s) contenders.push_back(p.rumor);
            std::sort(contenders.begin(),contenders.end());
            contenders.erase(std::unique(contenders.begin(),contenders.end()),contenders.end());
            std::uniform_int_distribution<int> pick(0,(int)contenders.size()-1);
            next[v]=contenders[pick(gen)];
        }
        for(int u:stiflers) next[u]=(old[u]==1)?3:4;
        state.swap(next);
        round_no++;
        if(round_no > 7501) throw std::runtime_error("Simulation exceeded finite-contact safety limit");
    }
    int A=0,B=0,I=0;
    for(int s:state) { if(s==1||s==3) A++; else if(s==2||s==4) B++; else I++; }
    return {A,B,I,round_no};
}
