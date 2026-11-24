from pathlib import Path
import sys
import simulation_engine
from experiment_core import build_adjacency

ENGINE_DIR=Path(__file__).resolve().parents[1]/"cpp_engine"
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0,str(ENGINE_DIR))

def make_cpp_simulator(nodes, G):
    node_ids=list(map(int,nodes))
    index,adj,probs,polarity,strength=build_adjacency(G,node_ids)
    def run(A,B,seed):
        result=simulation_engine.simulate(node_ids,adj,probs,polarity,strength,
            list(map(int,A)),list(map(int,B)),int(seed))
        return {"spread_A":int(result.spread_A),"spread_B":int(result.spread_B),
                "ignorant":int(result.ignorant),"rounds":int(result.rounds)}
    return run
