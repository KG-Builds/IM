import json, time, math, random
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import networkx as nx

K_VALUES=[10,20,30,40,50]
MASTER_SEED=42
SSA_POPULATION=12
SSA_ITERATIONS=10
SSA_MUTATION_RATE=0.20
OPT_MC_RUNS=15
FINAL_MC_RUNS=100
POOL_RATIO=10
RANDOM_POOL_FRACTION=0.30
SWAP_MAX_TESTS=100

VARIANTS={
    "Original SSA":{"type":"weighted","alpha":1.0,"beta":1.5},
    "Competition-aware":{"type":"net"},
    "Explicit blocking":{"type":"blocking"},
    "Combined influence-blocking":{"type":"weighted","alpha":1.0,"beta":2.0},
    "Gain_A + blocking":{"type":"gain_blocking","beta":2.0},
    "Budget-aware":{"type":"budget"},
}

def load_network(path):
    df=pd.read_csv(path)
    nodes=sorted(set(df.source.astype(int))|set(df.target.astype(int)))
    G=nx.DiGraph(); G.add_nodes_from(nodes)
    for r in df.itertuples(index=False):
        G.add_edge(int(r.source),int(r.target),probability=float(r.probability),
                   polarity=int(r.polarity),strength=int(r.strength))
    return df,G,nodes

def opponent_orders(G,nodes,scenario_id,seed=42,max_k=50):
    rng=random.Random(seed+810001)
    if scenario_id=="B1":
        order=rng.sample(nodes,max_k)
    elif scenario_id=="B2":
        order=sorted(nodes,key=lambda u:(-G.out_degree(u),int(u)))[:max_k]
    elif scenario_id=="B3":
        pos=nx.DiGraph(); pos.add_nodes_from(nodes)
        pos.add_edges_from((u,v,{"weight":d["probability"]}) for u,v,d in G.edges(data=True) if d["polarity"]>0)
        pr=nx.pagerank(pos,alpha=.85,weight="weight",max_iter=200,tol=1e-10) if pos.number_of_edges() else {u:1/len(nodes) for u in nodes}
        order=sorted(nodes,key=lambda u:(-pr.get(u,0),int(u)))[:max_k]
    elif scenario_id=="B4":
        und=nx.Graph(); und.add_nodes_from(nodes); und.add_edges_from((u,v) for u,v in G.edges() if u!=v)
        comms=list(nx.community.louvain_communities(und,seed=seed,weight=None)) if und.number_of_edges() else [{u} for u in nodes]
        largest=sorted(comms,key=lambda c:(-len(c),min(c)))[0]
        order=sorted(largest,key=lambda u:(-G.out_degree(u),int(u)))
        if len(order)<max_k: raise ValueError("Largest community smaller than max K.")
    else: raise ValueError("Unknown scenario")
    return order[:max_k]

def build_adjacency(G,nodes):
    index={node:i for i,node in enumerate(nodes)}
    adj=[]; probs=[]; pol=[]; strength=[]
    for u in nodes:
        nbrs=sorted(G.successors(u))
        adj.append([index[v] for v in nbrs])
        probs.append([G[u][v]["probability"] for v in nbrs])
        pol.append([G[u][v]["polarity"] for v in nbrs])
        strength.append([G[u][v]["strength"] for v in nbrs])
    return index,adj,probs,pol,strength

def metric_cache_eval(sim_func,A,B,seeds):
    outcomes=[sim_func(A,B,s) for s in seeds]
    return {
        "spread_A":float(np.mean([x["spread_A"] for x in outcomes])),
        "spread_B":float(np.mean([x["spread_B"] for x in outcomes])),
        "net_advantage":float(np.mean([x["spread_A"]-x["spread_B"] for x in outcomes])),
        "ignorant":float(np.mean([x["ignorant"] for x in outcomes])),
        "rounds":float(np.mean([x["rounds"] for x in outcomes]))
    }

def objective_score(m,variant,k):
    v=VARIANTS[variant]; t=v["type"]
    if t=="net": return m["net_advantage"]
    if t=="blocking": return m["blocked_B"]
    if t=="weighted": return v["alpha"]*m["spread_A"]+v["beta"]*m["blocked_B"]
    if t=="gain_blocking": return (m["spread_A"]-m["spread_B"])+v["beta"]*m["blocked_B"]
    if t=="budget":
        if k<=20: a,b=.75,2.25
        elif k<=40: a,b=1.,1.75
        else: a,b=1.25,1.5
        return a*m["spread_A"]+b*m["blocked_B"]
    raise ValueError(t)

def make_candidate_pool(G,nodes,B,k,seed):
    pool_size=k*POOL_RATIO
    # polarity-aware candidate ranking, precomputed by outgoing edges
    score={}
    for u in nodes:
        pos=sum(G[u][v]["probability"] for v in G.successors(u) if G[u][v]["polarity"]>0)
        neg=sum(G[u][v]["probability"] for v in G.successors(u) if G[u][v]["polarity"]<0)
        score[u]=pos-neg
    ranked=sorted([u for u in nodes if u not in set(B)],key=lambda u:(-score[u],u))
    n_rank=int(pool_size*(1-RANDOM_POOL_FRACTION))
    ranked_part=ranked[:n_rank]
    rng=random.Random(seed)
    remaining=[u for u in nodes if u not in set(B) and u not in set(ranked_part)]
    random_part=rng.sample(remaining,min(pool_size-len(ranked_part),len(remaining)))
    pool=list(dict.fromkeys(ranked_part+random_part))
    if len(pool)<k: raise ValueError("Candidate pool too small")
    return pool,score

def repair_solution(nodes, pool, k, rng):
    pool_set=set(pool); result=[]; seen=set()
    for u in nodes:
        if u in pool_set and u not in seen: result.append(u); seen.add(u)
    rem=[u for u in pool if u not in seen]
    if len(result)<k: result.extend(rng.sample(rem,k-len(result)))
    return result[:k]

def run_discrete_ssa(pool,B,k,sim_seeds,variant,score_seed_set,population_size=SSA_POPULATION,
                      iterations=SSA_ITERATIONS,mutation_rate=SSA_MUTATION_RATE,random_seed=42):
    rng=random.Random(random_seed)
    score=lambda sol: score_seed_set(sol,B,sim_seeds,variant,k)
    population=[rng.sample(pool,k) for _ in range(population_size)]
    scores=[score(s) for s in population]
    best_i=int(np.argmax(scores)); food=list(population[best_i]); food_score=float(scores[best_i])
    history=[{"iteration":0,"best_objective_score":food_score,"mean_objective_score":float(np.mean(scores))}]
    for iteration in range(1,iterations+1):
        c1=2.0*math.exp(-((4.0*iteration/iterations)**2))
        exploit=1.0-min(c1/2.0,1.0)
        new_pop=[]; leader=[]; current=population[0]
        for pos in range(k):
            if rng.random()<exploit:
                choices=[u for u in food if u not in leader]
                if choices: leader.append(rng.choice(choices)); continue
            choices=[u for u in current if u not in leader]
            if choices and rng.random()<.5: leader.append(rng.choice(choices))
            else: leader.append(rng.choice([u for u in pool if u not in leader]))
        new_pop.append(repair_solution(leader,pool,k,rng))
        for i in range(1,population_size):
            updated=list(population[i]); previous=new_pop[i-1]
            for pos in rng.sample(range(k),rng.randint(1,k)):
                used=set(updated[:pos]+updated[pos+1:])
                choices=[u for u in previous if u not in used]
                if choices: updated[pos]=rng.choice(choices)
            for pos in range(k):
                if rng.random()<mutation_rate:
                    used=set(updated[:pos]+updated[pos+1:])
                    choices=[u for u in pool if u not in used]
                    if choices: updated[pos]=rng.choice(choices)
            new_pop.append(repair_solution(updated,pool,k,rng))
        new_scores=[score(s) for s in new_pop]
        ib=int(np.argmax(new_scores))
        if new_scores[ib]>food_score: food=list(new_pop[ib]); food_score=float(new_scores[ib])
        iw=int(np.argmin(new_scores)); new_pop[iw]=list(food)
        population=new_pop
        history.append({"iteration":iteration,"best_objective_score":food_score,"mean_objective_score":float(np.mean(new_scores))})
    return food,food_score,pd.DataFrame(history)

def sampled_swap_refinement(initial,pool,B,k,seeds,variant,score_seed_set,max_tests=SWAP_MAX_TESTS,random_seed=1):
    """Best improving sampled one-for-one swap, matching the source notebook's behavior."""
    rng=random.Random(random_seed)
    selected=list(initial); B=set(B); pool=list(dict.fromkeys(pool))
    cache={}
    current=score_seed_set(selected,B,seeds,variant,k)
    outsiders=[u for u in pool if u not in set(selected) and u not in B]
    total=len(selected)*len(outsiders)
    if total<=max_tests:
        pairs=[(old,new) for old in selected for new in outsiders]
    else:
        pair_set=set()
        while len(pair_set)<max_tests:
            pair_set.add((rng.choice(selected),rng.choice(outsiders)))
        pairs=sorted(pair_set)
    best_score=current; best_pair=None
    for old,new in pairs:
        trial=[u for u in selected if u!=old]+[new]
        value=score_seed_set(trial,B,seeds,variant,k)
        if value>best_score+1e-12:
            best_score=value; best_pair=(old,new)
    if best_pair:
        old,new=best_pair; selected.remove(old); selected.append(new); current=best_score
    return sorted(selected),{"swap_tests":len(pairs),"accepted_swap":best_pair,
        "objective_before":float(score_seed_set(initial,B,seeds,variant,k)),
        "objective_after":float(current),"unique_evaluations":len(cache)}
