import json, time, random, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from experiment_core import *

def run_scenario(scenario_id, results_dir, network_path, engine_simulator, run_swap=True):
    results_dir=Path(results_dir); results_dir.mkdir(parents=True,exist_ok=True)
    out_csv=results_dir/"experiment_results.csv"
    checkpoints=results_dir/"checkpoints"; checkpoints.mkdir(exist_ok=True)
    df,G,nodes=load_network(network_path)
    B_order=opponent_orders(G,nodes,scenario_id,MASTER_SEED)
    sim_cache={}
    baseline_cache={}
    score_cache={}
    cache_hits=0; cache_misses=0
    def cached(A,B,simseed):
        nonlocal cache_hits,cache_misses
        key=(tuple(sorted(set(map(int,A)))),tuple(sorted(set(map(int,B)))),int(simseed))
        if key in sim_cache: cache_hits+=1; return sim_cache[key]
        cache_misses+=1
        result=engine_simulator(A,B,simseed); sim_cache[key]=result; return result
    def evaluate(A,B,seeds):
        A=set(A); B=set(B); seeds=tuple(map(int,seeds))
        bkey=(tuple(sorted(B)),seeds)
        if bkey not in baseline_cache:
            baseline_cache[bkey]=float(np.mean([cached([],B,s)["spread_B"] for s in seeds]))
        outcomes=[cached(A,B,s) for s in seeds]
        a=float(np.mean([x["spread_A"] for x in outcomes]))
        b=float(np.mean([x["spread_B"] for x in outcomes]))
        net=float(np.mean([x["spread_A"]-x["spread_B"] for x in outcomes]))
        blocked=max(0.0,baseline_cache[bkey]-b)
        return {"spread_A":a,"spread_B":b,"net_advantage":net,"blocked_B":blocked,
                "b_only_baseline_spread":baseline_cache[bkey],
                "ignorant":float(np.mean([x["ignorant"] for x in outcomes]))}
    def score_set(A,B,seeds,variant,k):
        key=(tuple(sorted(set(A))),tuple(sorted(set(B))),tuple(map(int,seeds)))
        if key not in score_cache: score_cache[key]=evaluate(A,B,seeds)
        return objective_score(score_cache[key],variant,k)
    records=pd.read_csv(out_csv).to_dict("records") if out_csv.exists() else []
    keys={(int(r["K"]),r["variant"],r["mode"]) for r in records}
    for k in K_VALUES:
        B=set(B_order[:k])
        pool,pre_scores=make_candidate_pool(G,nodes,B,k,MASTER_SEED+600000+int(scenario_id[1:]))
        opt_seeds=[MASTER_SEED*100000+3000000+int(scenario_id[1:])*10000+i for i in range(OPT_MC_RUNS)]
        final_seeds=[900000000+i for i in range(FINAL_MC_RUNS)]
        for variant_idx,variant in enumerate(VARIANTS):
            checkpoint=checkpoints/f"{scenario_id}_{k}_{variant_idx}.json"
            config_signature={"scenario":scenario_id,"k":k,"variant":variant,"network":network_path.name,
                             "ssa_pop":SSA_POPULATION,"ssa_iter":SSA_ITERATIONS,"opt_mc":OPT_MC_RUNS,
                             "final_mc":FINAL_MC_RUNS,"seed":MASTER_SEED}
            if checkpoint.exists():
                saved=json.loads(checkpoint.read_text())
                if saved.get("config")!=config_signature:
                    raise ValueError(f"Checkpoint signature mismatch: {checkpoint}")
                ssa_set=saved["ssa_seed_set"]; ssa_score=saved["score"]; ssa_sec=saved["ssa_runtime"]
            else:
                rng_seed=MASTER_SEED+500000+int(scenario_id[1:])*1000+k*10+variant_idx
                scorer=lambda A,B,simseeds,v,kk: score_set(A,B,simseeds,v,kk)
                t=time.perf_counter()
                ssa_set,ssa_score,hist=run_discrete_ssa(pool,B,k,opt_seeds,variant,scorer,
                    random_seed=rng_seed)
                ssa_sec=time.perf_counter()-t
                saved={"config":config_signature,"ssa_seed_set":list(map(int,ssa_set)),
                       "score":float(ssa_score),"ssa_runtime":float(ssa_sec),"history":hist.to_dict("records")}
                checkpoint.write_text(json.dumps(saved,default=str))
            modes=["SSA only"]+(["SSA + Swap refinement"] if run_swap else [])
            for mode in modes:
                key=(k,variant,mode)
                if key in keys: continue
                seed_set=list(ssa_set); swap_info=None; swap_sec=0
                if mode=="SSA + Swap refinement":
                    scorer=lambda A,B,simseeds,v,kk: score_set(A,B,simseeds,v,kk)
                    ts=time.perf_counter()
                    seed_set,swap_info=sampled_swap_refinement(ssa_set,pool,B,k,opt_seeds,variant,scorer,
                        max_tests=SWAP_MAX_TESTS,random_seed=770000+int(scenario_id[1:])*1000+k+variant_idx)
                    swap_sec=time.perf_counter()-ts
                ts=time.perf_counter(); metrics=evaluate(seed_set,B,final_seeds); final_sec=time.perf_counter()-ts
                row={"scenario":scenario_id,"K":k,"variant":variant,"mode":mode,
                     "seed_set":json.dumps(sorted(map(int,seed_set))),"opponent_seed_set":json.dumps(sorted(map(int,B))),
                     "objective_score":float(ssa_score),"spread_A":metrics["spread_A"],"spread_B":metrics["spread_B"],
                     "net_advantage":metrics["net_advantage"],"blocked_B":metrics["blocked_B"],
                     "b_only_baseline_spread":metrics["b_only_baseline_spread"],"ignorant":metrics["ignorant"],
                     "ssa_runtime_sec":ssa_sec,"swap_runtime_sec":swap_sec,"final_evaluation_runtime_sec":final_sec,
                     "optimization_mc_runs":OPT_MC_RUNS,"final_evaluation_mc_runs":FINAL_MC_RUNS,
                     "swap_tests":(swap_info or {}).get("swap_tests",0),
                     "swap_accepted":bool((swap_info or {}).get("accepted_swap") is not None),
                     "swap_objective_before":(swap_info or {}).get("objective_before",np.nan),
                     "swap_objective_after":(swap_info or {}).get("objective_after",np.nan)}
                records.append(row); keys.add(key)
                pd.DataFrame(records).drop_duplicates(["K","variant","mode"],keep="last").to_csv(out_csv,index=False)
                print(f"Saved: {scenario_id} K={k} {variant} {mode} | Net Advantage={row['net_advantage']:.3f}")
    print(f"Finished {scenario_id}; CSV: {out_csv}; cache hits={cache_hits} misses={cache_misses}")
    return pd.read_csv(out_csv)
