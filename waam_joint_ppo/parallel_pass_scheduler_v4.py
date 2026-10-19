"""PASS-baseline-first collision-checked same-layer parallel scheduling.

Reuses immutable centerline deposition and travel segments from confirmed PASS input.
Layer boundaries stay globally sequential. Within each layer, robot blocks may
start concurrently when the Validator's own ARM/TCP simulation accepts the overlap.
The global candidate is only adopted after the official Validator CLI reports PASS.
No PPO is trained in this version. No mathematical optimality is claimed.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, shutil, sys, time
from pathlib import Path
from waam_joint_ppo.pass_local_bc_ppo_v3 import (
    locate_source,read_tracks,write_tracks,layer_plan,rebuild_layers,
    real_finish,deposition_signature,travel_signature,layer_order_ok,
    fast_collision_check,validate_official,
)


def json_save(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def make_layer_tracks(z, plan, blocks, parked, starts):
    """An EXACT time-shifted one-layer trajectory; inactive robots remain parked.

    The only modifications are the start timestamps and the waiting intervals.
    All deposited and traveled XY/Z line segments and their durations are fixed.
    This local scenario suffices to precheck intra-layer collision because layer
    operations are globally non-overlapping, including the shared boundary.
    Every successful full schedule is rechecked by the official validator.
    """
    layer=plan[z]
    end=max((float(starts.get(r,0.))+item['duration'] for r,item in layer.items()),default=.01)
    result={}
    for r in (1,2,3):
        home=list(parked[r])
        if r not in layer:
            result[r]=[[0.,*home,'W'],[end+.001,*home,'W']]
            continue
        seg=blocks[(z,r)]
        pos=list(seg[0][1:4]);
        if any(abs(a-b)>1e-4 for a,b in zip(home,pos)):
            raise ValueError(f'R{r}: layer {z} start pose mismatch, cannot time-shift safely')
        offset=float(starts.get(r,layer[r]['offset']))
        if offset<0:raise ValueError('Negative start time')
        if offset>1e-7:
            rr=[[0.,*pos,'W'],[offset,*pos,seg[0][4]]]
        else:
            rr=[[0.,*pos,seg[0][4]]]
        t0=seg[0][0]
        for original in seg[1:]:
            newtime=offset+(original[0]-t0)
            if newtime<=rr[-1][0]+1e-8:raise ValueError('Non-increasing timestamp in layer')
            rr.append([newtime,*original[1:]])
        rr[-1][4]='W'
        rr.append([end+.001,*rr[-1][1:4],'W'])
        result[r]=rr
    return result


def parked_positions(reference,plan,blocks):
    parked={r:list(reference[r][0][1:4]) for r in (1,2,3)}
    result={}
    for z,items in sorted(plan.items()):
        result[z]={r:list(p) for r,p in parked.items()}
        for r in items:
            start=list(blocks[(z,r)][0][1:4]);
            if any(abs(a-b)>1e-4 for a,b in zip(parked[r],start)):
                raise ValueError(f'R{r} start pose discontinuity at z={z}')
            parked[r]=list(blocks[(z,r)][-1][1:4])
    return result


def layer_duration(plan,z,offsets):
    return max((offsets[r]+i['duration'] for r,i in plan[z].items()),default=0.)


def scan_start_times(baseline, *, granularity=12):
    """Coarse-to-fine proposals, including near-zero and near-baseline offsets."""
    if baseline<.001:return [0.]
    fractions=(0.,.1,.25,.40,.55,.7,.8,.88,.93,.96,.98,.99,1.)
    return sorted(set(round(baseline*x,7) for x in fractions))


def find_layer_schedule(z,plan,blocks,parked,config,*,passes=2,max_probes=50,verbose=True):
    """Coordinate descent on robot start offsets, plus paired starts for escape.

    Each probe is checked by the genuine Validator collision simulator using
    a complete *local* three-robot trace; no guessed clearance is accepted.
    Global final official PASS is always required.
    """
    layer=plan[z]
    offsets={r:float(info['offset']) for r,info in layer.items()}
    original=layer_duration(plan,z,offsets)
    if len(layer)<2:return offsets,0,original,0
    count=0;accepted=0;cache={}
    def check(starts):
        nonlocal count
        key=tuple(round(starts[r],6) for r in sorted(starts))
        if key in cache:return cache[key]
        if count>=max_probes:return None
        count+=1
        local=make_layer_tracks(z,plan,blocks,parked,starts)
        value=fast_collision_check(local,config)
        cache[key]=value
        return value
    # Known safe from source, but local precheck still explicitly verifies.
    if check(offsets) is not True:raise RuntimeError(f'Layer z={z}: original PASS layer unexpectedly fails local collision check')
    keys=sorted(layer)
    # Paired advances FIRST. Some collision-safe starts are only reachable when
    # two robots shift together: a one-robot greedy step would reject them.
    if len(keys)==3 and count<max_probes:
        r2,r3=keys[-2:]
        base2=layer[r2]['offset'];base3=layer[r3]['offset']
        guesses=[(18,28),(15,22),(25,35),(10,15),(35,50),(20,50),
                 (8,18),(28,18),(40,45),(40,65),(5,8),(0,22),(22,0),
                 (.1*base2,.1*base3),(.2*base2,.2*base3),
                 (.35*base2,.35*base3),(.5*base2,.5*base3)]
        best=offsets
        for delta2,delta3 in guesses:
            trial={**offsets,r2:max(0.,base2-delta2),r3:max(0.,base3-delta3)}
            if layer_duration(plan,z,trial)>=layer_duration(plan,z,best)-1e-4:continue
            safe=check(trial)
            if safe is None:break
            if safe:best=trial;accepted+=1
        offsets=best
    # Greedy move robot starts, preferring the robot on the critical path.
    for _ in range(max(1,passes)):
        changed=False
        for r in sorted(keys,key=lambda r: offsets[r]+layer[r]['duration'],reverse=True):
            current=layer_duration(plan,z,offsets)
            values=scan_start_times(float(layer[r]['offset']))
            if offsets[r]>0:
                values.extend(max(0.,offsets[r]-s) for s in [1.,3.,8.,15.,30.,60.])
            values=sorted(set(round(v,7) for v in values if v<=layer[r]['offset']+1e-7))
            # Limit proposals to ones that reduce the layer makespan.
            values=[v for v in values if v<offsets[r]-1e-5 and
                    layer_duration(plan,z,{**offsets,r:v}) < current-1e-4]
            values.sort()
            best=None
            for val in values:
                test={**offsets,r:val}
                safe=check(test)
                if safe is None:break
                if safe:
                    best=test;break  # smaller offset is faster for this fixed context
            if best is not None:
                offsets=best;accepted+=1;changed=True
            if count>=max_probes:break
        if count>=max_probes or not changed:break
    # Joint starts: avoid being stuck when moving one robot alone always collides.
    if count<max_probes and len(keys)>=2:
        candidate_options={r:scan_start_times(layer[r]['offset']) for r in keys}
        from itertools import product
        combos=[]
        for vals in product(*(candidate_options[r] for r in keys)):
            trial=dict(zip(keys,vals))
            duration=layer_duration(plan,z,trial)
            if duration < layer_duration(plan,z,offsets)-1e-4:
                combos.append((duration,tuple(vals),trial))
        # Sampling bound limits quadratic/cubic explosion for variable robot counts.
        for dur,_,trial in sorted(combos):
            ok=check(trial)
            if ok is None:break
            if ok:
                offsets=trial;accepted+=1;break
    reduced=original-layer_duration(plan,z,offsets)
    if verbose:print(f'  layer z={z:g}: {original:.3f}s -> {original-reduced:.3f}s, '+
                     f'saved={reduced:.3f}s, safety_probes={count}, accepted={accepted}',flush=True)
    return offsets,reduced,original,count


def simultaneous_deposition_seconds(tracks):
    events=[]
    for rows in tracks.values():
        for a,b in zip(rows,rows[1:]):
            if a[4]=='D':
                events.append((a[0],1))
                events.append((b[0],-1))
    events.sort()
    total=0.;active=0;last=None
    for t,delta in events:
        if last is not None and active>=2:
            total+=t-last
        active+=delta;last=t
    return total


def optimize(root,model,*,trial='parallel_v4_pilot',max_probes=50,passes=2,timeout=3600,
             max_layers=0,keep_existing=True):
    root=Path(root).resolve();model=str(model).zfill(2)
    source,validated_base=locate_source(root,model)
    # Source confirmed official PASS and source tracking match are checked above.
    reference=read_tracks(source/'trajectory.csv')
    blocks,plan=layer_plan(reference)
    if not layer_order_ok(reference):raise RuntimeError('Source violates global layer ordering')
    parks=parked_positions(reference,plan,blocks)
    reportdir=root/'ppo_project'/'waam_parallel_pass_v4'/trial/model
    reportdir.mkdir(parents=True,exist_ok=True)
    baseline_sig=(deposition_signature(reference),travel_signature(reference))
    best_adv={};layers=[];optimized_seconds=0.;all_probes=0
    t_start=time.monotonic()
    zs=list(sorted(plan))
    if max_layers:
        print(f'NOTE: only first {max_layers} layers SEARCHED, other layers remain unchanged. Full candidate still generated and validated.',flush=True)
    for k,z in enumerate(zs):
        if max_layers and k>=max_layers:break
        offsets,improvement,before,nprobes=find_layer_schedule(z,plan,blocks,parks[z],source/'config.yaml',
                passes=passes,max_probes=max_probes)
        best_adv.update({(z,r):max(0.,plan[z][r]['offset']-offsets[r]) for r in plan[z]})
        optimized_seconds+=improvement;all_probes+=nprobes
        layers.append({'z_mm':z,'baseline_layer_s':before,
            'optimized_layer_s':before-improvement,'saved_s':improvement,'offsets_s':offsets,
            'safety_probes':nprobes})
        print(f'[{model}] layer {k+1}/{len(zs)} cumulative_saved={optimized_seconds:.3f}s',flush=True)
    candidate=rebuild_layers(reference,best_adv,blocks,plan)
    if deposition_signature(candidate)!=baseline_sig[0] or travel_signature(candidate)!=baseline_sig[1]:
        raise RuntimeError('Geometry/speed signatures changed: candidate rejected')
    if not layer_order_ok(candidate):raise RuntimeError('Layer deposition order changed: candidate rejected')
    trial_makespan=real_finish(candidate)
    if trial_makespan>=validated_base-1e-3:
        result={'model':model,'status':'KEEP_BASELINE_PASS','baseline_makespan_s':validated_base,
                'candidate_precheck_makespan_s':trial_makespan,'official_candidate_status':'NOT_RUN_NO_GAIN',
                'baseline_simultaneous_deposition_s':simultaneous_deposition_seconds(reference),
                'layers':layers,'probes':all_probes,'wall_s':round(time.monotonic()-t_start,2)}
        json_save(reportdir/'report.json',result);print(json.dumps({k:v for k,v in result.items() if k!='layers'},indent=2));return result
    # Whole-model collision check catches any local-window reconstruction mistake.
    if not fast_collision_check(candidate,source/'config.yaml'):
        result={'model':model,'status':'KEEP_BASELINE_PASS','baseline_makespan_s':validated_base,
            'candidate_precheck_makespan_s':trial_makespan,'official_candidate_status':'NOT_RUN_FULL_COLLISION_FAIL',
            'candidate_simultaneous_deposition_s':simultaneous_deposition_seconds(candidate),
            'layers':layers,'probes':all_probes}
        json_save(reportdir/'report.json',result);print(json.dumps({k:v for k,v in result.items() if k!='layers'},indent=2));return result
    output=reportdir/'parallel_candidate'
    official=validate_official(source,candidate,output,timeout)
    deposit_parallel=simultaneous_deposition_seconds(candidate)
    if deposit_parallel<.001:
        print(f'NOTICE [{model}]: Makespan reduced by safe scheduling, but concurrent DEPOSITION remains 0.0s; bead-level reordering is still needed.',flush=True)
    else:
        print(f'NOTICE [{model}]: Detected {deposit_parallel:.3f}s of simultaneous DEPOSITION; subject to official Validator PASS.',flush=True)
    if official is not None and official<validated_base-1e-3:
        verdict='NEW_PASS_BETTER'
    else:
        verdict='KEEP_BASELINE_PASS'
    result={'model':model,'baseline_makespan_s':validated_base,
        'candidate_precheck_makespan_s':trial_makespan,
        'validated_makespan_s':official,
        'baseline_simultaneous_deposition_s':simultaneous_deposition_seconds(reference),
        'candidate_simultaneous_deposition_s':deposit_parallel,
        'actual_parallel_deposition_observed':bool(deposit_parallel>=.001),
        'optimization_scope':'per-layer fixed-geometry start-time scheduling',
        'official_candidate_status':'PASS' if official is not None else 'NOT_PASS',
        'status':verdict,'parallel_candidate':str(output),'safe_fallback':str(source),
        'layers':layers,'probes':all_probes,'total_estimated_saved_s':optimized_seconds,
        'wall_s':round(time.monotonic()-t_start,2)}
    json_save(reportdir/'report.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='layers'},ensure_ascii=False,indent=2),flush=True)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path('.'))
    ap.add_argument('--models',nargs='+',default=['02','03'])
    ap.add_argument('--trial',default='parallel_v4_pilot')
    ap.add_argument('--max-probes-per-layer',type=int,default=50)
    ap.add_argument('--passes',type=int,default=2)
    ap.add_argument('--max-layers',type=int,default=0,help='Search first N layers but still validate whole model')
    ap.add_argument('--timeout',type=int,default=3600)
    ap.add_argument('--allow-model-01',action='store_true',help='Explicitly allow model 01; default protects serial-only assumption')
    args=ap.parse_args()
    if args.max_probes_per_layer<1 or args.passes<1:ap.error('Positive --max-probes-per-layer and --passes required')
    for model in args.models:
        m=str(model).zfill(2)
        if m=='01' and not args.allow_model_01:
            ap.error('Model 01 assumed serial-only; use --allow-model-01 only if parallelism is permitted')
        optimize(args.root,m,trial=args.trial,max_probes=args.max_probes_per_layer,
                 passes=args.passes,max_layers=args.max_layers,timeout=args.timeout)

if __name__=='__main__':main()
