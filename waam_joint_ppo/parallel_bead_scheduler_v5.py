"""v5: validated PASS baseline -> collision-checked inter-bead pause / start optimization.

No change to D or T xyz, durations or robot assignment. No interruption inside
an active D interval. Search increases opportunities for same-layer parallel D.
Prechecks use the Validator's own collision engine; only official CLI PASS
may promote an output above the immutable source PASS. No global optimality claim.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from waam_joint_ppo.pass_local_bc_ppo_v3 import (
    read_tracks,layer_plan,layer_order_ok,deposition_signature,travel_signature,
    fast_collision_check,validate_official,real_finish,locate_source,write_json,
)
from waam_joint_ppo.parallel_pass_scheduler_v4 import parked_positions,simultaneous_deposition_seconds


def source_for(root, model, parallel_trial):
    root=Path(root)
    candidate=root/'ppo_project'/'waam_parallel_pass_v4'/parallel_trial/model/'parallel_candidate'
    record=candidate/'validation_status.json'
    if record.is_file():
        report=json.loads(record.read_text(encoding='utf-8'))
        if report.get('status')=='PASS':
            required=('config.yaml','target.stl','trajectory.csv')
            if all((candidate/f).is_file() for f in required):
                return candidate,float(report['schedule']['makespan_s']),'v4_official_pass'
    base,score=locate_source(root,model)
    return base,score,'centerline_official_pass'


def bead_pause_sites(segment):
    """Indices marking a T<->D transition; NEVER insert mid-deposition."""
    points=[]
    for i in range(1,len(segment)-1):
        old=segment[i-1][4];now=segment[i][4]
        if old!=now and {'D','T'}=={old,now}:
            points.append(i)
    # Pauses at bead ends (D -> T) and travel ends (T -> D).
    return points


def layer_tracks(z,plan,blocks,parked,starts,pauses):
    """Build exact local 3-robot timeline, with W gaps at bead boundaries."""
    layer=plan[z]
    durations={r:float(starts[r])+layer[r]['duration']+
               sum(float(v) for v in pauses.get(r,{}).values()) for r in layer}
    end=max(durations.values(),default=0.)
    result={}
    for r in (1,2,3):
        home=list(parked[r])
        if r not in layer:
            result[r]=[[0.,*home,'W'],[end+.001,*home,'W']]
            continue
        seg=blocks[(z,r)]
        if max(abs(a-b) for a,b in zip(home,seg[0][1:4]))>1e-4:
            raise ValueError(f'R{r}: initial pose mismatch at layer {z}')
        offset=float(starts[r]);t0=seg[0][0]
        if offset < -1e-8:raise ValueError('Negative start offset')
        rr=[[0.,*home,'W']]
        if offset>1e-8:rr.append([offset,*home,seg[0][4]])
        else:rr[0][4]=seg[0][4]
        total_wait=0.;sites=pauses.get(r,{})
        available=set(bead_pause_sites(seg))
        if any(i not in available or v<=0 for i,v in sites.items()):
            raise ValueError('Invalid or negative bead pause')
        for i,row in enumerate(seg[1:],start=1):
            ti=offset+row[0]-t0+total_wait
            if ti<=rr[-1][0]+1e-9:
                raise ValueError(f'Non-increasing time at z={z}, R{r}, {i}')
            rr.append([ti,*row[1:]])
            if i in sites:
                # The arrived point remains unchanged. Instead of starting
                # its T or D interval, wait at that coordinate first.
                rr[-1][4]='W'
                pause=float(sites[i]);total_wait+=pause
                rr.append([ti+pause,*row[1:]])
        if rr[-1][0]>end+1e-5:
            raise ValueError('Local finish greater than layer duration')
        rr[-1][4]='W'
        if end+.001<=rr[-1][0]:raise ValueError('End anchor not increasing')
        rr.append([end+.001,*rr[-1][1:4],'W'])
        result[r]=rr
    return result


def merge_layers(reference,blocks,plan,parks,schedules):
    out={r:[[0.,*reference[r][0][1:4],'W']] for r in (1,2,3)}
    clock=0.
    for z in sorted(plan):
        items=plan[z]
        starts,pauses=schedules.get(z,({r:float(items[r]['offset']) for r in items},{}))
        local=layer_tracks(z,plan,blocks,parks[z],starts,pauses)
        duration=local[1][-1][0]
        for r in (1,2,3):
            dest=out[r]
            if max(abs(a-b) for a,b in zip(dest[-1][1:4],local[r][0][1:4]))>1e-4:
                raise ValueError('Pose discontinuity across layers')
            if clock>dest[-1][0]+1e-8:
                dest[-1][4]='W'
                dest.append([clock,*dest[-1][1:4],'W'])
            dest[-1][4]=local[r][0][4]
            for row in local[r][1:]:
                dest.append([clock+row[0],*row[1:]])
        clock+=duration
    # Every robot ends with a W anchor; do not alter any T or D interval.
    return out


def layer_end(z,plan,starts,pauses):
    return max(starts[r]+plan[z][r]['duration']+sum(pauses.get(r,{}).values()) for r in plan[z])


def probe_layer(z,plan,blocks,parked,config,starts,pauses):
    return fast_collision_check(layer_tracks(z,plan,blocks,parked,starts,pauses),config)


def optimize_layer(z,plan,blocks,parked,config,budget=100,passes=2,seed=42):
    """Bounded stochastic local search, prechecking every accepted candidate."""
    rng=random.Random(seed+round(z*1000))
    keys=sorted(plan[z])
    initial={r:float(plan[z][r]['offset']) for r in keys}
    starts=dict(initial);pauses={}
    duration=layer_end(z,plan,starts,pauses)
    seen=set();probes=0;accepted=0
    sites={r:bead_pause_sites(blocks[(z,r)]) for r in keys}

    def safe(s,p):
        nonlocal probes
        key=(tuple((r,round(s[r],4)) for r in keys),tuple((r,i,round(v,4)) for r in sorted(p) for i,v in sorted(p[r].items())))
        if key in seen:return None
        if probes>=budget:return None
        seen.add(key);probes+=1
        return probe_layer(z,plan,blocks,parked,config,s,p)

    if len(keys)<2 or budget<2:return starts,pauses,duration,probes,accepted
    if safe(starts,pauses) is not True:raise RuntimeError(f'Existing PASS layer z={z} fails isolated collision precheck')
    # First search robot-start PERMUTATIONS. v4 only advanced starts without
    # delaying a different robot; this missed safe pair (R1,R3) in model 02.
    # A safe pair starts together; the third robot joins after the pair has
    # finished, or after a sampled earlier/later offset.
    from itertools import combinations
    permutations=[]
    for a,b in combinations(keys,2):
        c=next(x for x in keys if x not in (a,b))
        longest=max(plan[z][a]['duration'],plan[z][b]['duration'])
        delays=[0.,3.,6.,10.,20.,35.]
        third=[0.,25.,50.,75.,90.,100.,110.,115.,120.,125.,130.,135.,140.,145.,150.,160.,175.,190.,
               longest-4,longest,longest+2,longest+5,longest+10,longest+20]
        for paired_lag in delays:
            for value in third:
                trial={a:0.,b:paired_lag,c:max(0.,round(value,6))}
                t=layer_end(z,plan,trial,{})
                if t<duration-.001:
                    priority=(0 if {a,b}=={1,3} else 1,abs(value-(longest+5.)),paired_lag,t)
                    permutations.append((priority,t,trial))
    # First test plausible safe sequencing points near the end of a
    # concurrent pair, then move the third robot closer to create overlap.
    permutations.sort(key=lambda item:item[0])
    for _,_,ss in permutations:
        if probes>=max(2,budget//2):break
        if layer_end(z,plan,ss,{})>=duration-.001:continue
        ok=safe(ss,{})
        if ok is True:
            starts,pauses=ss,{}
            duration=layer_end(z,plan,ss,{})
            accepted+=1
    # Then probe inter-bead waits and timing changes around best safe order.
    start_shifts=(3.,6.,10.,15.,20.,28.,40.,55.,75.,100.,150.)
    wait_choices=(2.,5.,8.,12.,18.,25.,35.,50.,70.)
    ordered=sorted(keys,reverse=True)
    tries=[]
    for r in ordered:
        for advance in start_shifts:
            val=max(0.,starts[r]-advance)
            if val>=starts[r]-1e-5:continue
            tries.append((r,val,0,0.))
            checkpoints=sites[r]
            if not checkpoints:continue
            sampling=sorted(set(checkpoints[k] for k in (0,len(checkpoints)//4,len(checkpoints)//2,
                  (3*len(checkpoints))//4,len(checkpoints)-1)))
            for idx in sampling:
                for delay in wait_choices:
                    if delay<advance:
                        tries.append((r,val,idx,delay))
    rng.shuffle(tries)
    for _ in range(max(1,passes)):
        for r,val,idx,wait in tries:
            if probes>=budget:break
            if val>=starts[r]-1e-5:continue
            ss={**starts,r:val}
            pp={k:dict(v) for k,v in pauses.items()}
            if idx:pp.setdefault(r,{})[idx]=wait
            if layer_end(z,plan,ss,pp)>=duration-.001:continue
            ok=safe(ss,pp)
            if ok is True:
                starts,pauses=ss,pp
                duration=layer_end(z,plan,ss,pp)
                accepted+=1
        if probes>=budget:break
    return starts,pauses,duration,probes,accepted


def optimize(root,model,trial='bead_v5_pilot',v4_trial='parallel_02_03_pilot',
             budget=100,passes=2,max_layers=0,timeout=3600,seed=42):
    root=Path(root).resolve();model=str(model).zfill(2)
    if model=='01':raise ValueError('Model 01 is deliberately serial-only and excluded from v5')
    source,baseline,origin=source_for(root,model,v4_trial)
    reference=read_tracks(source/'trajectory.csv')
    blocks,plan=layer_plan(reference)
    if not layer_order_ok(reference):raise RuntimeError('Baseline layer order is invalid')
    parks=parked_positions(reference,plan,blocks)
    signatures=deposition_signature(reference),travel_signature(reference)
    reportdir=root/'ppo_project'/'waam_parallel_bead_v5'/trial/model
    reportdir.mkdir(parents=True,exist_ok=True)
    t0=time.time();schedules={};probes=0;changes=0;details=[]
    levels=sorted(plan)
    for k,z in enumerate(levels):
        if max_layers and k>=max_layers:break
        start,pauses,duration,n,accepted=optimize_layer(z,plan,blocks,parks[z],source/'config.yaml',budget,passes,seed)
        schedules[z]=(start,pauses)
        base_end=layer_end(z,plan,{r:plan[z][r]['offset'] for r in plan[z]}, {})
        gain=base_end-duration
        probes+=n;changes+=accepted
        details.append({'z':z,'baseline_s':base_end,'candidate_s':duration,'saved_s':gain,
                        'collision_probes':n,'accepted_changes':accepted,
                        'bead_waits':sum(len(v) for v in pauses.values())})
        print(f'[{model}] layer {k+1}/{len(levels)} z={z:g}: saved={gain:.3f}s, probes={n}, accepted={accepted}',flush=True)
    candidate=merge_layers(reference,blocks,plan,parks,schedules)
    if deposition_signature(candidate)!=signatures[0] or travel_signature(candidate)!=signatures[1]:
        raise RuntimeError('Deposited or traveled segment signature changed: candidate rejected')
    if not layer_order_ok(candidate):raise RuntimeError('Layer order changed: candidate rejected')
    new=real_finish(candidate)
    simult=simultaneous_deposition_seconds(candidate)
    status='KEEP_BASELINE_PASS';official=None
    savepath=reportdir/'candidate'
    if new<baseline-.01:
        if not fast_collision_check(candidate,source/'config.yaml'):
            print('Candidate failed full-model collision precheck; preserving baseline',flush=True)
            status='KEEP_BASELINE_PASS_FULL_COLLISION_FAIL'
        else:
            official=validate_official(source,candidate,savepath,timeout)
            if official is not None and official<baseline-.01:
                status='NEW_PASS_BETTER'
            else:status='KEEP_BASELINE_PASS_VALIDATOR_FAIL_OR_SLOW'
    else:
        print('No gain over verified baseline; official CLI skipped.',flush=True)
    result={'model':model,'status':status,'source_baseline':origin,'source_path':str(source),
        'baseline_validated_makespan_s':baseline,'candidate_precheck_makespan_s':new,
        'candidate_official_pass_makespan_s':official,
        'candidate_simultaneous_deposition_s':simult,'baseline_simultaneous_deposition_s':simultaneous_deposition_seconds(reference),
        'candidate_path':str(savepath) if official is not None else None,
        'safe_fallback':str(source),
        'collisions_prechecked':probes,'accepted_local_changes':changes,
        'layers':details,'elapsed_wall_s':time.time()-t0,'note':'No bead rearrangement or robot reassignment; timed starts and safe pauses only.'}
    write_json(reportdir/'report.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='layers'},ensure_ascii=False,indent=2),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--models',nargs='+',default=['02','03'])
    p.add_argument('--trial',default='bead_v5_pilot')
    p.add_argument('--v4-trial',default='parallel_02_03_pilot')
    p.add_argument('--max-probes-per-layer',type=int,default=100)
    p.add_argument('--passes',type=int,default=2)
    p.add_argument('--max-layers',type=int,default=0)
    p.add_argument('--timeout',type=int,default=3600)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    for m in args.models:
        optimize(args.root,m,args.trial,args.v4_trial,args.max_probes_per_layer,args.passes,args.max_layers,args.timeout,args.seed)

if __name__=='__main__':main()
