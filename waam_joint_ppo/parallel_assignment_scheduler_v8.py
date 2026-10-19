"""WAAM v8: safe, PASS-first bead reorder/robot reassignment + parallel timing.

Full D segment set (xyz + original speed) and strict inter-layer deposition order
are immutable. Each proposal may change (1) within-layer bead ordering,
(2) robot assignment within XY reach, (3) robot start delays and T motions.
Safety: real Validator collision precheck, then its OFFICIAL CLI for whole candidate.
No candidate replaces the immutable PASS source without a faster official PASS.
This is a bounded stochastic heuristic, not an optimality certificate nor PPO.
"""
from __future__ import annotations
import argparse, copy, csv, hashlib, itertools, json, math, random, shutil, time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple
import numpy as np
import yaml
from waam_joint_ppo.pass_local_bc_ppo_v3 import (read_tracks, real_finish, layer_plan,
    layer_order_ok, fast_collision_check, validate_official, write_json)
from waam_joint_ppo.parallel_pass_scheduler_v4 import parked_positions, simultaneous_deposition_seconds

XYZ_TOL=1e-5


@dataclass(frozen=True)
class Bead:
    key: str
    layer: float
    source_robot: int
    points: tuple  # xyz of all vertices in the original deposited line
    durations: tuple  # validated durations of deposited segments

    @property
    def length(self):
        return sum(math.dist(a,b) for a,b in zip(self.points,self.points[1:]))

    @property
    def seconds(self):
        return sum(self.durations)


def extract_beads(tracks):
    """Extract maximal uninterrupted D runs, preserving each original D segment."""
    result={}
    for rid,rows in tracks.items():
        run_points=[];run_durations=[];run_z=None;index=0
        def finish():
            nonlocal run_points,run_durations,run_z,index
            if run_durations:
                z=round(run_z,4)
                bead=Bead(f'{rid}:{index:05}',z,rid,tuple(run_points),tuple(run_durations))
                result.setdefault(z,{1:[],2:[],3:[]})[rid].append(bead)
                index+=1
            run_points=[];run_durations=[];run_z=None
        for a,b in zip(rows,rows[1:]):
            if a[4]=='D':
                z=(a[3]+b[3])/2
                if run_durations and abs(z-run_z)>0.25:
                    finish()
                if not run_durations:
                    run_z=z;run_points=[tuple(a[1:4])]
                if math.dist(run_points[-1],a[1:4])>XYZ_TOL:
                    raise ValueError('A deposited path has discontinuous vertices')
                run_points.append(tuple(b[1:4]))
                run_durations.append(b[0]-a[0])
            else:finish()
        finish()
    return dict(sorted(result.items()))


def immutable_d_signature(beads):
    """Assignment/order invariant multi-set; includes deposited XYZ and durations."""
    payload=[]
    for layer in beads.values():
        for jobs in layer.values():
            for bead in jobs:
                for a,b,d in zip(bead.points,bead.points[1:],bead.durations):
                    payload.append((tuple(round(float(v),5) for v in a),
                                    tuple(round(float(v),5) for v in b),round(d,4)))
    return tuple(sorted(payload))


def from_tracks_signature(tracks):
    return immutable_d_signature(extract_beads(tracks))


def read_cell(source):
    cfg=yaml.safe_load((source/'config.yaml').read_text(encoding='utf-8-sig'))
    robots={int(r['id']):r for r in cfg['robots']}
    if set(robots)!={1,2,3}:raise ValueError('Only the benchmark 3-robot cell is supported')
    return cfg,robots


def bead_reachable(bead,rid,robots,margin=0.1):
    robot=robots[rid]
    bx,by=robot['base_xyz_mm'][:2]
    radius=float(robot['xy_reach_radius_mm'])
    return all(math.hypot(p[0]-bx,p[1]-by)<radius-margin for p in bead.points)


def assignments_reachable(tasks,robots):
    return all(bead_reachable(j,r,robots) for r,jobs in tasks.items() for j in jobs)


def _move(rows,pos,velocity,mode):
    old=rows[-1][1:4]
    delta=math.dist(old,pos)
    if delta<1e-7:return
    if velocity<=0:raise ValueError('Invalid velocity')
    rows[-1][4]=mode
    rows.append([rows[-1][0]+delta/velocity,*pos,'W'])


def _wait(rows,seconds):
    if seconds<=1e-9:return
    rows[-1][4]='W'
    rows.append([rows[-1][0]+seconds,*rows[-1][1:4],'W'])


def _deposit(rows,bead):
    for a,b,d in zip(bead.points,bead.points[1:],bead.durations):
        if math.dist(rows[-1][1:4],a)>XYZ_TOL:
            raise ValueError('Deposit start not at travel endpoint')
        if d<=0:raise ValueError('Non-positive D duration')
        rows[-1][4]='D'
        rows.append([rows[-1][0]+d,*b,'W'])


def layer_routes(layer_tasks,start_parks,end_parks,robots,config,pauses=None):
    """Full three-robot local routes with T recomputed and original D preserved.

    Uses direct 3D travel at configured speed; Validator must approve all travel.
    Row mode represents interval starting at that row, as in Validator CSV.
    """
    if not assignments_reachable(layer_tasks,robots):raise ValueError('Out-of-reach assignment')
    speed=float(config['process']['travel_speed_mm_s'])
    rows={}
    pauses=pauses or {}
    for r in (1,2,3):
        rr=[[0.,*start_parks[r],'W']]
        for i,j in enumerate(layer_tasks[r]):
            delay=pauses.get((r,i),0.)
            _wait(rr,delay)
            _move(rr,j.points[0],speed,'T')
            _deposit(rr,j)
        _move(rr,end_parks[r],speed,'T')
        rows[r]=rr
    return rows


def _offset_routes(routes,starts):
    """Delay entire independent trajectories without changing D or T durations."""
    end=max(starts.get(r,0.)+routes[r][-1][0] for r in (1,2,3))
    out={}
    for r in (1,2,3):
        delay=float(starts.get(r,0.))
        if delay<0:raise ValueError('Negative robot start')
        seg=routes[r]
        if delay>1e-7:
            rr=[[0.,*seg[0][1:4],'W'],[delay,*seg[0][1:4],seg[0][4]]]
            rr.extend([[row[0]+delay,*row[1:]] for row in seg[1:]])
        else:rr=[[row[0],*row[1:]] for row in seg]
        if rr[-1][0]<end-1e-8:
            _wait(rr,end-rr[-1][0])
        else:rr[-1][4]='W'
        # Always include a positive terminal W interval; compatible with local validator.
        rr.append([end+.001,*rr[-1][1:4],'W'])
        out[r]=rr
    return out


def merge_scheduled_layers(reference,plans):
    """Concatenate safe local schedules; no overlap between different layers."""
    out={r:[[0.,*reference[r][0][1:4],'W']] for r in (1,2,3)}
    clock=0.
    for z,local in sorted(plans.items()):
        duration=local[1][-1][0]
        if any(abs(local[r][-1][0]-duration)>1e-5 for r in (1,2,3)):
            raise ValueError('Local robots have different terminal time')
        for r in (1,2,3):
            dest=out[r]
            if math.dist(dest[-1][1:4],local[r][0][1:4])>XYZ_TOL:
                raise ValueError(f'Layer {z}: R{r} pose discontinuity')
            if clock>dest[-1][0]+1e-8:
                dest[-1][4]='W'
                dest.append([clock,*dest[-1][1:4],local[r][0][4]])
            else:dest[-1][4]=local[r][0][4]
            for row in local[r][1:]:
                dest.append([clock+row[0],*row[1:]])
        clock+=duration
    return out


def duration_scheduled(local):return local[1][-1][0]


def delay_patterns(routes,base_duration,budget=20,seed=0,reference_offsets=None):
    """Produce joint starts: simultaneous, a pair ahead of another, or serial.

    Searches non-monotonically because starting all three at once may collide,
    whereas starting e.g. R1+R3 ahead of R2 is safe (as in model 02 v5).
    """
    rng=random.Random(seed)
    lengths={r:routes[r][-1][0] for r in (1,2,3)}
    items=[]
    def add(starts):
        t=max(starts[r]+lengths[r] for r in (1,2,3))
        if t<base_duration-0.002:
            items.append((t,tuple(round(starts[r],5) for r in (1,2,3)),starts))
    add({1:0.,2:0.,3:0.})
    # Crucial: v5 model-02 PASS starts R1+R3 at 0, R2 at 125s.
    # Generalize that SAFE ordering as an anchor instead of testing only
    # overly aggressive offsets which all collide.
    if reference_offsets:
        baseline={r:float(reference_offsets.get(r,0.)) for r in (1,2,3)}
        add(baseline)
        for active in (1,2,3):
            for delta in (-15.,-10.,-6.,-3.,-1.,1.,3.,6.,10.,15.):
                trial=dict(baseline);trial[active]=max(0.,trial[active]+delta)
                add(trial)
        for a,b in itertools.combinations((1,2,3),2):
            for delta in (-8.,-4.,4.,8.):
                trial=dict(baseline)
                trial[a]=max(0.,trial[a]+delta);trial[b]=max(0.,trial[b]+delta)
                add(trial)
    for a,b in itertools.combinations((1,2,3),2):
        c=({1,2,3}-{a,b}).pop()
        finish=max(lengths[a],lengths[b])
        for lag in (0.,3.,8.,15.):
            for factor in (.35,.55,.7,.85,1.,1.05):
                add({a:0.,b:lag,c:max(0.,factor*finish)})
                add({a:lag,b:0.,c:max(0.,factor*finish)})
        for third in (75.,90.,100.,110.,115.,120.,125.,130.,135.,140.,145.,150.,160.,175.):
            add({a:0.,b:0.,c:third})
    for order in itertools.permutations((1,2,3)):
        starts={order[0]:0.,order[1]:lengths[order[0]],order[2]:lengths[order[0]]+lengths[order[1]]}
        add(starts)
    # Explore across the timing range, with reference-adjacent cases first.
    # Starting three robots at zero is fast but almost always unsafe.
    items.sort(key=lambda x:x[0]);unique=[];seen=set()
    seeded=[]
    if reference_offsets:
        baseline=tuple(round(reference_offsets.get(r,0.),5) for r in (1,2,3))
        nearby=[x for x in items if max(abs(x[1][k]-baseline[k]) for k in range(3))<=20.]
        nearby.sort(key=lambda x:(max(abs(x[1][k]-baseline[k]) for k in range(3)),x[0]))
        seeded=nearby[:max(4,budget//2)]
    stride=max(1,len(items)//max(1,budget))
    selection=seeded+items[:min(4,budget)]+items[::stride]
    for x in selection:
        if x[1] not in seen:unique.append(x);seen.add(x[1])
        if len(unique)>=budget:break
    return [z[2] for z in unique]



def quick_collision_reject(tracks,config,step_s=1.):
    """Fast vectorized necessary-condition filter; NEVER certifies safety.

    At sampled time instants any definite TCP or segment-segment capsule violation
    rejects the candidate. A surviving candidate must still be checked with the
    official Validator collision simulator (and final complete official CLI).
    """
    cs=config['collision']; robots={int(r['id']):r for r in config['robots']}
    last=max(row[-1][0] for row in tracks.values())
    grid=np.arange(0.,last+step_s,step_s)
    events=np.asarray([row[0] for rows in tracks.values() for row in rows],dtype=float)
    sample=np.unique(np.concatenate([grid,events]))
    poses={}
    for r in (1,2,3):
        rows=tracks[r];tt=np.asarray([x[0] for x in rows]);xy=np.asarray([x[1:3] for x in rows],float)
        poses[r]=np.column_stack([np.interp(sample,tt,xy[:,0]),np.interp(sample,tt,xy[:,1])])
    def cross(a,b):return a[:,0]*b[:,1]-a[:,1]*b[:,0]
    def segdist(a,b,c,d):
        # Each segment is A->B or C->D for every time. Intersect => 0.
        ab=b-a;cd=d-c;ac=c-a;ad=d-a;ca=a-c;cb=b-c
        o1=cross(ab,ac);o2=cross(ab,ad);o3=cross(cd,ca);o4=cross(cd,cb)
        bbox=(np.minimum(a[:,0],b[:,0])<=np.maximum(c[:,0],d[:,0])+1e-9) & \
             (np.minimum(c[:,0],d[:,0])<=np.maximum(a[:,0],b[:,0])+1e-9) & \
             (np.minimum(a[:,1],b[:,1])<=np.maximum(c[:,1],d[:,1])+1e-9) & \
             (np.minimum(c[:,1],d[:,1])<=np.maximum(a[:,1],b[:,1])+1e-9)
        intersect=(o1*o2<=0)&(o3*o4<=0)&bbox
        def point_seg(p,s,e):
            v=e-s;norm=np.sum(v*v,axis=1);q=np.sum((p-s)*v,axis=1)/np.maximum(norm,1e-12)
            q=np.clip(q,0.,1.);delta=p-(s+q[:,None]*v)
            return np.sqrt(np.sum(delta*delta,axis=1))
        return np.where(intersect,0.,np.minimum.reduce([point_seg(a,c,d),point_seg(b,c,d),
                                               point_seg(c,a,b),point_seg(d,a,b)]))
    for r,s in ((1,2),(1,3),(2,3)):
        pr,ps=poses[r],poses[s]
        tcp=float(robots[r]['tcp_radius_mm'])+float(robots[s]['tcp_radius_mm'])
        if cs['check_tcp_radius'] and np.any(np.linalg.norm(pr-ps,axis=1)<=tcp+1e-6):return True
        arm=float(robots[r]['arm_envelope_radius_mm'])+float(robots[s]['arm_envelope_radius_mm'])+float(cs['arm_clearance_mm'])
        if cs['check_arm_envelope']:
            br=np.repeat(np.asarray(robots[r]['base_xyz_mm'][:2],float)[None,:],len(sample),axis=0)
            bs=np.repeat(np.asarray(robots[s]['base_xyz_mm'][:2],float)[None,:],len(sample),axis=0)
            if np.any(segdist(br,pr,bs,ps)<=arm+1e-6):return True
    return False


def evaluate_assignment(tasks,z,start_parks,end_parks,robots,config,baseline_layer_s,
                        safety,probe_budget=8,seed=0,reference_offsets=None):
    """Return fastest validated-safe schedule for ONE fixed assignment/order."""
    if not assignments_reachable(tasks,robots):return None,0
    routes=layer_routes(tasks,start_parks,end_parks,robots,config)
    tested=0;best=None;score=baseline_layer_s
    for starts in delay_patterns(routes,score,probe_budget,seed,reference_offsets):
        trial=_offset_routes(routes,starts)
        if duration_scheduled(trial)>=score-.002:continue
        tested+=1
        if quick_collision_reject(trial,config):continue
        if safety(trial):
            best=trial;score=duration_scheduled(trial)
    return best,tested


def mutate_tasks(tasks,robots,rng,attempts=65):
    """Generate bounded local proposals: reverse/swap, transfer, cross-robot swap."""
    keys=(1,2,3);out=[];seen=set()
    def emit(new,tag):
        assignment=tuple((r,tuple(x.key for x in new[r])) for r in keys)
        if assignment not in seen and assignments_reachable(new,robots):
            seen.add(assignment);out.append((new,tag))
    # Workload-driven candidates first. When deposition is strongly concentrated
    # on one robot, move reachable beads to a less-loaded robot; never assume
    # load balancing is the objective (all candidates are still timed/validated).
    loads={r:sum(b.seconds for b in tasks[r]) for r in keys}
    bottleneck=max(keys,key=lambda r:loads[r])
    for bead in sorted(tasks[bottleneck], key=lambda b:-b.seconds):
        for target in sorted((r for r in keys if r!=bottleneck),key=lambda r:loads[r]):
            if not bead_reachable(bead,target,robots):continue
            for insertion in sorted({0,len(tasks[target]),len(tasks[target])//2}):
                new={r:list(tasks[r]) for r in keys}
                new[bottleneck].remove(bead)
                new[target].insert(insertion,bead)
                emit(new,'critical_transfer')
                if len(out)>=min(attempts//3,30):break
            if len(out)>=min(attempts//3,30):break
        if len(out)>=min(attempts//3,30):break
    for _ in range(attempts):
        new={r:list(tasks[r]) for r in keys}
        typ=rng.choice(('swap','reverse','transfer','transfer','cross'))
        if typ=='swap':
            r=rng.choice(keys)
            if len(new[r])<2:continue
            a=rng.randrange(len(new[r])-1);new[r][a],new[r][a+1]=new[r][a+1],new[r][a]
        elif typ=='reverse':
            r=rng.choice(keys)
            if len(new[r])<2:continue
            a=rng.randrange(len(new[r])-1);b=rng.randrange(a+1,len(new[r]))
            new[r][a:b+1]=reversed(new[r][a:b+1])
        elif typ=='transfer':
            nonempty=[r for r in keys if new[r]]
            if not nonempty:continue
            a=rng.choice(nonempty);b=rng.choice([r for r in keys if r!=a]);ix=rng.randrange(len(new[a]))
            if not bead_reachable(new[a][ix],b,robots):continue
            job=new[a].pop(ix);j=rng.randrange(len(new[b])+1);new[b].insert(j,job)
        else:
            a,b=rng.sample(list(keys),2)
            if not new[a] or not new[b]:continue
            ia=rng.randrange(len(new[a]));ib=rng.randrange(len(new[b]))
            if not(bead_reachable(new[a][ia],b,robots) and bead_reachable(new[b][ib],a,robots)):continue
            new[a][ia],new[b][ib]=new[b][ib],new[a][ia]
        emit(new,typ)
    return out


def optimize_layer(z,tasks,old_duration,start_parks,end_parks,robots,config,
                   collision,budget=70,beam=3,seed=42,reference_offsets=None):
    """Best-improvement search of distinct mutations. Safety failures are not accepted."""
    rng=random.Random(seed+int(z*1000))
    best=None;best_duration=old_duration;best_jobs=tasks
    seen=0;accepted=0;categories={};winner_types=[]
    # The first pass explores alternative bead permutations and robot ownership.
    proposals=mutate_tasks(tasks,robots,rng,max(budget*4,80))
    # Do not starve bead reordering when guided transfers fill the search budget.
    guided=[pair for pair in proposals if pair[1]=='critical_transfer']
    other=[pair for pair in proposals if pair[1]!='critical_transfer']
    mixed=[]
    for i in range(max(len(guided),len(other))):
        if i<len(other):mixed.append(other[i])
        if i<len(guided):mixed.append(guided[i])
    proposals=mixed
    for new,tag in proposals:
        if seen>=budget:break
        # Cheap optimistic bound: remaining T travel cannot be faster than 0.
        dep_lb=max(sum(x.seconds for x in new[r]) for r in (1,2,3))
        if dep_lb>=best_duration-.002:continue
        found,n=evaluate_assignment(new,z,start_parks,end_parks,robots,config,best_duration,
                                    collision,probe_budget=beam*6,seed=seed+seen,reference_offsets=reference_offsets)
        seen+=1;categories[tag]=categories.get(tag,0)+1
        if found is not None:
            duration=duration_scheduled(found)
            if duration<best_duration-.002:
                best_duration=duration;best=found;best_jobs=new;accepted+=1;winner_types.append(tag)
    return best,best_duration,seen,accepted,categories,winner_types


def source_for(root,model,v4_trial='unused',source_override=None):
    """Explicit source only: v8 never chooses a path based on a model ID."""
    if not source_override:
        raise ValueError('v8 search requires --source-dir or the automatic router')
    source=Path(source_override).resolve()
    for name in ('config.yaml','target.stl','trajectory.csv'):
        if not (source/name).is_file():raise FileNotFoundError(source/name)
    return source,'official_pass_source_selected_by_hash'


def verify_official_source(source,output,timeout=3600):
    """Always revalidate baseline in user's installation; never trust a bare label."""
    import subprocess,sys
    command=[sys.executable,'-m','waam_validator.cli','run',str(source),'--json']
    try:p=subprocess.run(command,capture_output=True,text=True,timeout=timeout)
    except subprocess.TimeoutExpired:raise RuntimeError('Baseline Validator CLI timeout')
    (output/'source_validator_stdout.txt').write_text(p.stdout,encoding='utf-8')
    (output/'source_validator_stderr.txt').write_text(p.stderr,encoding='utf-8')
    try:result=json.loads(p.stdout)
    except Exception:raise RuntimeError(f'Baseline Validator CLI did not return JSON: {p.stderr[-600:]}')
    if p.returncode or result.get('status')!='PASS':
        raise RuntimeError(f'Baseline is not official PASS: {result.get("status")} {result.get("failure_reasons")}')
    return result


def execute(root,model,trial='v6_pilot',v4_trial='parallel_02_03_pilot',
            budget=70,beam=3,max_layers=0,max_beads_per_layer=100,source_dir=None,seed=42,timeout=3600,
            baseline_verified_report=None):
    root=Path(root).resolve();model=str(model).zfill(2)
    source,origin=source_for(root,model,v4_trial,source_dir)
    report=root/'ppo_project'/'waam_auto_optimizer_v8'/trial/model/'search'
    report.mkdir(parents=True,exist_ok=True)
    if baseline_verified_report:
        original_report=json.loads(Path(baseline_verified_report).read_text(encoding='utf-8'))
        if original_report.get('status')!='PASS':raise ValueError('Report does not say PASS')
        # If user wants to skip revalidation, source must have same fixed hash.
        raise ValueError('External report reuse disabled until source digest binding is implemented')
    original_report=verify_official_source(source,report,timeout)
    validated_base=float(original_report['schedule']['makespan_s'])
    tracks=read_tracks(source/'trajectory.csv');original_signature=from_tracks_signature(tracks)
    if not layer_order_ok(tracks):raise ValueError('Official baseline does not have strict layer ordering')
    cfg,robots=read_cell(source)
    blocks,plan=layer_plan(tracks)
    parks=parked_positions(tracks,plan,blocks)
    extracted=extract_beads(tracks)
    if set(extracted)!=set(plan):raise ValueError('Layer extraction does not match reference')
    # Each robot has a known physical terminal park pose from the PASS source.
    terminals={z:{r:list(blocks[(z,r)][-1][1:4]) if (z,r) in blocks else parks[z][r]
                     for r in (1,2,3)} for z in plan}
    levels=sorted(plan);local_solutions={}
    tested=accepted=0;notes=[];wall=time.monotonic();searched_count=0
    for i,z in enumerate(levels):
        baseline_layer=max(q['offset']+q['duration'] for q in plan[z].values())
        jobcount=sum(len(extracted[z][r]) for r in (1,2,3))
        if jobcount>max_beads_per_layer or (max_layers and searched_count>=max_layers):
            why=('bead_budget_exceeded' if jobcount>max_beads_per_layer else 'layer_budget_exceeded')
            notes.append({'layer':i+1,'z_mm':z,'baseline_s':baseline_layer,'candidate_s':baseline_layer,
                          'saved_s':0.,'proposals_checked':0,'accepted_changes':0,
                          'skip_reason':why,'bead_count':jobcount})
            print(f'[{model}] layer {i+1}/{len(levels)} SKIP ({why}, beads={jobcount})',flush=True)
            continue
        searched_count+=1
        best,seconds,n,a,cats,winners=optimize_layer(z,extracted[z],baseline_layer,parks[z],terminals[z],
                         robots,cfg,lambda cand:fast_collision_check(cand,source/'config.yaml'),
                         budget=budget,beam=beam,seed=seed,reference_offsets={r:plan[z][r]['offset'] for r in plan[z]})
        tested+=n;accepted+=a
        if best is not None:
            local_solutions[z]=best
        notes.append({'layer':i+1,'z_mm':z,'baseline_s':baseline_layer,'candidate_s':seconds,
                      'saved_s':baseline_layer-seconds,'proposals_checked':n,
                      'accepted_changes':a,'tested_mutation_types':cats,'accepted_mutation_types':winners})
        print(f'[{model}] layer {i+1}/{len(levels)} z={z:g}: saved={baseline_layer-seconds:.3f}s, '+
              f'checked={n}, accepted={a}',flush=True)
    if not local_solutions:
        status='KEEP_BASELINE_PASS_NO_GAIN';final_candidate=None;new=validated_base;verified=None
    else:
        # Include untouched layers EXACTLY from original with the existing v5 builder.
        from waam_joint_ppo.parallel_bead_scheduler_v5 import layer_tracks
        plans={}
        for z in levels:
            if z in local_solutions:plans[z]=local_solutions[z]
            else:
                defaults={r:float(plan[z][r]['offset']) for r in plan[z]}
                plans[z]=layer_tracks(z,plan,blocks,parks[z],defaults,{})
        generated=merge_scheduled_layers(tracks,plans)
        if from_tracks_signature(generated)!=original_signature:
            raise RuntimeError('Deposition XYZ/speed changed or lost: FAIL CLOSED')
        if not layer_order_ok(generated):raise RuntimeError('Layer ordering changed')
        new=real_finish(generated)
        verified=None;status='KEEP_BASELINE_PASS_NO_GAIN';final_candidate=None
        if new<validated_base-.01:
            if fast_collision_check(generated,source/'config.yaml'):
                tentative=report/'candidate'
                verified=validate_official(source,generated,tentative,timeout)
                if verified is not None and verified<validated_base-.01:
                    status='NEW_PASS_BETTER';final_candidate=str(tentative)
                else:status='KEEP_BASELINE_PASS_OFFICIAL_FAILED'
            else:status='KEEP_BASELINE_PASS_GLOBAL_COLLISION'
    result={'model':model,'status':status,'source_type':origin,'source':str(source),
            'verified_baseline_s':validated_base,'candidate_precheck_s':new,
            'candidate_official_pass_s':verified,'candidate':final_candidate,
            'safe_fallback':str(source),'searched_layers':len(notes),
            'accepted_local_changes':accepted,'mutation_proposals_checked':tested,
            'elapsed_wall_s':time.monotonic()-wall,
            'scope':'same-layer order changes + XY-feasible robot reassignment + new T + collision-safe starts',
            'guarantee':'only official PASS faster candidates are promoted; no optimality certificate',
            'layers':notes}
    write_json(report/'report.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='layers'},ensure_ascii=False,indent=2),flush=True)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path('.'))
    ap.add_argument('--models',nargs='+',default=['02','03'])
    ap.add_argument('--trial',default='v6_pilot')
    ap.add_argument('--v4-trial',default='parallel_02_03_pilot')
    ap.add_argument('--source-dir',type=Path,default=None,help='Only with ONE model')
    ap.add_argument('--mutation-probes-per-layer',type=int,default=24)
    ap.add_argument('--beam',type=int,default=2,help='Timing candidate budget scale')
    ap.add_argument('--max-layers',type=int,default=0,help='Limit searched layers, validate full model')
    ap.add_argument('--max-beads-per-layer',type=int,default=100,help='Skip costly layers')
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--timeout',type=int,default=3600)
    a=ap.parse_args()
    if a.source_dir and len(a.models)!=1:ap.error('--source-dir requires one --models entry')
    for m in a.models:
        execute(a.root,m,a.trial,a.v4_trial,max(1,a.mutation_probes_per_layer),
                max(1,a.beam),a.max_layers,a.max_beads_per_layer,a.source_dir,a.seed,a.timeout)

if __name__=='__main__':main()
