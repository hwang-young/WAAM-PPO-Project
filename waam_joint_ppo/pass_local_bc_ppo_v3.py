"""PASS-first BC + clipped PPO on model-independent layer-start features.

Scope (v3): preserve robot assignment, all D/T waypoints and layer ordering. PPO
proposes early starts for already-defined robot activity blocks; the safety shield
shrinks proposals and accepts changes only after an official Validator PASS.
This is NOT a global optimality proof or a generic path/assignment optimizer.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, random, shutil, subprocess, sys, tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

COLS = ['robot_id','time_s','x_mm','y_mm','z_mm','mode']
ACTIONS = (0., .25, .65)  # keep; remove 25%; remove 65% of the selected W interval
FEATURES = 9


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def locate_source(root, model):
    bank = root/'ppo_project'/'centerline_general_bank_v1'
    leaderboard = json.loads((bank/'leaderboard.json').read_text(encoding='utf-8'))
    record = leaderboard.get(model, {})
    if record.get('status') != 'PASS':
        raise RuntimeError(f'Model {model} has no official PASS leaderboard entry')
    p = Path(record['candidate'])
    if not p.is_dir():
        p = bank/model/p.name  # handle bank moved to a different machine/path
    for name in ('trajectory.csv','config.yaml','target.stl','official_validation.json'):
        if not (p/name).is_file(): raise FileNotFoundError(p/name)
    rep=json.loads((p/'official_validation.json').read_text(encoding='utf-8'))
    if rep.get('status')!='PASS':raise RuntimeError(f'Unverified baseline: {p}')
    expected=float(record['best_makespan_s'])
    observed=float(rep['makespan_s'])
    if abs(expected-observed)>1e-3:raise RuntimeError('Leaderboard and official report disagree')
    meta_file=root/'ppo_project'/'waam_centerline_shared_v1'/'datasets'/model/'manifest.json'
    if meta_file.exists():
        meta=json.loads(meta_file.read_text(encoding='utf-8'))
        if meta.get('source_trajectory_sha256')!=sha(p/'trajectory.csv'):
            raise RuntimeError('PASS source trajectory differs from tracked dataset')
    return p,expected


def read_tracks(path):
    tracks=defaultdict(list)
    with Path(path).open(newline='',encoding='utf-8-sig') as f:
        rr=csv.DictReader(f)
        if rr.fieldnames!=COLS:raise ValueError('Wrong trajectory CSV header')
        for row in rr:
            rid=int(row['robot_id']);mode=row['mode']
            if rid not in (1,2,3) or mode not in ('W','T','D'):raise ValueError('Invalid robot or mode')
            tracks[rid].append([float(row['time_s']),float(row['x_mm']),float(row['y_mm']),float(row['z_mm']),mode])
    if set(tracks)!={1,2,3}:raise ValueError('Need three robot trajectories')
    for r,rr in tracks.items():
        if rr[0][0]!=0 or any(b[0]<=a[0] for a,b in zip(rr,rr[1:])):
            raise ValueError(f'Non-increasing time for R{r}')
    return dict(tracks)


def write_tracks(path, tracks):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(COLS)
        for rid in (1,2,3):
            for row in tracks[rid]:
                w.writerow([rid,f'{row[0]:.9f}',*[f'{v:.6f}' for v in row[1:4]],row[4]])


def clone(tracks):
    return {r:[row.copy() for row in v] for r,v in tracks.items()}


def motion_finish_index(rows):
    indices=[i for i,row in enumerate(rows[:-1]) if row[4] in ('T','D')]
    return max(indices)+1 if indices else 0


def get_gaps(tracks, min_wait=6.):
    gaps=[]
    for rid,rows in tracks.items():
        last=motion_finish_index(rows)
        for i in range(last):
            a,b=rows[i:i+2]
            if a[4]!='W' or b[0]-a[0]<min_wait:continue
            if max(abs(a[j]-b[j]) for j in (1,2,3))>1e-5:continue
            gaps.append((rid,i))
    return gaps


def real_finish(tracks):
    return max(rows[motion_finish_index(rows)][0] for rows in tracks.values())


def synchronize(tracks, slack=.001):
    out=clone(tracks)
    for rid,rows in out.items():
        out[rid]=rows[:motion_finish_index(rows)+1]
    end=max(v[-1][0] for v in out.values()) + slack
    for rid,rr in out.items():
        # The final interval is W. Keep the last motion's endpoint unchanged.
        rr[-1][4]='W'
        rr.append([end,*rr[-1][1:4],'W'])
    return out


def compress_gap(tracks, gap, fraction, max_advance):
    rid,i=gap
    out=clone(tracks);rr=out[rid]
    wait=rr[i+1][0]-rr[i][0]
    if not (rr[i][4]=='W' and wait>0):return None
    cut=min(wait-.03, max_advance, wait*fraction)
    if cut<=.001:return None
    for row in rr[i+1:]:row[0]-=cut
    out=synchronize(out)
    if any(any(b[0]-a[0]<1e-7 for a,b in zip(rows,rows[1:])) for rows in out.values()):return None
    return out


def layer_order_ok(tracks, margin=1e-5):
    layers=defaultdict(lambda:[math.inf,-math.inf])
    for rr in tracks.values():
        for a,b in zip(rr,rr[1:]):
            if a[4]=='D':
                key=round((a[3]+b[3])/2.,4)
                layers[key][0]=min(layers[key][0],a[0])
                layers[key][1]=max(layers[key][1],b[0])
    order=sorted(layers)
    return all(layers[lo][1]<=layers[hi][0]+margin for lo,hi in zip(order,order[1:]))


def feature(tracks,gap):
    rid,i=gap;rr=tracks[rid];a,b=rr[i:i+2]
    dur=b[0]-a[0];total=max(x[-1][0] for x in tracks.values())
    last=motion_finish_index(rr)
    zvals=[x[3] for x in rr if x[4]=='D']
    z_min=min(zvals) if zvals else 0
    z_max=max(zvals) if zvals else z_min+1
    return np.asarray([
        min(1.,dur/max(1.,total)),
        min(1.,dur/250.),
        a[0]/max(1.,total),
        max(0.,(rr[last][0]-b[0])/max(1.,total)),
        (rid-1)/2.,
        i/max(1,len(rr)-1),
        (a[3]-z_min)/max(1,z_max-z_min),
        min(1.,len(get_gaps(tracks))/120.),
        min(1.,total/30000.),
    ],dtype=np.float32)


def layer_blocks(tracks, min_wait=6.):
    """Extract one closed robot motion block per layer, delimited by long parked waits.

    Reject ambiguous trajectories: the input must provide one robot block per layer,
    each block must deposit only on one layer, and the original schedule must be
    layer-ordered. No geometry is synthesized by this routine.
    """
    if not layer_order_ok(tracks):raise ValueError('Original baseline violates global layer order')
    result={}
    for rid,rows in tracks.items():
        last=motion_finish_index(rows)
        gaps=[i for i in range(last) if rows[i][4]=='W' and rows[i+1][0]-rows[i][0]>=min_wait
              and max(abs(rows[i][j]-rows[i+1][j]) for j in (1,2,3))<1e-5]
        starts=[0]+[i+1 for i in gaps];ends=gaps+[last]
        for start,end in zip(starts,ends):
            segment=rows[start:end+1]
            zs={round((a[3]+b[3])/2,4) for a,b in zip(segment,segment[1:]) if a[4]=='D'}
            if not zs:continue
            if len(zs)!=1:raise ValueError(f'Robot {rid} motion block deposits across multiple layers')
            z=next(iter(zs))
            if (z,rid) in result:raise ValueError('Multiple blocks per robot per layer; use hierarchical scheduler')
            result[(z,rid)]=segment
    if not result:raise ValueError('No layer blocks found')
    return result


def layer_plan(tracks):
    blocks=layer_blocks(tracks)
    levels=sorted({z for z,_ in blocks})
    plan={}
    for z in levels:
        current=[(r,blocks[(z,r)]) for r in (1,2,3) if (z,r) in blocks]
        first=min(seg[0][0] for r,seg in current)
        plan[z]={r:{'offset':float(seg[0][0]-first),
                    'duration':float(seg[-1][0]-seg[0][0])} for r,seg in current}
    return blocks,plan


def layer_actions(plan):
    options=[]
    for z,rr in sorted(plan.items()):
        for rid,desc in rr.items():
            if desc['offset']>1.0:options.append((z,rid))
    return options


def layer_feature(tracks,plan,key):
    z,rid=key
    zs=sorted(plan)
    layer=plan[z];ref=layer[rid]
    maxduration=max(q['duration']+q['offset'] for q in layer.values())
    all_duration=max(1.,max(row[-1][0] for row in tracks.values()))
    return np.asarray([
        min(1.,ref['offset']/max(1.,maxduration)),
        min(1.,ref['offset']/250.),
        zs.index(z)/max(1,len(zs)-1),
        ref['duration']/max(1.,maxduration),
        (rid-1)/2.,
        min(1.,maxduration/1000.),
        len(layer)/3.,
        min(1.,len(zs)/100.),
        min(1.,all_duration/30000.),
    ],dtype=np.float32)


def rebuild_layers(original, advances, blocks=None, plan=None):
    """Re-time complete fixed-geometry layer blocks; only start offsets change.

    Advancing a robot block can enable same-layer parallel deposition. All layers
    remain globally ordered, and each robot retains its original sequence and poses.
    """
    if blocks is None or plan is None: blocks,plan=layer_plan(original)
    output={rid:[[0.,*original[rid][0][1:4],'W']] for rid in (1,2,3)}
    t_layer=0.
    for z in sorted(plan):
        items=plan[z]
        starttimes={rid:t_layer+max(0.,info['offset']-advances.get((z,rid),0.)) for rid,info in items.items()}
        for rid,tstart in starttimes.items():
            seg=blocks[(z,rid)];dest=output[rid]
            if tstart<dest[-1][0]-1e-6:raise ValueError('Robot activities would overlap in time')
            if max(abs(dest[-1][k]-seg[0][k]) for k in (1,2,3))>1e-4:
                raise ValueError('Robot block poses are not continuous; repair geometry first')
            if tstart>dest[-1][0]+1e-7:
                dest[-1][4]='W'
                dest.append([tstart,*seg[0][1:4],seg[0][4]])
            else:
                dest[-1][4]=seg[0][4]
            first=seg[0][0]
            for row in seg[1:]:dest.append([tstart+row[0]-first,*row[1:]])
        t_layer=max(starttimes[r]+items[r]['duration'] for r in items)
    return synchronize(output)


def deposition_signature(tracks):
    return [(rid,a[1:4],b[1:4],round(b[0]-a[0],4))
            for rid in (1,2,3) for a,b in zip(tracks[rid],tracks[rid][1:]) if a[4]=='D']


def travel_signature(tracks):
    return [(rid,a[1:4],b[1:4],round(b[0]-a[0],4))
            for rid in (1,2,3) for a,b in zip(tracks[rid],tracks[rid][1:]) if a[4]=='T']


def attempt_layer_change(original,advances,key,action,max_advance,blocks,plan):
    if action==0:return advances,None
    offset=plan[key[0]][key[1]]['offset']
    prior=advances.get(key,0.)
    change=min(max_advance,ACTIONS[action]*offset,offset-prior)
    if change<=.0001:return advances,None
    candidate_advances=dict(advances);candidate_advances[key]=prior+change
    candidate=rebuild_layers(original,candidate_advances,blocks,plan)
    if deposition_signature(candidate)!=deposition_signature(original):
        raise AssertionError('Layer reconstruction changed deposition geometry/speed')
    if travel_signature(candidate)!=travel_signature(original):
        raise AssertionError('Layer reconstruction changed travel geometry/speed')
    if not layer_order_ok(candidate):
        raise AssertionError('Layer reconstruction violates global layer order')
    return candidate_advances,candidate


class GapPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.body=nn.Sequential(nn.Linear(FEATURES,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh())
        self.actor=nn.Linear(64,len(ACTIONS));self.critic=nn.Linear(64,1)
    def forward(self,x):
        y=self.body(x)
        return self.actor(y),self.critic(y).squeeze(-1)


def bc_initialize(policy, references, epochs=25, move_type="layer"):
    observations=[]
    for tracks in references:
        if move_type=="layer":
            _,plan=layer_plan(tracks)
            observations.extend(layer_feature(tracks,plan,k) for k in layer_actions(plan))
        else:
            observations.extend(feature(tracks,g) for g in get_gaps(tracks))
    if not observations:raise RuntimeError('No reference wait gaps for behavior cloning')
    x=torch.tensor(np.stack(observations));targets=torch.zeros(len(x),dtype=torch.long)
    opt=torch.optim.Adam(policy.parameters(),lr=.002)
    for _ in range(epochs):
        logits,_=policy(x)
        # Mild smoothing preserves exploration while imitating baseline wait decisions.
        loss=F.cross_entropy(logits,targets,label_smoothing=.45)
        opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad():return len(x),torch.softmax(policy(x)[0],-1).mean(0).tolist()


def fast_collision_check(tracks, config_path):
    # Import the *real Validator collision simulator*. This requires the same installed
    # Validator dependencies as the official command; never treat failure as PASS.
    from waam_validator.collision.simulator import run_collision_analysis
    from waam_validator.config.loader import load_config
    from waam_validator.models import RobotTrajectory,TrajectorySet
    from waam_validator.constants import MODE_FROM_TEXT
    trajectories=[]
    for rid in (1,2,3):
        rows=tracks[rid]
        trajectories.append(RobotTrajectory(robot_id=rid,
            time_s=np.asarray([a[0] for a in rows],dtype=np.float64),
            xyz_mm=np.asarray([a[1:4] for a in rows],dtype=np.float32),
            mode=np.asarray([MODE_FROM_TEXT[a[4]] for a in rows],dtype=np.uint8)))
    results=run_collision_analysis(TrajectorySet(robots=tuple(trajectories),row_count=sum(map(len,tracks.values()))),load_config(config_path))
    return results.arm_envelope_event_count==0 and results.tcp_radius_event_count==0


def validate_official(source, tracks, target, timeout):
    target.mkdir(parents=True,exist_ok=True)
    for name in ('config.yaml','target.stl'):
        shutil.copy2(source/name,target/name)
    write_tracks(target/'trajectory.csv',tracks)
    cmd=[sys.executable,'-m','waam_validator.cli','run',str(target),'--json']
    try: cp=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout)
    except subprocess.TimeoutExpired:
        write_json(target/'validation_status.json',{'status':'UNKNOWN','error':'timeout'})
        return None
    (target/'validator_stdout.txt').write_text(cp.stdout,encoding='utf-8')
    (target/'validator_stderr.txt').write_text(cp.stderr,encoding='utf-8')
    try: result=json.loads(cp.stdout.strip())
    except ValueError:result={'status':'UNKNOWN','stderr':cp.stderr[-1000:]}
    write_json(target/'validation_status.json',result)
    if cp.returncode!=0 or result.get('status')!='PASS':return None
    return float(result['schedule']['makespan_s'])


def evaluate_move(root,source,tracks,gap,action,max_advance,trial,timeout):
    if action==0:return tracks,0.,'unchanged'
    candidate=compress_gap(tracks,gap,ACTIONS[action],max_advance)
    if candidate is None:return tracks,-.03,'invalid_time'
    if not layer_order_ok(candidate):return tracks,-.04,'layer_order'
    old=real_finish(tracks); new=real_finish(candidate)
    if new>=old-.0001:return tracks,-.01,'no_gain'
    try:
        if not fast_collision_check(candidate,source/'config.yaml'):
            return tracks,-.08,'collision'
    except Exception as ex:
        raise RuntimeError(f'Validator collision precheck unavailable: {ex}') from ex
    # Official test is mandatory for any accepted improvement.
    verified=validate_official(source,candidate,trial,timeout)
    if verified is None:return tracks,-.10,'validator_fail'
    if verified>=old-1e-3:return tracks,-.02,'no_validated_gain'
    return candidate, min(3.,100.*(old-verified)/max(1,old))+.15,'PASS_better'


def run_episode_layer(policy,root,source,reference,model,out,steps,max_advance,timeout,train=True):
    blocks,plan=layer_plan(reference)
    keys=layer_actions(plan)
    chosen_advances={};current=clone(reference);trans=[];notes=[]
    old_signature=deposition_signature(reference)
    for j in range(steps):
        if not keys:break
        key=random.choice(keys)
        x=layer_feature(current,plan,key)
        with torch.no_grad():
            logits,_=policy(torch.tensor(x)[None,:]);dist=Categorical(logits=logits)
            a=dist.sample() if train else logits.argmax(-1)
            action=int(a.item());logp=float(dist.log_prob(a).item())
        if action==0:
            next_state,reward,status=current,0.,'unchanged'
            next_adv=chosen_advances
        else:
            # PPO proposes a cut; the safety shield shrinks it if required.
            # Avoid unsafe 60s overlap proposals masking genuinely safe 1-3s advances.
            proposed=max_advance
            candidate=None;next_adv=chosen_advances;valid_cut=False
            old=real_finish(current)
            attempts=0
            for attempts in range(1,9):
                tmp_adv,tmp=attempt_layer_change(reference,chosen_advances,key,action,proposed,blocks,plan)
                if tmp is None:break
                if real_finish(tmp)>=old-.001:break
                if fast_collision_check(tmp,source/'config.yaml'):
                    candidate=tmp;next_adv=tmp_adv;valid_cut=True;break
                proposed*=.5
                if proposed<.1:break
            if not valid_cut:
                next_state,reward,status=current,-.04,'collision_or_no_gain'
            else:
                verified=validate_official(source,candidate,out/f'probe_{j:03d}',timeout)
                if verified is None:
                    next_state,reward,status=current,-.10,'validator_fail'
                elif verified>=old-.001:
                    next_state,reward,status=current,-.02,'no_validated_gain'
                else:
                    next_state=candidate;reward=min(3.,100.*(old-verified)/max(1,old))+.15
                    status='PASS_better'
        trans.append((x,action,logp,reward))
        if status=='PASS_better':
            current=next_state;chosen_advances=next_adv
        notes.append({'step':j+1,'layer_z':key[0],'robot':key[1], 'action':action,
                      'result':status,'reward':round(float(reward),5),'current_makespan_s':round(real_finish(current),6), 'search_attempts':attempts if action else 0})
    if deposition_signature(current)!=old_signature:raise AssertionError('Depositions changed')
    return current,trans,notes


def ppo_update(policy, optimizer,transitions,epochs=3):
    if not transitions:return None
    obs=torch.tensor(np.stack([x[0] for x in transitions]))
    actions=torch.tensor([x[1] for x in transitions],dtype=torch.long)
    oldlog=torch.tensor([x[2] for x in transitions])
    rewards=torch.tensor([x[3] for x in transitions],dtype=torch.float32)
    with torch.no_grad(): _,oldvalues=policy(obs)
    advantages=rewards-oldvalues
    if len(transitions)>1:advantages=(advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-5)
    returns=rewards
    for _ in range(epochs):
        logits,values=policy(obs);dist=Categorical(logits=logits)
        ratio=(dist.log_prob(actions)-oldlog).exp()
        surrogate=torch.minimum(ratio*advantages,ratio.clamp(.8,1.2)*advantages)
        loss=-surrogate.mean()+.4*F.mse_loss(values,returns)-.015*dist.entropy().mean()
        optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(policy.parameters(),.8);optimizer.step()
    return float(loss.detach())


def run_episode(policy,root,source,base, model,out,steps,max_advance,timeout,train=True):
    current=clone(base);trans=[];notes=[];best=real_finish(current)
    for j in range(steps):
        gaps=get_gaps(current)
        if not gaps:break
        gap=random.choice(gaps)
        x=feature(current,gap)
        with torch.no_grad():
            logits,val=policy(torch.as_tensor(x).unsqueeze(0))
            dist=Categorical(logits=logits)
            a=dist.sample() if train else torch.argmax(logits,dim=-1)
            action=int(a.item()); logp=float(dist.log_prob(a).item())
        attempt=out/f'probe_{j:03d}'
        updated,reward,status=evaluate_move(root,source,current,gap,action,max_advance,attempt,timeout)
        trans.append((x,action,logp,reward))
        if status=='PASS_better':current=updated;best=real_finish(current)
        notes.append({'step':j+1,'robot':gap[0],'gap_index':gap[1], 'action':action,'result':status,
                      'reward':round(float(reward),5),'current_makespan_s':round(best,6)})
    return current,trans,notes


def load_model(root,m):
    src,time=locate_source(root,m)
    tracks=read_tracks(src/'trajectory.csv')
    if abs(max(rr[-1][0] for rr in tracks.values())-time)>1e-3:
        raise RuntimeError(f'Source CSV time mismatch for model {m}')
    return src,time,tracks


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['train','evaluate'])
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--train-models',nargs='+',default=['02','03'])
    p.add_argument('--model',default='06')
    p.add_argument('--trial',default='bc_local_v3_01')
    p.add_argument('--episodes',type=int,default=4)
    p.add_argument('--steps',type=int,default=4)
    p.add_argument('--max-advance',type=float,default=60.)
    p.add_argument('--timeout',type=int,default=2400)
    p.add_argument('--seed',type=int,default=41)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--eval-explore',action='store_true',help='Sample instead of greedy; useful for exploratory tests')
    p.add_argument('--move-type',choices=['layer','wait'],default='layer',help='layer = reschedule same-layer robot blocks; wait = old single-gap experimental mode')
    a=p.parse_args()
    if a.episodes<1 or a.steps<1 or a.max_advance<=0:p.error('Positive episodes, steps, advance required')
    root=a.root.resolve();random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    dest=root/'ppo_project'/'waam_bc_local_v3'/a.trial;dest.mkdir(parents=True,exist_ok=True)
    ckpt=dest/'policy.pt'
    policy=GapPolicy();optimizer=torch.optim.Adam(policy.parameters(),lr=.0003)
    if a.mode=='train':
        if a.model in a.train_models:pass  # model arg unused in train
        if len(set(a.train_models))!=len(a.train_models):p.error('Duplicate train model ID')
        if ckpt.exists() and not a.resume:p.error('Checkpoint exists; choose new --trial or --resume')
        sources={m:load_model(root,m) for m in a.train_models}
        identity={m:sha(src/'trajectory.csv') for m,(src,_,_) in sources.items()}
        if a.resume:
            data=torch.load(ckpt,map_location='cpu',weights_only=True)
            if data['trained_models']!=a.train_models or data['source_sha256']!=identity or data.get('move_type')!=a.move_type:
                p.error('Resume training set/inputs are incompatible')
            policy.load_state_dict(data['policy']);optimizer.load_state_dict(data['optimizer'])
            prior=int(data['episodes_completed']);print(f'RESUMED {prior} episodes',flush=True)
        else:
            prior=0
            n,probs=bc_initialize(policy,[v[2] for v in sources.values()],move_type=a.move_type)
            print(f'BC prepared {n} reference W decisions; initial action probabilities={probs}',flush=True)
        history=[];best={m:tm for m,(s,tm,t) in sources.items()}
        for ep in range(a.episodes):
            m=a.train_models[(prior+ep)%len(a.train_models)]
            src,tm,reference=sources[m]
            accepted,records,notes=(run_episode_layer if a.move_type=="layer" else run_episode)(policy,root,src,reference,m,dest/'train'/f'ep_{prior+ep+1:04d}_{m}',a.steps,a.max_advance,a.timeout,train=True)
            best[m]=min(best[m],real_finish(accepted))
            if any(n['result']=='PASS_better' for n in notes):
                best_path=dest/'best_validated'/m
                best_time_path=best_path/'metadata.json'
                old_best=float(json.loads(best_time_path.read_text())['makespan_s']) if best_time_path.exists() else tm
                if real_finish(accepted)<old_best-1e-3:
                    shutil.copy2(src/'config.yaml',best_path/'config.yaml') if best_path.exists() else None
                    best_path.mkdir(parents=True,exist_ok=True)
                    for name in ('config.yaml','target.stl'):shutil.copy2(src/name,best_path/name)
                    write_tracks(best_path/'trajectory.csv',accepted)
                    write_json(best_time_path,{'model':m,'makespan_s':real_finish(accepted),'status':'PASS', 'origin':'validated episode improvement'})
            loss=ppo_update(policy,optimizer,records)
            row={'episode':prior+ep+1,'model':m,'makespan_s':real_finish(accepted),'baseline_s':tm,
                 'ppo_loss':loss,'attempts':notes}
            history.append(row)
            print(f'[{ep+1}/{a.episodes}] {m}: {real_finish(accepted):.3f}s (PASS fallback={tm:.3f}s), validated_changes={sum(n["result"]=="PASS_better" for n in notes)}, loss={loss:.4f}',flush=True)
            torch.save({'policy':policy.state_dict(),'optimizer':optimizer.state_dict(), 'trained_models':a.train_models,
                'source_sha256':identity,'episodes_completed':prior+ep+1,'feature_version':1,'move_type':a.move_type},ckpt)
        write_json(dest/'training_last_run.json',history)
        print('Policy checkpoint:',ckpt,'\nNo official PASS baseline files modified.',flush=True)
    else:
        if not ckpt.is_file():p.error(f'No trained checkpoint: {ckpt}')
        data=torch.load(ckpt,map_location='cpu',weights_only=True)
        if a.model in data['trained_models']:p.error('Evaluation model is not held out')
        policy.load_state_dict(data['policy']);policy.eval()
        if data.get('move_type')!=a.move_type:p.error('Policy was trained with another --move-type')
        src,baseline,reference=load_model(root,a.model)
        output=dest/'evaluation'/a.model
        output.mkdir(parents=True,exist_ok=True)
        accepted,records,notes=(run_episode_layer if a.move_type=="layer" else run_episode)(policy,root,src,reference,a.model,output,a.steps,a.max_advance,a.timeout,train=a.eval_explore)
        best=real_finish(accepted)
        report={'model':a.model,'held_out':True,'trained_models':data['trained_models'],
            'baseline_official_PASS_s':baseline,'baseline_candidate':str(src),'returned_makespan_s':best,
            'returned_status':'PASS_BASELINE' if best>=baseline-1e-3 else 'PASS_IMPROVED',
            'improvement_s':round(max(0.,baseline-best),6),'steps':notes,
            'caveat':'Fixed deposition geometry and robot assignment; only layer-level start offsets and W intervals change.'}
        if best<baseline-1e-3:
            for name in ('config.yaml','target.stl'):shutil.copy2(src/name,output/name)
            write_tracks(output/'trajectory.csv',accepted)
        write_json(output/'result.json',report)
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
