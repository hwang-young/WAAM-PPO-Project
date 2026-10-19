"""Fresh-input WAAM optimizer. NEVER uses a previously saved trajectory as input.

Each run independently generates toolpaths from target.stl and config.yaml,
validates them with WAAM Validator CLI, optionally searches collision-safe
reordering/reassignment, and retains the fastest *newly* validated PASS.

Scope: 3-robot public WAAM cell supported by the included path generator.
No PPO, no global optimality guarantee, no old PASS cache, no model-ID rules.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

GENERATION_VARIANTS = (
    ("serial_two_half", 1.0, "skeleton", .60, .95, 0.0, "two_halves"),
    ("serial_three_zone", 1.0, "skeleton", .60, .95, 0.0, "three_zone"),
    ("three_zone_infill_90", .8, "skeleton", .50, .94, 90.0, "three_zone"),
    ("two_half_medial", .7, "medial", .50, .95, 0.0, "two_halves"),
    ("three_zone_dense", .6, "skeleton", .45, .90, 45.0, "three_zone"),
)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def fresh_input(input_dir):
    root=Path(input_dir)
    for name in ('target.stl','config.yaml'):
        if not (root/name).is_file():raise FileNotFoundError(str(root/name))
    return {'target_stl_sha256':sha256(root/'target.stl'),
            'config_yaml_sha256':sha256(root/'config.yaml')}


def run_command(command, cwd, log_dir, timeout):
    t=time.monotonic()
    log_dir=Path(log_dir)
    log_dir.mkdir(parents=True,exist_ok=True)
    command=[str(q) for q in command]
    (log_dir/'command.json').write_text(json.dumps(command,ensure_ascii=False,indent=2),encoding='utf-8')
    try:
        p=subprocess.run(command,cwd=str(cwd),capture_output=True,text=True,timeout=max(1,float(timeout)),
            encoding='utf-8',errors='replace')
        out,err=p.stdout,p.stderr
        status='FINISHED' if p.returncode==0 else 'PROCESS_ERROR'
        rc=p.returncode
    except subprocess.TimeoutExpired as e:
        out=e.stdout or b'';err=e.stderr or b''
        if isinstance(out,bytes):out=out.decode('utf-8',errors='replace')
        if isinstance(err,bytes):err=err.decode('utf-8',errors='replace')
        rc=None;status='TIMEOUT'
    (log_dir/'stdout.txt').write_text(out,encoding='utf-8')
    (log_dir/'stderr.txt').write_text(err,encoding='utf-8')
    return {'status':status,'exit_code':rc,'wall_s':round(time.monotonic()-t,6),
            'stderr_tail':err[-350:]}


def official_validate(root,source,logs,timeout):
    record=run_command([sys.executable,'-m','waam_validator.cli','run',str(source),'--json'],root,logs,timeout)
    result=None
    # CLI returns exit code 1 for ordinary Validator FAIL but still prints JSON.
    # Parse even on PROCESS_ERROR to preserve FAIL reasons (never certify PASS).
    if record['status']!='TIMEOUT':
        try:result=json.loads((Path(logs)/'stdout.txt').read_text(encoding='utf-8').strip())
        except (json.JSONDecodeError,ValueError):pass
    record['report_status']=(result or {}).get('status','ERROR')
    record['failure_reasons']=(result or {}).get('failure_reasons',[])
    record['makespan_s']=(result or {}).get('schedule',{}).get('makespan_s')
    record['passed']=(record['status']=='FINISHED' and record['report_status']=='PASS'
                      and isinstance(record['makespan_s'],(float,int))
                      and math.isfinite(float(record['makespan_s'])))
    return record


def command_for_variant(input_dir,out,variant,skip_local=True):
    name,grid,method,simplify,pitch,angle,partition=variant
    cmd=[sys.executable,'-m','waam_centerline.optimize_general_centerline',
         '--model-dir',str(input_dir),'--output',str(out),'--grid',str(grid),
         '--method',method,'--simplify',str(simplify),'--pitch-ratio',str(pitch),
         '--infill-angle',str(angle),'--partition',partition]
    if skip_local:cmd.append('--skip-local-collision')
    return cmd


def choose_fastest(passed):
    # Only candidates generated during this run and officially validated may enter here.
    clean=[p for p in passed if p.get('validation',{}).get('passed')]
    return min(clean,key=lambda p:float(p['validation']['makespan_s'])) if clean else None


def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def solve_one(root,input_dir,label,out,opts):
    start=time.monotonic();out.mkdir(parents=True,exist_ok=False)
    entry={'model':str(label),'input_dir':str(input_dir),'status':'NOT_STARTED',
           'generation_wall_s':0.0,'validator_wall_s':0.0,'refinement_wall_s':0.0,
           'total_wall_s':0.0,'makespan_s':None,'selected_mode':None,
           'selected_candidate':None,'variants':[],'refinement':None,'error':None}
    try:
        entry['input_hashes']=fresh_input(input_dir)
        passed=[]
        for variant in GENERATION_VARIANTS[:opts.variants]:
            name=variant[0]
            dest=out/'generated'/name
            print(f'[{label}] STL → new path: {name}',flush=True)
            g=run_command(command_for_variant(input_dir,dest,variant),root,
                          out/'logs'/name/'generator',opts.generator_timeout_s)
            entry['generation_wall_s']+=g['wall_s']
            rec={'name':name,'source':'GENERATED_FROM_STL_THIS_RUN','candidate':str(dest),
                 'generation':g,'validation':None}
            if g['status']=='FINISHED' and all((dest/n).is_file() for n in ('target.stl','config.yaml','trajectory.csv')):
                # Prevent using anything other than the exact supplied STL+config.
                actual=fresh_input(dest)
                if actual!=entry['input_hashes']:
                    rec['error']='GENERATED_INPUT_HASH_MISMATCH'
                else:
                    check=official_validate(root,dest,out/'logs'/name/'official',opts.validator_timeout_s)
                    entry['validator_wall_s']+=check['wall_s'];rec['validation']=check
                    print(f'[{label}] {name}: {check["report_status"]}, {check["makespan_s"]}s',flush=True)
                    if check['passed']:passed.append(rec)
            else:
                print(f'[{label}] {name}: GENERATION {g["status"]}, {g["stderr_tail"]}',flush=True)
            entry['variants'].append(rec)
            write_json(out/'progress.json',entry)
        best=choose_fastest(passed)
        if best is None:
            entry['status']='NO_NEW_PASS'
            entry['error']='All fresh STL-generated candidates failed or timed out; see logs. Existing PASS files were NOT used.'
            return entry
        entry['status']='PASS'
        selected=Path(best['candidate']);seconds=float(best['validation']['makespan_s'])
        entry['initial_pass_s']=seconds
        entry['selected_method']=best['name']
        # Reassignment is only an OPTIONAL improvement from this run's generated PASS.
        if opts.search_probes>0:
            search_trial=opts.trial+'_search_'+str(label)
            cmd=[sys.executable,'-m','waam_joint_ppo.parallel_assignment_scheduler_v8',
                 '--root',str(root),'--models',str(label),'--trial',search_trial,
                 '--source-dir',str(selected),
                 '--mutation-probes-per-layer',str(opts.search_probes),
                 '--beam',str(opts.beam),'--max-layers',str(opts.search_layers),
                 '--max-beads-per-layer',str(opts.max_beads_per_layer),
                 '--timeout',str(int(opts.validator_timeout_s)),
                 '--seed',str(opts.seed)]
            print(f'[{label}] collision-safe reorder/reassign; max layers={opts.search_layers}',flush=True)
            rr=run_command(cmd,root,out/'logs'/'search',opts.search_timeout_s)
            entry['refinement_wall_s']+=rr['wall_s']
            # Optimizer invokes Validator internally; don't claim pure CPU time.
            sr=root/'ppo_project'/'waam_auto_optimizer_v8'/search_trial/str(label).zfill(2)/'search'/'report.json'
            if rr['status']=='FINISHED' and sr.is_file():
                detail=json.loads(sr.read_text(encoding='utf-8'))
                rr['optimizer_status']=detail.get('status')
                candidate=detail.get('candidate')
                if detail.get('status')=='NEW_PASS_BETTER' and candidate:
                    src=Path(candidate)
                    if all((src/n).is_file() for n in ('target.stl','config.yaml','trajectory.csv')) and fresh_input(src)==entry['input_hashes']:
                        check=official_validate(root,src,out/'logs'/'reverify',opts.validator_timeout_s)
                        entry['validator_wall_s']+=check['wall_s']
                        rr['candidate_revalidation']=check
                        if check['passed'] and float(check['makespan_s'])<seconds-0.01:
                            selected=src;seconds=float(check['makespan_s']);entry['selected_method']='collision_aware_reorder'
            entry['refinement']=rr
        final_dir=out/'selected_pass'
        final_dir.mkdir(parents=True,exist_ok=False)
        for fn in ('target.stl','config.yaml','trajectory.csv'):
            shutil.copy2(selected/fn,final_dir/fn)
        # One more actual final pass, for safe delivery.
        final=official_validate(root,final_dir,out/'logs'/'selected_final',opts.validator_timeout_s)
        entry['validator_wall_s']+=final['wall_s']
        entry['final_validation']=final
        if not final['passed']:
            entry['status']='FINAL_REVALIDATION_FAILED'
            entry['error']='Output selected candidate unexpectedly failed final official validation'
            return entry
        entry['makespan_s']=float(final['makespan_s'])
        entry['selected_candidate']=str(final_dir)
        entry['selected_mode']='PARALLEL' if simultaneous_deposition(final_dir/'trajectory.csv')>0.001 else 'SERIAL'
        entry['simultaneous_deposition_s']=simultaneous_deposition(final_dir/'trajectory.csv')
        entry['status']='PASS'
        return entry
    except Exception as exc:
        entry['status']='ERROR'
        entry['error']=f'{type(exc).__name__}: {exc}'
        return entry
    finally:
        entry['total_wall_s']=round(time.monotonic()-start,6)
        write_json(out/'result.json',entry)


def simultaneous_deposition(csv_path):
    events=[];tracks={}
    with Path(csv_path).open('r',newline='',encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            tracks.setdefault(int(row['robot_id']),[]).append((float(row['time_s']),row['mode']))
    for rows in tracks.values():
        for a,b in zip(rows,rows[1:]):
            if a[1]=='D' and b[0]>a[0]:events.extend(((a[0],1),(b[0],-1)))
    events.sort(key=lambda x:(x[0],x[1]))
    active=0;prev=None;total=0.0
    for t,change in events:
        if prev is not None and active>=2:total+=t-prev
        active+=change;prev=t
    return total


def format_duration(seconds):
    if seconds is None:return 'N/A'
    q=int(round(seconds))
    days,q=divmod(q,86400);hours,q=divmod(q,3600);minutes,secs=divmod(q,60)
    return f'{days:02d}:{hours:02d}:{minutes:02d}:{secs:02d}'


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'),help='WAAM Validator project root')
    src=p.add_mutually_exclusive_group(required=True)
    src.add_argument('--models',nargs='+',help='Models under root/tests, e.g. 01 02 03 04 05 06')
    src.add_argument('--input-dir',type=Path,help='Any folder containing target.stl and config.yaml')
    p.add_argument('--label',help='Unique label for --input-dir')
    p.add_argument('--trial',required=True,help='NEW output trial name. Existing output is never loaded.')
    p.add_argument('--variants',type=int,default=2,help='Independent STL toolpath variants (1..5)')
    p.add_argument('--generator-timeout-s',type=int,default=1800)
    p.add_argument('--validator-timeout-s',type=int,default=1800)
    p.add_argument('--search-timeout-s',type=int,default=300)
    p.add_argument('--search-probes',type=int,default=12,help='0 disables local parallel refinement')
    p.add_argument('--search-layers',type=int,default=2,help='0 explores all layers; never truncates output geometry')
    p.add_argument('--beam',type=int,default=2)
    p.add_argument('--max-beads-per-layer',type=int,default=100)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args(argv)
    if not 1<=a.variants<=len(GENERATION_VARIANTS):p.error('--variants must be 1..5')
    root=a.root.resolve()
    if a.input_dir:
        items=[(a.label or a.input_dir.name,a.input_dir.resolve())]
    else:
        items=[(label,root/'tests'/label) for label in a.models]
    output=root/'ppo_project'/'waam_from_stl_v11'/a.trial
    if output.exists():p.error(f'Output already exists: {output}. Use a new --trial; no prior outputs will be loaded.')
    output.mkdir(parents=True)
    summaries=[]
    for label,path in items:
        print(f'\n===== FRESH STL OPTIMIZATION {label} =====',flush=True)
        rec=solve_one(root,path,label,output/str(label),a)
        summaries.append(rec)
        write_json(output/'summary.json',summaries)
        with (output/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.writer(f)
            w.writerow(['model','status','makespan_s','makespan_DD_HH_MM_SS',
                'total_wall_s','generation_wall_s','validator_wall_s',
                'refinement_wall_s_includes_own_validation','mode','selected_candidate','reason'])
            for row in summaries:
                w.writerow([row.get('model'),row.get('status'),row.get('makespan_s'),
                    format_duration(row.get('makespan_s')),row.get('total_wall_s'),
                    round(row.get('generation_wall_s',0),3),round(row.get('validator_wall_s',0),3),
                    round(row.get('refinement_wall_s',0),3),row.get('selected_mode'),
                    row.get('selected_candidate'),row.get('error')])
        print(f'[{label}] {rec["status"]} {rec.get("makespan_s")}s; total {rec["total_wall_s"]:.2f}s',flush=True)
    print('\n=== NEWLY GENERATED INPUTS ONLY ===',flush=True)
    for r in summaries:print(f'{r["model"]}: {r["status"]}  {format_duration(r.get("makespan_s"))}  wall={r["total_wall_s"]:.2f}s',flush=True)
    print('Report:',output/'summary.csv',flush=True)
    return 0 if all(r['status']=='PASS' for r in summaries) else 2

if __name__=='__main__':sys.exit(main())
