"""Model-independent WAAM candidate generation from STL.

Unlike the model-01 demonstration, this offers infill and an R1/R2/R3
spatial partition. The result is always a CANDIDATE until officially validated.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from shapely import LineString, box, union_all
from shapely.affinity import rotate
import trimesh
import yaml
from waam_centerline.optimize_model01_centerline import (
    Schedule, target_layer, skeleton_lines, flatten_geoms, euler_trails,
    reorder_trails, shape_metrics, reach_check, check_serial_collision,
)


def infill_core(shape, half_width, pitch, angle=0):
    """Boundary inset / constant-width internal raster with valid 8-mm footprint."""
    inset = shape.buffer(-half_width, quad_segs=8)
    if inset.is_empty:
        return [], []
    perimeters = list(flatten_geoms(inset.boundary))
    rotated = rotate(inset, -angle, origin='centroid')
    x0, y0, x1, y1 = rotated.bounds
    y = y0 + pitch/2
    scans=[]
    while y < y1:
        cutter=LineString([(x0-10, y),(x1+10,y)])
        for part in flatten_geoms(cutter.intersection(rotated)):
            if part.length>0.5:scans.append(rotate(part,angle,origin='centroid'))
        y+=pitch
    return scans, perimeters


def metric(shape,lines,halfwidth,resolution):
    footprint = union_all([q.buffer(halfwidth,quad_segs=resolution,cap_style='round',join_style='round') for q in lines])
    covered=footprint.intersection(shape).area
    out=footprint.area-covered
    union=shape.area+out
    return {'coverage':covered/shape.area if shape.area else 1.,
            'iou':covered/union if union else 1.,
            'overfill':out/shape.area if shape.area else 0.,
            'length_mm':sum(q.length for q in lines),'pieces':len(lines)}


def choose_paths(shape,cfg,grid,method,simplify,pitch_ratio,angle,include_medial=True):
    bead=float(cfg['process']['bead_width_mm']);half=bead/2
    res=int(cfg['shape_validation']['polygon_buffer_resolution'])
    candidates=[]
    scan,perimeter=infill_core(shape,half,bead*pitch_ratio,angle)
    skeleton=skeleton_lines(shape,grid,method,simplify) if include_medial else []
    variants=[('centerline',skeleton),('infill_perimeter',scan+perimeter),
              ('centerline_infill',skeleton+scan),('centerline_full',skeleton+scan+perimeter)]
    for name,lines in variants:
        if not lines:continue
        stat=metric(shape,lines,half,res)
        stat['strategy']=name
        candidates.append((stat,lines))
    if not candidates:raise RuntimeError('No valid toolpath candidate produced for cross section')
    goal=max(float(cfg['shape_validation']['minimum_layer_iou'])+.02, .915)
    feasible=[(s,l) for s,l in candidates if s['iou']>=goal and s['coverage']>=.90 and s['overfill']<=.29]
    # Arc and inter-bead travel can outweigh small deposition savings.
    pool=feasible or sorted(candidates,key=lambda pair:pair[0]['iou'],reverse=True)[:1]
    chosen,lines=min(pool,key=lambda pair:(pair[0]['length_mm']+bead*1.5*pair[0]['pieces'],pair[0]['length_mm']))
    return lines,chosen,[s for s,_ in candidates]


def split_robots(lines,split_x=-100.,split_y=0.,r1_enabled=True,partition="three_zone"):
    if partition == "two_halves":
        # Geometric region split, not a special case for model 01.
        # Model-independent alternative to explore when serial 3-way travel is costly.
        upper=box(-10000,split_y,10000,10000)
        lower=box(-10000,-10000,10000,split_y)
        out={1:[],2:[],3:[]}
        for q in lines:
            mid=q.interpolate(q.length/2)
            if abs(mid.y-split_y)<1.e-5 and all(abs(pt[1]-split_y)<1.e-5 for pt in q.coords):
                out[2].append(q);continue
            out[2].extend(p for p in flatten_geoms(q.intersection(upper)) if p.length>.05)
            out[3].extend(p for p in flatten_geoms(q.intersection(lower)) if p.length>.05)
        return out
    if partition != "three_zone":raise ValueError(f"Unsupported partition {partition}")
    # Designed for the three fixed base positions in the public benchmark cell.
    # Reach is checked explicitly; different robot cells are NOT automatically supported.
    R1=box(-10000,-10000,split_x,10000)
    R2=box(split_x,-10000,10000,10000).intersection(box(-10000,split_y,10000,10000))
    out={1:[],2:[],3:[]}
    for q in lines:
        remainder=q
        if r1_enabled:
            out[1].extend(p for p in flatten_geoms(q.intersection(R1)) if p.length>.05)
            remainder=q.difference(R1)
        out[2].extend(p for p in flatten_geoms(remainder.intersection(R2)) if p.length>.05)
        out[3].extend(p for p in flatten_geoms(remainder.difference(R2)) if p.length>.05)
    return out


def make_candidate(model_dir,output,*,grid=1.,method='skeleton',simplify=.6,pitch_ratio=.95,
                   infill_angle=0.,collision_step=1.,physical_arcs=False,no_local_collision=False,
                   max_layers=0,make_tasks=False,partition="three_zone"):
    root=Path(model_dir);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    cfg=yaml.safe_load((root/'config.yaml').read_text(encoding='utf-8-sig'))
    # This scheduler uses an explicit spatial partition calibrated to this cell.
    bases={int(r['id']):r['base_xyz_mm'] for r in cfg['robots']}
    reference={1:(-1400.,0.),2:(700.,1212.4356),3:(700.,-1212.4356)}
    if any(r not in bases or math.dist(tuple(bases[r][:2]),reference[r])>5. for r in reference):
        raise ValueError('Unsupported robot bases: spatial partition requires the fixed public three-robot cell')
    mesh=trimesh.load(root/'target.stl',process=False)
    H=float(cfg['process']['layer_height_mm']);z0=float(cfg['process']['build_plane_z_mm'])
    n=math.ceil((float(mesh.bounds[1,2])-z0-1e-7)/H)
    if max_layers and max_layers<n:
        raise ValueError(f'Model has {n} layers; partial output is not a valid candidate. Refusing --max-layers={max_layers}')
    schedule=Schedule(cfg)
    schedule.safepark[1]=tuple(float(x) for x in cfg['robots'][0]['home_xyz_mm'])
    if not physical_arcs:schedule.arc_on=schedule.arc_off=0.
    paths={};shapes={};details=[]
    for k in range(n):
        z=z0+(k+.5)*H
        shape=target_layer(mesh,z)
        if shape.is_empty:continue
        shapes[k]=shape
        lines,chosen,considered=choose_paths(shape,cfg,grid,method,simplify,pitch_ratio,infill_angle,include_medial=(n<=120))
        # For model-01 baseline the existing 2-robot path should be used instead;
        # 3-way spatial partition is a general baseline, not an optimal assignment.
        assignment=split_robots(lines,partition=partition)
        paths[k]={}
        for rid in (1,2,3):
            trails=reorder_trails(euler_trails(assignment[rid]),schedule.pos(rid)[:2])
            paths[k][rid]=trails
            for trail in trails:schedule.toolpath(rid,trail,z)
            if trails:
                schedule.park(rid)
        details.append({'layer':k+1,'z_mm':z,**chosen,'path_counts_by_robot':{str(r):len(paths[k][r]) for r in (1,2,3)},'alternatives':considered})
        if (k+1)%max(1,min(50,n//10))==0:
            print(f'[{root.name}] layer {k+1}/{n}, method={chosen["strategy"]}, iou={chosen["iou"]:.4f}, length={chosen["length_mm"]:.1f}mm',flush=True)
    makespan=schedule.finish()
    schedule.to_csv(out/'trajectory.csv')
    shutil.copy2(root/'config.yaml',out/'config.yaml')
    shutil.copy2(root/'target.stl',out/'target.stl')
    shape=shape_metrics(shapes,paths,cfg)
    reach=reach_check(paths,cfg)
    # Long builds should avoid an unbounded local per-0.25s sampling cost.
    if no_local_collision:
        collision={'passed':None,'reason':'explicitly skipped, official Validator required'}
    else:
        collision=check_serial_collision(schedule,cfg,collision_step)
    summary={'model':root.name,'generator':'general_centerline_v1','status':'UNVALIDATED',
        'makespan_estimate_s':makespan,'deposition_time_s':schedule.dtime,
        'deposition_length_mm':schedule.dlen,'travel_time_s':schedule.travel_time,
        'arc_events':schedule.dcount,'arc_mode':'physical' if physical_arcs else 'validator',
        'strictly_serial_deposition':True,'partition':partition,'geometry':shape,
        'reach_violations':reach,'collision_sampling':collision,
        'config_sha256':hashlib.sha256((root/'config.yaml').read_bytes()).hexdigest(),
        'stl_sha256':hashlib.sha256((root/'target.stl').read_bytes()).hexdigest(),
        'parameterization':{'grid':grid,'method':method,'simplify':simplify,
             'pitch_ratio':pitch_ratio,'infill_angle':infill_angle},
        'layers':details}
    (out/'candidate_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'[{root.name}] CANDIDATE {makespan:.3f}s, deposit {schedule.dtime:.3f}s, '+
          f'IoU={shape["overall_iou"]:.4f}, local reach={len(reach)} errors, '+
          f'collision={collision.get("passed")}. NOT AN OFFICIAL PASS.',flush=True)
    return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-dir',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--grid',type=float,default=1.)
    ap.add_argument('--method',choices=['skeleton','medial'],default='skeleton')
    ap.add_argument('--simplify',type=float,default=.6)
    ap.add_argument('--pitch-ratio',type=float,default=.95)
    ap.add_argument('--infill-angle',type=float,default=0.)
    ap.add_argument('--collision-step',type=float,default=1.)
    ap.add_argument('--physical-arcs',action='store_true')
    ap.add_argument('--partition',choices=['three_zone','two_halves'],default='three_zone')
    ap.add_argument('--skip-local-collision',action='store_true')
    a=ap.parse_args()
    make_candidate(a.model_dir,a.output,grid=a.grid,method=a.method,simplify=a.simplify,
                   pitch_ratio=a.pitch_ratio,infill_angle=a.infill_angle,collision_step=a.collision_step,
                   physical_arcs=a.physical_arcs,no_local_collision=a.skip_local_collision,partition=a.partition)
if __name__=='__main__':main()
