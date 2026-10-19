"""WAAM STL centerline based geometry optimizer + serial 2-robot scheduler.
No original deposition segments required. Official PASS requires WAAM Validator.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
import networkx as nx
import numpy as np
import trimesh
import yaml
from shapely import MultiLineString, Polygon, LineString, contains_xy, get_parts, line_merge, set_precision, union_all
from shapely.ops import split
from skimage.morphology import skeletonize, medial_axis

NEIGHBORS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
COLS=('robot_id','time_s','x_mm','y_mm','z_mm','mode')


def target_layer(mesh, z, snap=1e-4):
    raw = trimesh.intersections.mesh_plane(mesh, plane_origin=[0,0,z], plane_normal=[0,0,1])
    if not len(raw):return Polygon()
    ml=MultiLineString([seg[:,:2] for seg in raw])
    merged=line_merge(set_precision(ml, snap))
    loops=[]
    for p in get_parts(merged):
        if not isinstance(p,LineString) or len(p.coords)<3:continue
        if np.linalg.norm(np.asarray(p.coords[0])-np.asarray(p.coords[-1]))>snap*2:continue
        poly=Polygon(p.coords)
        if not poly.is_empty:loops.append(poly)
    shape=Polygon()
    for poly in sorted(loops,key=lambda q:(-q.area,q.bounds)):
        shape=shape.symmetric_difference(poly)
    return shape


def skeleton_lines(shape, grid, method, simplify_mm=0.65, min_length=1.0):
    x0,y0,x1,y1=shape.bounds
    pad=grid*3
    x0-=pad;y0-=pad;x1+=pad;y1+=pad
    xs=np.arange(x0,x1+grid,grid); ys=np.arange(y0,y1+grid,grid)
    X,Y=np.meshgrid(xs,ys)
    inside=contains_xy(shape,X,Y)
    thin=medial_axis(inside, rng=0) if method=='medial' else skeletonize(inside)
    points=set(map(tuple,np.argwhere(thin)))
    neigh={p:[(p[0]+dr,p[1]+dc) for dr,dc in NEIGHBORS if (p[0]+dr,p[1]+dc) in points] for p in points}
    def canon(a,b):return (a,b) if a<b else (b,a)
    visited=set();sequences=[]
    for p in sorted(points):
        if len(neigh[p])==2:continue
        for q in sorted(neigh[p]):
            e=canon(p,q)
            if e in visited:continue
            visited.add(e);seq=[p,q];prev,cur=p,q
            while len(neigh[cur])==2:
                nxt=next(x for x in neigh[cur] if x!=prev)
                edge=canon(cur,nxt)
                if edge in visited:break
                visited.add(edge);seq.append(nxt);prev,cur=cur,nxt
            sequences.append(seq)
    for p in sorted(points):
        for q in sorted(neigh[p]):
            if canon(p,q) in visited:continue
            visited.add(canon(p,q));seq=[p,q];prev,cur=p,q
            while True:
                opts=[x for x in neigh[cur] if x!=prev and canon(cur,x) not in visited]
                if not opts:break
                nxt=opts[0];visited.add(canon(cur,nxt));seq.append(nxt);prev,cur=cur,nxt
            sequences.append(seq)
    result=[]
    for seq in sequences:
        coords=[(x0+c*grid,y0+r*grid) for r,c in seq]
        line=LineString(coords)
        if line.length>min_length:
            simplified=line.simplify(simplify_mm,preserve_topology=False)
            if simplified.length>.001:result.append(simplified)
    return result


def flatten_geoms(g):
    if isinstance(g,LineString):
        if g.length>1e-5:yield g
    elif hasattr(g,'geoms'):
        for item in g.geoms:
            yield from flatten_geoms(item)


def assign_halves(lines):
    upper=Polygon([(-1000,0),(1000,0),(1000,1000),(-1000,1000)])
    lower=Polygon([(-1000,-1000),(1000,-1000),(1000,0),(-1000,0)])
    out={2:[],3:[]}
    for line in lines:
        plus=list(flatten_geoms(line.intersection(upper)))
        minus=list(flatten_geoms(line.intersection(lower)))
        # Along cut Y=0 use only R2.
        shared=line.intersection(LineString([(-1000,0),(1000,0)]))
        if not shared.is_empty and shared.length>1e-5:
            minus=[q for q in minus if abs(q.interpolate(q.length/2).y)>1e-5]
        out[2].extend(plus);out[3].extend(minus)
    return out


def euler_trails(lines):
    """Edge-disjoint trails using dummy-vertex decomposition per connected component."""
    G=nx.MultiGraph()
    for ix,line in enumerate(lines):
        a=tuple(round(v,4) for v in line.coords[0]);b=tuple(round(v,4) for v in line.coords[-1])
        G.add_edge(a,b,key=ix,coords=list(map(tuple,line.coords)))
    out=[]
    for nodes in nx.connected_components(G):
        H=G.subgraph(nodes).copy()
        odd=[v for v,d in H.degree() if d%2]
        dummy=('dummy',len(out))
        if odd:
            for i,node in enumerate(odd):H.add_edge(dummy,node,key=f'fake-{i}',coords=None)
            start=dummy
        else:start=min(nodes)
        walk=list(nx.eulerian_circuit(H,source=start,keys=True))
        cur=[]
        for a,b,k in walk:
            if a==dummy or b==dummy:
                if len(cur)>1:out.append(cur)
                cur=[];continue
            coords=H.edges[a,b,k]['coords']
            if tuple(round(x,4) for x in coords[0])!=a:coords=list(reversed(coords))
            if cur and np.linalg.norm(np.subtract(cur[-1],coords[0]))>0.001:
                out.append(cur);cur=[]
            cur.extend(coords if not cur else coords[1:])
        if len(cur)>1:out.append(cur)
    return [path for path in out if sum(math.dist(a,b) for a,b in zip(path,path[1:]))>0.01]


def reorder_trails(trails, origin):
    remaining=[list(q) for q in trails]
    arranged=[];pos=origin
    while remaining:
        cand=[]
        for i,p in enumerate(remaining):
            cand.extend([(math.dist(pos,p[0]),i,False),(math.dist(pos,p[-1]),i,True)])
        _,i,reverse=min(cand)
        path=remaining.pop(i)
        if reverse:path.reverse()
        arranged.append(path);pos=path[-1]
    return arranged


class Schedule:
    def __init__(self,config, start_idle=0.01):
        self.robots={int(r['id']):r for r in config['robots']}
        self.points={i:[(0.,tuple(float(v) for v in r['home_xyz_mm']),'W')] for i,r in self.robots.items()}
        self.current_time=float(start_idle)
        self.dtime=0.;self.tlen=0.;self.dlen=0.;self.dcount=0;self.travel_time=0.
        self.dep_speed=float(config['process']['deposition_speed_mm_s'])
        self.speed=float(config['process']['travel_speed_mm_s'])
        self.arc_on=float(config['process'].get('arc_on_time_s',0))
        self.arc_off=float(config['process'].get('arc_off_time_s',0))
        self.base={i:np.asarray(r['base_xyz_mm'][:2],float) for i,r in self.robots.items()}
        self.reach={i:float(r['xy_reach_radius_mm']) for i,r in self.robots.items()}
        self.safepark={2:(300.,400.,20.),3:(300.,-400.,20.)}
    def pos(self,r):return self.points[r][-1][1]
    def wait(self,r,until):
        rows=self.points[r];t,xyz,mode=rows[-1]
        if until>t+1e-8:rows[-1]=(t,xyz,'W');rows.append((until,xyz,'W'))
    def move(self,r,newxyz,mode):
        rows=self.points[r];t,old,oldmode=rows[-1]
        dist=math.dist(old,newxyz)
        if dist<1e-5:return
        v=self.dep_speed if mode=='D' else self.speed
        duration=dist/v
        rows[-1]=(t,old,mode)
        rows.append((t+duration,tuple(float(x) for x in newxyz),'W'))
        if mode=='D':self.dtime+=duration;self.dlen+=dist
        else:self.travel_time+=duration;self.tlen+=dist
    def park(self,r):
        self.move(r,self.safepark[r],'T')
        self.current_time=self.points[r][-1][0]
    def toolpath(self,r,points,z):
        self.wait(r,self.current_time)
        pos=(points[0][0],points[0][1],z)
        if math.dist(self.pos(r),pos) >1e-5:self.move(r,pos,'T')
        self.wait(r,self.points[r][-1][0]+self.arc_on)
        for x,y in points[1:]:self.move(r,(x,y,z),'D')
        self.wait(r,self.points[r][-1][0]+self.arc_off)
        self.dcount+=1
        self.current_time=self.points[r][-1][0]
    def finish(self):
        stop=max(self.points[r][-1][0] for r in self.robots)
        for rid in self.robots:self.wait(rid,stop+0.001)
        for rid,rows in self.points.items():
            for a,b in zip(rows,rows[1:]):
                if b[0]<=a[0]:raise AssertionError((rid,a,b))
        return stop+0.001
    def to_csv(self,path):
        with open(path,'w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(COLS)
            for rid,rows in sorted(self.points.items()):
                for t,p,mode in rows:w.writerow([rid,f'{t:.9f}',*(f'{v:.6f}' for v in p),mode])


def shape_metrics(shapes, paths, cfg):
    br=float(cfg['process']['bead_width_mm'])/2
    sv=cfg['shape_validation'];H=float(cfg['process']['layer_height_mm'])
    ints=[];depvol=targetvol=ivol=overvol=0.;fail=0
    for k,target in sorted(shapes.items()):
        lines=[LineString(q) for robot in paths[k].values() for q in robot]
        deposited=union_all([line.buffer(br,quad_segs=int(sv['polygon_buffer_resolution']),cap_style='round',join_style='round') for line in lines]) if lines else Polygon()
        a=target.area;b=deposited.area;c=target.intersection(deposited).area
        iou=c/(a+b-c) if a+b-c>0 else 1
        coverage=c/a if a else 1
        over=(b-c)/a if a else 0
        ints.append({'layer':k+1,'target_area_mm2':a,'deposited_area_mm2':b,'coverage':coverage,'iou':iou,'overfill':over})
        fail+=int(iou<float(sv['minimum_layer_iou']))
        targetvol+=a*H;depvol+=b*H;ivol+=c*H;overvol+=(b-c)*H
    all_iou=ivol/(targetvol+depvol-ivol)
    all_cov=ivol/targetvol;over=overvol/targetvol
    passed=(all_cov>=float(sv['minimum_overall_coverage']) and over<=float(sv['maximum_overall_overfill_ratio']) and all_iou>=float(sv['minimum_overall_iou']) and fail/len(ints)<=float(sv['maximum_failed_layer_ratio']))
    return {'shape_estimate_pass':passed,'overall_iou':all_iou,'overall_coverage':all_cov,'overall_overfill':over,'failed_layers':fail,'layers':ints}


def reach_check(paths,cfg,margin=.03):
    robots={int(r['id']):r for r in cfg['robots']}
    errs=[]
    for level,assignment in paths.items():
        for rid,items in assignment.items():
            r=robots[rid];base=np.asarray(r['base_xyz_mm'][:2],float);reach=float(r['xy_reach_radius_mm'])
            for p in items:
                a=np.asarray(p,float)
                if np.max(np.linalg.norm(a-base,axis=1))>=reach-margin:
                    errs.append((level,rid,float(np.linalg.norm(a-base,axis=1).max())))
    return errs


def check_serial_collision(schedule, cfg, step=.5):
    """Independent conservative XY capsule test at all trajectory segment interpolation steps.
    This is a local sampling check only, NOT the official Validator.
    """
    # optimize event-based samples: each active segment sampled at <=step seconds.
    bases={i:np.asarray(r['base_xyz_mm'][:2],float) for i,r in schedule.robots.items()}
    radius={i:float(r['arm_envelope_radius_mm']) for i,r in schedule.robots.items()}
    tradius={i:float(r['tcp_radius_mm']) for i,r in schedule.robots.items()}
    collision_margin=float(cfg['collision']['arm_clearance_mm'])
    allrows=schedule.points
    times=sorted({t for rows in allrows.values() for t,_,_ in rows})
    if not times:return {'passed':False,'events':[{'error':'empty'}]}
    # at segment bounds and small enough intermediate slices to discover swept proximity
    import bisect
    tcheck=[]
    for a,b in zip(times,times[1:]):
        n=max(1,math.ceil((b-a)/step))
        tcheck.extend(a+(b-a)*i/n for i in range(n))
    tcheck.append(times[-1])
    tarrays={r:[p[0] for p in rows] for r,rows in allrows.items()}
    errors=[];minarm=float('inf');mintcp=float('inf')
    for time in tcheck:
        xyz={}
        for r in (1,2,3):
            rows=allrows[r];idx=max(0,min(len(rows)-2,bisect.bisect_right(tarrays[r],time)-1))
            st,sp,_=rows[idx];en,ep,_=rows[idx+1]
            t=(time-st)/(en-st) if en>st else 0
            xyz[r]=np.asarray(sp[:2])+max(0.,min(1.,t))*(np.asarray(ep[:2])-np.asarray(sp[:2]))
        for a,b in [(1,2),(1,3),(2,3)]:
            arm=LineString([bases[a],xyz[a]]).distance(LineString([bases[b],xyz[b]]))
            tcp=float(np.linalg.norm(xyz[a]-xyz[b]));minarm=min(minarm,arm-radius[a]-radius[b]-collision_margin);mintcp=min(mintcp,tcp-tradius[a]-tradius[b])
            if arm<=radius[a]+radius[b]+collision_margin+1e-6 or tcp<=tradius[a]+tradius[b]+1e-6:
                errors.append({'t':round(time,4),'robots':(a,b),'arm_centerline_mm':round(arm,3),'tcp_mm':round(tcp,3)})
                if len(errors)>12:return {'passed':False,'events':errors,'min_arm_margin_mm':minarm,'min_tcp_margin_mm':mintcp}
    return {'passed':not errors,'events':errors,'samples':len(tcheck),'min_arm_margin_mm':minarm,'min_tcp_margin_mm':mintcp}


def main():
    ap=argparse.ArgumentParser(description='Model01 collision-separated centerline slicing (candidate, not official PASS)')
    ap.add_argument('--model-dir',default='tests/01')
    ap.add_argument('--output',default='ppo_project/centerline_model01_v1')
    ap.add_argument('--grid',type=float,default=0.5)
    ap.add_argument('--method',choices=('medial','skeleton'),default='medial')
    ap.add_argument('--simplify',type=float,default=0.65)
    ap.add_argument('--collision-step',type=float,default=0.2)
    ap.add_argument('--arc-mode',choices=('validator','physical'),default='validator',help='Validator does not mandate per-bead arc waits; physical adds config arc on/off waits')
    ap.add_argument('--no-collision-check',action='store_true')
    args=ap.parse_args()
    root=Path(args.model_dir);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    cfg=yaml.safe_load((root/'config.yaml').read_text(encoding='utf-8'))
    mesh=trimesh.load(root/'target.stl',process=False)
    n=math.ceil((mesh.bounds[1,2]-float(cfg['process']['build_plane_z_mm'])-1e-7)/float(cfg['process']['layer_height_mm']))
    schedule=Schedule(cfg);allpaths={};shapes={}
    config_arc_on=schedule.arc_on;config_arc_off=schedule.arc_off
    if args.arc_mode=='validator':schedule.arc_on=0.;schedule.arc_off=0.
    for k in range(n):
        z=float(cfg['process']['build_plane_z_mm'])+(k+0.5)*float(cfg['process']['layer_height_mm'])
        shape=target_layer(mesh,z)
        if shape.is_empty:continue
        shapes[k]=shape
        centerlines=skeleton_lines(shape,args.grid,args.method,args.simplify)
        assignments=assign_halves(centerlines)
        allpaths[k]={}
        for rid in [2,3]:
            trails=euler_trails(assignments[rid]);trails=reorder_trails(trails,schedule.pos(rid)[:2]);allpaths[k][rid]=trails
            for p in trails:schedule.toolpath(rid,p,z)
            schedule.wait(rid,schedule.current_time)
            schedule.park(rid)
        print(f'Layer {k+1}: centerline length {sum(sum(math.dist(a,b) for a,b in zip(p,p[1:])) for ps in allpaths[k].values() for p in ps):.2f} mm, R2 trails={len(allpaths[k][2])}, R3 trails={len(allpaths[k][3])}',flush=True)
    makespan=schedule.finish()
    schedule.to_csv(out/'trajectory.csv')
    shutil.copy2(root/'target.stl',out/'target.stl');shutil.copy2(root/'config.yaml',out/'config.yaml')
    geom=shape_metrics(shapes,allpaths,cfg);reach=reach_check(allpaths,cfg)
    collision=({'passed':None,'events':[],'note':'not checked'} if args.no_collision_check else check_serial_collision(schedule,cfg,args.collision_step))
    summary={'model':root.name,'makespan_estimate_s':makespan,'deposition_time_s':schedule.dtime,'deposition_length_mm':schedule.dlen,'travel_time_s':schedule.travel_time,'arc_events':schedule.dcount,'arc_mode':args.arc_mode,'scheduled_arc_wait_s':schedule.dcount*(schedule.arc_on+schedule.arc_off),'config_implied_arc_wait_s':schedule.dcount*(config_arc_on+config_arc_off),'strictly_serial_deposition':True,'geometry':geom,'reach_violations':reach,'collision_sampling':collision,'official_validator_pass':None}
    (out/'candidate_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f"CANDIDATE makespan={makespan:.2f}s, D={schedule.dtime:.2f}s length={schedule.dlen:.2f}mm, T={schedule.travel_time:.2f}s, arcs={schedule.dcount}",flush=True)
    print(f"SHAPE IoU={geom['overall_iou']:.4f} coverage={geom['overall_coverage']:.4f} over={geom['overall_overfill']:.4f}, local shape={geom['shape_estimate_pass']}",flush=True)
    print('REACH failures',len(reach),'XY collision sample',collision.get('passed'),'events',collision.get('events',[])[:2],flush=True)
    print('IMPORTANT: official Validator PASS is NOT asserted. Run: python -m waam_validator run '+str(out),flush=True)
if __name__=='__main__':main()
