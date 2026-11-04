"""Separate H optimization under an all-completions F floor.

Both an exact H separator and an exact F separator are mandatory. This module does
not modify either frozen scientific core. A finite-scenario F floor is insufficient
for accepting a feasible H incumbent.
"""
from dataclasses import dataclass
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def _load(name,path,expected):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise RuntimeError('Frozen dependency drift: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module

H_CORE=_load('floor_h_core',HERE/'class_design.py','0561248831fc24f67e4647b503a561b0d29cc273fbcf1591123ab42d592cf4cc')
F_CORE=_load('floor_f_core',ROOT/'research_method_20261104/full_design.py','8d1e6189e27911fc73856c8892a0cdfa079a1b234f12adb8b79ec91d2432678f')

def _integer(value,name):
    if type(value) is not int or value<0:raise ValueError(name+' must be a nonnegative integer')
    return value

@dataclass
class Instance:
    raw:dict
    def __post_init__(self):
        self.h=H_CORE.Instance(self.raw)
        self.f=F_CORE.Instance(self.raw)
        if self.h.cells!=self.f.cells or self.h.pairs!=self.f.pairs:raise AssertionError('H and F domains differ')
        if self.h.active!=self.f.active:raise AssertionError('H and F active domains differ')

def evaluate(instance,selected,gamma,f_floor):
    h=H_CORE.oracle(instance.h,selected,gamma)
    f=F_CORE.oracle(instance.f,selected,gamma)
    return dict(selected=instance.h.validate_selected(selected),H=h['value'],H0=h['nominal'],
                F=f['value'],F0=f['nominal'],h_attack=h['attack'],f_attack=f['attack'],
                f_floor_feasible=f['value']>=f_floor)

def solve_master(instance,pool,budget,gamma,f_floor,scenarios,seconds,symmetry=True):
    h=instance.h;n=len(pool);q=h.class_ceiling;p=h.p
    if not scenarios or tuple(scenarios[0])!=():raise ValueError('Empty completion must be first')
    for attack in scenarios:
        if len(set(attack))!=len(attack) or len(attack)>gamma:raise ValueError('Illegal scenario size')
        h.completed_masks(attack)
    if not n or not q or not budget:
        feasible=f_floor==0
        return dict(selected=[] if feasible else None,upper=0 if feasible else None,
                    infeasible=not feasible,status=0 if feasible else 2,success=True,
                    objective_upper=0. if feasible else None,seconds=0.,variables=0,constraints=0)
    # Each scenario has q canonical H representatives and p full XOR cover variables.
    width=q+p;eta=n+len(scenarios)*width;nv=eta+1
    rr=[];cc=[];vv=[];ub=[]
    def row(entries,upper):
        r=len(ub)
        for col,val in entries:rr.append(r);cc.append(col);vv.append(val)
        ub.append(upper)
    row([(j,1.) for j in range(n)],budget)
    if symmetry:
        positions={i:j for j,i in enumerate(pool)}
        for group in h.types.values():
            present=[positions[i] for i in group if i in positions]
            for a,b in zip(present,present[1:]):row([(b,1.),(a,-1.)],0.)
    for s,attack in enumerate(scenarios):
        offset=n+s*width;masks=h.completed_masks(attack)
        for j,earlier in enumerate(h.representatives):
            for pair in earlier:
                row([(offset+j,1.)]+[(k,-1.) for k,i in enumerate(pool) if masks[i]&(1<<pair)],0.)
        row([(eta,1.)]+[(offset+j,-1.) for j in range(q)],0.)
        for pair in range(p):
            row([(offset+q+pair,1.)]+[(k,-1.) for k,i in enumerate(pool) if masks[i]&(1<<pair)],0.)
        row([(offset+q+pair,-1.) for pair in range(p)],-float(f_floor))
    matrix=coo_matrix((np.asarray(vv),(np.asarray(rr,dtype=np.int32),np.asarray(cc,dtype=np.int32))),shape=(len(ub),nv)).tocsc()
    objective=np.zeros(nv);objective[eta]=-(q+1);objective[n:n+q]=-1.
    low=np.zeros(nv);high=np.ones(nv);high[eta]=h.attainable
    integrality=np.zeros(nv);integrality[:n]=1
    start=time.perf_counter()
    result=milp(objective,integrality=integrality,bounds=Bounds(low,high),
                constraints=LinearConstraint(matrix,np.full(len(ub),-np.inf),np.asarray(ub)),
                options={'time_limit':max(.01,seconds),'mip_rel_gap':0.,'presolve':True})
    elapsed=time.perf_counter()-start
    if result.status==2:
        return dict(selected=None,upper=None,infeasible=True,status=2,success=True,
                    objective_upper=None,seconds=elapsed,variables=nv,constraints=len(ub),message=str(result.message))
    dual=getattr(result,'mip_dual_bound',None)
    scalar_upper=-float(dual) if dual is not None and np.isfinite(dual) else None
    upper=min(h.attainable,max(0,math.floor((scalar_upper+1e-5)/(q+1)))) if scalar_upper is not None else h.attainable
    selected=None
    if result.x is not None:
        require_integral=all(abs(float(x)-round(float(x)))<1e-4 for x in result.x[:n])
        proposed=[pool[j] for j in range(n) if result.x[j]>.5]
        finite_feasible=len(proposed)<=budget and all(instance.f.full(proposed,attack)>=f_floor for attack in scenarios)
        if require_integral and finite_feasible:selected=proposed
    return dict(selected=selected,upper=upper,infeasible=False,status=int(result.status),
                success=bool(result.success),objective_upper=scalar_upper,seconds=elapsed,
                variables=nv,constraints=len(ub),message=str(result.message))

def exact_design(instance,budget,gamma,f_floor,reduction='kernel',seconds=20,max_scenarios=256,seeds=None,symmetry=True):
    started=time.perf_counter()
    for value,name in [(budget,'budget'),(gamma,'gamma'),(f_floor,'f_floor'),(max_scenarios,'max_scenarios')]:_integer(value,name)
    if gamma>2 or reduction not in ('kernel','none'):raise ValueError('Gamma must be 0/1/2 and reduction kernel/none')
    if not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or seconds<0:raise ValueError('seconds must be finite and nonnegative')
    h=instance.h
    pool=list(h.active) if reduction=='none' else h.kernel(budget,gamma)
    if pool!= (list(instance.f.active) if reduction=='none' else instance.f.kernel(budget,gamma)):
        raise AssertionError('The two objective kernels disagree')
    best=None;upper=h.attainable if pool and budget else 0;traces=[];scenarios=[()];infeasible=False;reason=None
    h_cuts=f_cuts=0
    if f_floor>instance.f.attainable or ((not pool or not budget) and f_floor>0):
        infeasible=True;reason='F floor exceeds a certified nominal attainable/budget-zero ceiling'
    candidates=[]
    for selected in seeds or []:
        selected=h.validate_selected(selected)
        if len(selected)<=budget:candidates.append(h.canonicalize(selected,pool))
    candidates.extend([[],h.canonicalize(pool[:budget],pool)])
    def consider(out):
        nonlocal best
        if out['f_floor_feasible'] and (best is None or (out['H'],out['H0'],-len(out['selected']))>(best['H'],best['H0'],-len(best['selected']))):
            best=out
    def separate(out):
        nonlocal h_cuts,f_cuts
        added=False
        # F-minimizing attacks are added independently of the H-minimizing attack.
        if not out['f_floor_feasible']:
            attack=tuple(out['f_attack'])
            if attack not in scenarios:scenarios.append(attack);f_cuts+=1;added=True
        attack=tuple(out['h_attack'])
        if attack not in scenarios:scenarios.append(attack);h_cuts+=1;added=True
        return added
    if not infeasible:
        for selected in candidates:
            out=evaluate(instance,selected,gamma,f_floor);consider(out);separate(out)
        while best is None or upper>best['H']:
            remaining=seconds-(time.perf_counter()-started)
            if remaining<=0 or len(scenarios)>max_scenarios:break
            master=solve_master(instance,pool,budget,gamma,f_floor,scenarios,remaining,symmetry=symmetry)
            if master['infeasible']:
                if best is not None:raise AssertionError('Finite master contradicts an all-completions feasible incumbent')
                infeasible=True;reason='Finite legal-scenario master is certified infeasible'
                traces.append(dict(master=master,scenarios=len(scenarios),best_lower=None,upper=None));break
            upper=min(upper,master['upper'])
            if master['selected'] is None:
                traces.append(dict(master=master,scenarios=len(scenarios),best_lower=None if best is None else best['H'],upper=upper));break
            out=evaluate(instance,master['selected'],gamma,f_floor);consider(out)
            traces.append(dict(master=master,separation=out,scenarios=len(scenarios),best_lower=None if best is None else best['H'],upper=upper))
            if best is not None and upper<best['H']:raise AssertionError('Upper below F-feasible H incumbent')
            if best is not None and upper==best['H']:break
            added=separate(out)
            if not out['f_floor_feasible'] and not added:
                # A master incumbent cannot violate a floor scenario it already contains.
                raise AssertionError('Unseparated F floor violation')
            if not added:break
    certified=not infeasible and best is not None and upper==best['H']
    return dict(status='infeasible' if infeasible else ('optimal' if certified else 'bounded'),
        selected=None if best is None else best['selected'],value=None if best is None else best['H'],
        nominal=None if best is None else best['H0'],F_value=None if best is None else best['F'],F_nominal=None if best is None else best['F0'],
        h_attack=None if best is None else best['h_attack'],f_attack=None if best is None else best['f_attack'],
        upper=None if infeasible else upper,certified=certified,infeasibility_certified=infeasible,infeasibility_reason=reason,
        feasible_incumbent=best is not None,f_floor=f_floor,budget=budget,gamma=gamma,pool_size=len(pool),
        h_cuts=h_cuts,f_cuts=f_cuts,scenarios=len(scenarios),traces=traces,seconds=time.perf_counter()-started,
        symmetry=symmetry,reduction=reduction,secondary_nominal_optimality_certified=False,
        objective_scope='H optimum subject to min_D F>=f_floor on the same fixed classes and positive-cell uncertainty domain.')
