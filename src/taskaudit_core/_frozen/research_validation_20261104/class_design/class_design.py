"""Exact robust selection for additional old task classes after positive completion.

Fixed pair universes must be disjoint unions of complete label-class cliques.
Gamma is 0, 1, or 2. Completion may create as well as destroy distinctions.
H counts refined task classes; full pair coverage F and retained witnesses R are different objectives.
General robust optimization, symmetry breaking and constraint generation are existing
methods. The source is standalone apart from NumPy/SciPy and the standard library.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
import math
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f'{name} must be a nonnegative built-in integer')
    return value


def _union(values):
    out = 0
    for value in values:
        out |= value
    return out


@dataclass
class Instance:
    raw: dict

    def __post_init__(self):
        self.name = self.raw.get('instance_id', 'fixture')
        self.n = len(self.raw['candidates'])
        self.pairs = []
        adjacency = defaultdict(set)
        for pair in self.raw['pairs']:
            if len(pair) != 2 or not all(isinstance(t, str) and t for t in pair):
                raise ValueError('Each pair needs two nonempty string labels')
            a, b = sorted(pair)
            if a == b:
                raise ValueError('Self pairs are not allowed')
            self.pairs.append((a, b))
            adjacency[a].add(b)
            adjacency[b].add(a)
        if len(set(self.pairs)) != len(self.pairs):
            raise ValueError('Duplicate pairs are not allowed')
        self.p = len(self.pairs)
        self.labels = tuple(sorted(adjacency))
        self.components = []
        remaining = set(self.labels)
        while remaining:
            pending, component = [min(remaining)], set()
            while pending:
                t = pending.pop()
                if t not in component:
                    component.add(t)
                    pending.extend(adjacency[t] - component)
            if any(adjacency[t] != component - {t} for t in component):
                raise ValueError('Fixed E0 must be a disjoint union of complete label-class cliques')
            self.components.append(frozenset(component))
            remaining -= component
        self.incident = {label: 0 for label in self.labels}
        for p, (a, b) in enumerate(self.pairs):
            self.incident[a] |= 1 << p
            self.incident[b] |= 1 << p
        self.annotations, self.masks = [], []
        relevant = set()
        for i, candidate in enumerate(self.raw['candidates']):
            labels = candidate['labels']
            if not all(isinstance(t, str) and t for t in labels):
                raise ValueError('Candidate annotations must be nonempty strings')
            obs = frozenset(labels)
            self.annotations.append(obs)
            self.masks.append(sum(1 << p for p, (t, u) in enumerate(self.pairs)
                                  if (t in obs) != (u in obs)))
            for component in self.components:
                present = obs & component
                if present and present != component:
                    relevant.update((i, label) for label in component - present)
        # Preserve supplied cell indices for trace compatibility, but independently
        # verify the cell inventory from the observed candidate annotations.
        if 'cells' in self.raw:
            declared = []
            for cell in self.raw['cells']:
                i = _integer(cell['candidate_index'], 'cell candidate index')
                if i >= self.n or not isinstance(cell['missing_label'], str):
                    raise ValueError('Invalid declared repair cell')
                declared.append((i, cell['missing_label']))
            if len(set(declared)) != len(declared) or set(declared) != relevant:
                raise ValueError('Declared cells disagree with reconstructed mixed-class missing cells')
            self.cells = tuple(declared)
        else:
            self.cells = tuple(sorted(relevant))
        self.cell_candidate = {j: i for j, (i, _) in enumerate(self.cells)}
        self.cell_toggle = {j: self.incident[label] for j, (_, label) in enumerate(self.cells)}
        self.by_candidate_cells = [[] for _ in range(self.n)]
        for j, (i, _) in enumerate(self.cells):
            self.by_candidate_cells[i].append(j)
        self.active = tuple(i for i, mask in enumerate(self.masks) if mask)
        self.pair_attainable_mask = _union(self.masks)
        self.pair_attainable = self.pair_attainable_mask.bit_count()
        self.types = defaultdict(list)
        self.candidate_type = {}
        for i in self.active:
            # Full observed pattern on E0 labels, not only original witness bundles.
            kind = tuple(label in self.annotations[i] for label in self.labels)
            self.types[kind].append(i)
            self.candidate_type[i] = kind

        self.classes = tuple(tuple(sorted(group)) for group in self.components)
        pair_index = {pair: p for p, pair in enumerate(self.pairs)}
        self.representatives = tuple(
            tuple(pair_index[tuple(sorted((label, previous)))] for previous in group[:j])
            for group in self.classes for j, label in enumerate(group) if j)
        self.class_ceiling = len(self.representatives)
        self.attainable = self.class_gain(self.active)

    def validate_selected(self, selected):
        selected = list(selected)
        if any(type(i) is not int or i < 0 or i >= self.n for i in selected):
            raise ValueError('Invalid selected candidate index')
        if len(set(selected)) != len(selected):
            raise ValueError('Selected candidates must be unique')
        return sorted(selected)

    def nominal(self, selected):
        return self.class_gain(selected)

    def class_gain(self, selected, attack=()):
        selected = self.validate_selected(selected)
        positions = {i: k for k, i in enumerate(selected)}
        codes = {t: sum(int(t in self.annotations[i]) << k for k, i in enumerate(selected))
                 for t in self.labels}
        if len(attack) != len(set(attack)):
            raise ValueError('Repeated repair cell')
        for cell in attack:
            if type(cell) is not int or cell not in self.cell_candidate:
                raise ValueError('Unknown repair cell')
            i, label = self.cells[cell]
            if i in positions:
                codes[label] |= 1 << positions[i]
        return sum(len({codes[t] for t in group}) - 1 for group in self.classes)

    def pair_nominal(self, selected):
        return _union(self.masks[i] for i in selected).bit_count()

    def completed_masks(self, attack):
        attack = tuple(attack)
        if len(set(attack)) != len(attack):
            raise ValueError('Repeated repair cell')
        masks = self.masks.copy()
        for cell in attack:
            if type(cell) is not int or cell not in self.cell_candidate:
                raise ValueError('Unknown repair cell')
            masks[self.cell_candidate[cell]] ^= self.cell_toggle[cell]
        return masks

    def pair_coverage(self, selected, attack=()):
        masks = self.completed_masks(attack)
        return _union(masks[i] for i in selected).bit_count()

    def kernel(self, budget, gamma):
        cap = min(budget, gamma + 1)
        return sorted(i for group in self.types.values() for i in group[:cap])

    def canonicalize(self, selected, pool):
        selected = self.validate_selected(selected)
        multiplicities = Counter(self.candidate_type[i] for i in selected if i in self.candidate_type)
        poolset = set(pool)
        out = []
        for kind, count in multiplicities.items():
            available = [i for i in self.types[kind] if i in poolset]
            out.extend(available[:count])
        return sorted(out)


def oracle(instance, selected, gamma):
    """Exact H minimum from raw selected-label bitcodes, Gamma in {0,1,2}."""
    gamma = _integer(gamma, 'gamma')
    if gamma > 2:
        raise ValueError('H oracle supports Gamma in {0,1,2}')
    selected = instance.validate_selected(selected)
    positions = {i: k for k, i in enumerate(selected)}
    codes = {t: sum(int(t in instance.annotations[i]) << k for k, i in enumerate(selected))
             for t in instance.labels}
    cells = sorted(j for i in selected for j in instance.by_candidate_cells[i])
    def evaluate(attack):
        updated = codes.copy()
        for cell in attack:
            i, label = instance.cells[cell]
            updated[label] |= 1 << positions[i]
        return sum(len({updated[t] for t in group}) - 1 for group in instance.classes)
    nominal = evaluate(())
    best, best_attack, examined = nominal, (), 1
    candidate_total = 1 + (len(cells) if gamma else 0)
    if gamma == 2:
        candidate_total += len(cells)*(len(cells)-1)//2
    for count in range(1, gamma+1):
        if not best:
            break
        for attack in combinations(cells, count):
            value = evaluate(attack)
            examined += 1
            if value < best:
                best, best_attack = value, attack
                if not best:
                    break
    if instance.class_gain(selected, best_attack) != best:
        raise AssertionError('H attack replay mismatch')
    return {'value': best, 'nominal': nominal, 'attack': list(best_attack),
            'exact': True, 'attacks_examined': examined,
            'relevant_attack_candidates': candidate_total,
            'objective_scope': 'Additional old eligible task classes after candidate positive completion.'}


def greedy(instance, budget, gamma, kind='robust'):
    """Heuristic only: H lacks the nominal coverage submodularity guarantee."""
    _integer(budget, 'budget')
    _integer(gamma, 'gamma')
    if gamma > 2 or kind not in ('robust', 'nominal'):
        raise ValueError('Gamma must be 0/1/2; kind robust or nominal')
    selected, remaining = [], list(instance.active)
    while remaining and len(selected) < budget:
        def key(i):
            candidate = selected + [i]
            nominal = instance.nominal(candidate)
            value = oracle(instance, candidate, gamma)['value'] if kind == 'robust' else nominal
            return value, nominal, -i
        chosen = max(remaining, key=key)
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def solve_master(instance, pool, budget, scenarios, seconds, symmetry=True):
    """Canonical representative variables count completed-codeword classes."""
    n, q = len(pool), instance.class_ceiling
    if not scenarios or tuple(scenarios[0]) != ():
        raise ValueError('Scenario zero must be the empty completion')
    if not n or not q or not budget:
        return {'selected': [], 'upper': 0, 'status': 0, 'success': True,
                'objective_upper': 0., 'seconds': 0., 'variables': 0,
                'constraints': 0, 'nonzeros': 0, 'representatives_per_scenario': q}
    eta = n + len(scenarios)*q
    nv = eta+1
    rr, cc, vv, ub = [], [], [], []
    def row(entries, upper):
        r = len(ub)
        for col, value in entries:
            rr.append(r); cc.append(col); vv.append(value)
        ub.append(upper)
    row([(j, 1.) for j in range(n)], budget)
    if symmetry:
        positions = {i: j for j, i in enumerate(pool)}
        for group in instance.types.values():
            present = [positions[i] for i in group if i in positions]
            for a, b in zip(present, present[1:]):
                row([(b, 1.), (a, -1.)], 0.)
    for s, attack in enumerate(scenarios):
        masks = instance.completed_masks(attack)
        for representative, earlier_pairs in enumerate(instance.representatives):
            r = n+s*q+representative
            # A label represents its codeword iff separated from every earlier label.
            for pair in earlier_pairs:
                entries = [(r, 1.)]
                entries.extend((j, -1.) for j, i in enumerate(pool) if masks[i] & (1 << pair))
                row(entries, 0.)
        row([(eta, 1.)]+[(n+s*q+r, -1.) for r in range(q)], 0.)
    matrix = coo_matrix((np.asarray(vv), (np.asarray(rr, dtype=np.int32),
                                       np.asarray(cc, dtype=np.int32))), shape=(len(ub), nv)).tocsc()
    objective = np.zeros(nv)
    objective[eta] = -(q+1)
    objective[n:n+q] = -1.
    low, high = np.zeros(nv), np.ones(nv)
    high[eta] = instance.attainable
    integrality = np.zeros(nv); integrality[:n] = 1
    started = time.perf_counter()
    result = milp(objective, integrality=integrality, bounds=Bounds(low, high),
                  constraints=LinearConstraint(matrix, np.full(len(ub), -np.inf), np.asarray(ub)),
                  options={'time_limit': max(.01, seconds), 'mip_rel_gap': 0., 'presolve': True})
    selected = [] if result.x is None else [pool[j] for j in range(n) if result.x[j] > .5]
    if len(selected) > budget:
        raise AssertionError('Master returned infeasible selection')
    dual = getattr(result, 'mip_dual_bound', None)
    scalar_upper = -float(dual) if dual is not None and np.isfinite(dual) else None
    upper = min(instance.attainable, math.floor((scalar_upper+1e-5)/(q+1))) if scalar_upper is not None else instance.attainable
    return {'selected': selected, 'upper': upper, 'status': int(result.status),
            'success': bool(result.success), 'objective_upper': scalar_upper,
            'seconds': time.perf_counter()-started, 'message': str(result.message),
            'variables': nv, 'constraints': len(ub), 'nonzeros': len(vv),
            'representatives_per_scenario': q, 'fixed_pairs_used': instance.p}


def exact_design(instance, budget, gamma, reduction='kernel', seconds=20,
                 max_scenarios=160, seeds=None, symmetry=True):
    """CCG for robust additional task classes; primary H, secondary nominal H."""
    started = time.perf_counter()
    _integer(budget, 'budget'); _integer(gamma, 'gamma')
    _integer(max_scenarios, 'max_scenarios')
    if gamma > 2 or reduction not in ('kernel', 'none'):
        raise ValueError('Gamma must be 0/1/2 and reduction kernel/none')
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('seconds must be finite and nonnegative')
    pool = list(instance.active) if reduction == 'none' else instance.kernel(budget, gamma)
    candidates = []
    for seed in seeds or []:
        seed = instance.validate_selected(seed)
        if len(seed) <= budget:
            candidates.append(instance.canonicalize(seed, pool))
    candidates.append(instance.canonicalize(pool[:budget], pool))
    best, best_o = [], oracle(instance, [], gamma)
    scenarios = [()]
    for selected in candidates:
        value = oracle(instance, selected, gamma)
        if (value['value'], value['nominal'], -len(selected)) > (best_o['value'], best_o['nominal'], -len(best)):
            best, best_o = selected, value
        attack = tuple(value['attack'])
        if attack not in scenarios:
            scenarios.append(attack)
    upper = instance.attainable if pool and budget else 0
    traces = []
    while upper > best_o['value'] and len(scenarios) <= max_scenarios:
        remaining = seconds-(time.perf_counter()-started)
        if remaining <= 0:
            break
        master = solve_master(instance, pool, budget, scenarios, remaining, symmetry=symmetry)
        upper = min(upper, master['upper'])
        selected = master['selected']
        value = oracle(instance, selected, gamma)
        if (value['value'], value['nominal'], -len(selected)) > (best_o['value'], best_o['nominal'], -len(best)):
            best, best_o = selected, value
        traces.append({'scenarios': len(scenarios), 'master': master,
                       'separation': value, 'best_lower': best_o['value'], 'upper': upper})
        if upper < best_o['value']:
            raise AssertionError('H upper bound below feasible robust value')
        if upper == best_o['value']:
            break
        attack = tuple(value['attack'])
        if attack in scenarios:
            break
        scenarios.append(attack)
    return {'selected': best, 'value': best_o['value'], 'nominal': best_o['nominal'],
            'attack': best_o['attack'], 'upper': upper, 'certified': upper == best_o['value'],
            'seconds': time.perf_counter()-started, 'pool_size': len(pool),
            'active_candidates': len(instance.active), 'pattern_types': len(instance.types),
            'scenarios': len(scenarios), 'reduction': reduction, 'symmetry': symmetry,
            'traces': traces, 'class_gain_ceiling': instance.class_ceiling,
            'secondary_nominal_optimality_certified': False,
            'objective_scope': 'Worst-case additional task classes within fixed original eligible classes; not independent biology.'}
