"""Exact robust selection for full XOR coverage after positive annotation completion.

Fixed pair universes must be disjoint unions of complete label-class cliques.
Gamma is 0, 1, or 2. Completion may create as well as destroy distinctions.
This is a separate scientific objective from retained-observed-witness robustness.
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
        self.attainable_mask = _union(self.masks)
        self.attainable = self.attainable_mask.bit_count()
        self.types = defaultdict(list)
        self.candidate_type = {}
        for i in self.active:
            # Full observed pattern on E0 labels, not only original witness bundles.
            kind = tuple(label in self.annotations[i] for label in self.labels)
            self.types[kind].append(i)
            self.candidate_type[i] = kind

    def validate_selected(self, selected):
        selected = list(selected)
        if any(type(i) is not int or i < 0 or i >= self.n for i in selected):
            raise ValueError('Invalid selected candidate index')
        if len(set(selected)) != len(selected):
            raise ValueError('Selected candidates must be unique')
        return sorted(selected)

    def nominal(self, selected):
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

    def full(self, selected, attack=()):
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
    """Exact full-completion minimum; cached row masks avoid rebuilding annotations."""
    gamma = _integer(gamma, 'gamma')
    if gamma > 2:
        raise ValueError('The exact full-completion oracle supports Gamma in {0,1,2}')
    selected = instance.validate_selected(selected)
    nominal = instance.nominal(selected)
    cells = sorted(cell for i in selected for cell in instance.by_candidate_cells[i])
    best, attack, examined = nominal, (), 1
    candidate_total = 1 + (len(cells) if gamma >= 1 else 0)
    if gamma == 2:
        candidate_total += len(cells) * (len(cells) - 1) // 2
    others_one = {i: _union(instance.masks[k] for k in selected if k != i) for i in selected}
    others_two = {(i, j): _union(instance.masks[k] for k in selected if k not in (i, j))
                  for i, j in combinations(selected, 2)}
    if gamma >= 1 and best:
        for cell in cells:
            i = instance.cell_candidate[cell]
            value = (others_one[i] | (instance.masks[i] ^ instance.cell_toggle[cell])).bit_count()
            examined += 1
            if value < best:
                best, attack = value, (cell,)
                if best == 0:
                    break
    if gamma == 2 and best:
        for a, b in combinations(cells, 2):
            i, j = instance.cell_candidate[a], instance.cell_candidate[b]
            if i == j:
                value = (others_one[i] | (instance.masks[i] ^ instance.cell_toggle[a]
                                         ^ instance.cell_toggle[b])).bit_count()
            else:
                other = others_two[min(i, j), max(i, j)]
                value = (other | (instance.masks[i] ^ instance.cell_toggle[a])
                         | (instance.masks[j] ^ instance.cell_toggle[b])).bit_count()
            examined += 1
            if value < best:
                best, attack = value, (a, b)
                if best == 0:
                    break
    if instance.full(selected, attack) != best:
        raise AssertionError('Full-completion attack replay mismatch')
    return {'value': best, 'attack': list(attack), 'nominal': nominal,
            'lost': nominal - best, 'exact': True, 'attacks_examined': examined,
            'relevant_attack_candidates': candidate_total,
            'objective_scope': 'Full fixed-E0 XOR coverage after positive completion.'}


def greedy(instance, budget, gamma, kind='full'):
    _integer(budget, 'budget')
    if kind not in ('full', 'nominal'):
        raise ValueError('Supported greedy kinds: full or nominal')
    selected, remaining = [], list(instance.active)
    while remaining and len(selected) < budget:
        def key(i):
            candidate = selected + [i]
            nominal = instance.nominal(candidate)
            value = oracle(instance, candidate, gamma)['value'] if kind == 'full' else nominal
            return value, nominal, -i
        chosen = max(remaining, key=key)
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def solve_master(instance, pool, budget, scenarios, seconds, symmetry=True, nominal_floor=0):
    """Finite scenarios use full completed masks, including originally unattainable pairs."""
    n, q = len(pool), instance.p
    if not scenarios or tuple(scenarios[0]) != ():
        raise ValueError('Scenario zero must be the empty completion')
    if not n or not q:
        if nominal_floor:
            raise ValueError('Positive nominal floor infeasible for an empty master')
        return {'selected': [], 'upper': 0, 'status': 0, 'success': True,
                'objective_upper': 0.0, 'seconds': 0.0}
    eta = n + len(scenarios) * q
    nv = eta + 1
    rr, cc, vv, ub = [], [], [], []

    def row(entries, upper):
        r = len(ub)
        for c, v in entries:
            rr.append(r)
            cc.append(c)
            vv.append(v)
        ub.append(upper)

    row([(j, 1.0) for j in range(n)], budget)
    if symmetry:
        positions = {i: j for j, i in enumerate(pool)}
        for group in instance.types.values():
            present = [positions[i] for i in group if i in positions]
            for earlier, later in zip(present, present[1:]):
                row([(later, 1.0), (earlier, -1.0)], 0.0)
    if nominal_floor:
        row([(n + k, -1.0) for k in range(q)], -float(nominal_floor))
    for s, attack in enumerate(scenarios):
        masks = instance.completed_masks(attack)
        for p in range(q):
            entries = [(n + s*q + p, 1.0)]
            entries.extend((j, -1.0) for j, i in enumerate(pool) if masks[i] & (1 << p))
            row(entries, 0.0)
        row([(eta, 1.0)] + [(n + s*q + p, -1.0) for p in range(q)], 0.0)
    matrix = coo_matrix((np.asarray(vv), (np.asarray(rr, dtype=np.int32),
                                       np.asarray(cc, dtype=np.int32))), shape=(len(ub), nv)).tocsc()
    objective = np.zeros(nv)
    objective[eta] = -(q + 1)
    objective[n:n+q] = -1.0
    low, high = np.zeros(nv), np.ones(nv)
    high[eta] = instance.attainable
    integrality = np.zeros(nv)
    integrality[:n] = 1
    started = time.perf_counter()
    result = milp(objective, integrality=integrality, bounds=Bounds(low, high),
                  constraints=LinearConstraint(matrix, np.full(len(ub), -np.inf), np.asarray(ub)),
                  options={'time_limit': max(0.01, seconds), 'mip_rel_gap': 0.0, 'presolve': True})
    selected = []
    if result.x is not None:
        selected = [pool[j] for j in range(n) if result.x[j] > 0.5]
        if len(selected) > budget or instance.nominal(selected) < nominal_floor:
            raise AssertionError('Master returned an infeasible incumbent')
    dual = getattr(result, 'mip_dual_bound', None)
    upper_score = -float(dual) if dual is not None and np.isfinite(dual) else None
    upper = (min(instance.attainable, math.floor((upper_score + 1e-5) / (q+1)))
             if upper_score is not None else instance.attainable)
    return {'selected': selected, 'upper': upper, 'status': int(result.status),
            'success': bool(result.success), 'objective_upper': upper_score,
            'seconds': time.perf_counter() - started, 'message': str(result.message),
            'variables': nv, 'constraints': len(ub), 'nonzeros': len(vv),
            'all_fixed_pairs_in_master': q,
            'originally_unattainable_pairs_retained': q - instance.attainable}


def exact_design(instance, budget, gamma, reduction='kernel', seconds=20,
                 max_scenarios=160, seeds=None, symmetry=True, nominal_floor=0):
    """Certified full-XOR robust design; timeout results retain a valid primary interval."""
    started = time.perf_counter()
    _integer(budget, 'budget')
    _integer(nominal_floor, 'nominal_floor')
    _integer(gamma, 'gamma')
    if gamma > 2 or reduction not in ('kernel', 'none'):
        raise ValueError('Gamma must be 0/1/2 and reduction kernel/none')
    pool = list(instance.active) if reduction == 'none' else instance.kernel(budget, gamma)
    candidates = []
    for seed in seeds or []:
        seed = instance.validate_selected(seed)
        if len(seed) <= budget:
            # Mapping real IDs to pattern-equivalent kept copies preserves the full
            # robust objective, nominal coverage and at-most-budget feasibility.
            candidates.append(instance.canonicalize(seed, pool))
    candidates.append(instance.canonicalize(pool[:budget], pool))
    candidates = [s for s in candidates if instance.nominal(s) >= nominal_floor]
    best, best_o = [], oracle(instance, [], gamma)
    scenarios = [()]
    for selected in candidates:
        value = oracle(instance, selected, gamma)
        if (value['value'], value['nominal'], -len(selected)) > (
                best_o['value'], best_o['nominal'], -len(best)):
            best, best_o = selected, value
        attack = tuple(value['attack'])
        if attack not in scenarios:
            scenarios.append(attack)
    if best_o['nominal'] < nominal_floor:
        raise ValueError('Nominal floor requires a verified feasible supplied or automatic initial selection')
    upper = instance.attainable
    traces = []
    while upper > best_o['value'] and len(scenarios) <= max_scenarios:
        remaining = seconds - (time.perf_counter() - started)
        if remaining <= 0:
            break
        master = solve_master(instance, pool, budget, scenarios, remaining,
                              symmetry=symmetry, nominal_floor=nominal_floor)
        upper = min(upper, master['upper'])
        selected = master['selected']
        value = oracle(instance, selected, gamma)
        if value['nominal'] >= nominal_floor and (value['value'], value['nominal'], -len(selected)) > (
                best_o['value'], best_o['nominal'], -len(best)):
            best, best_o = selected, value
        traces.append({'scenarios': len(scenarios), 'master': master,
                       'separation': value, 'best_lower': best_o['value'], 'upper': upper})
        if upper < best_o['value']:
            raise AssertionError('Full-design upper bound below a feasible full robust value')
        if upper == best_o['value']:
            break
        attack = tuple(value['attack'])
        if attack in scenarios:
            break
        scenarios.append(attack)
    if instance.nominal(best) < nominal_floor:
        raise AssertionError('Final incumbent violates nominal floor')
    return {'selected': best, 'value': best_o['value'], 'nominal': best_o['nominal'],
            'attack': best_o['attack'], 'upper': upper, 'certified': upper == best_o['value'],
            'seconds': time.perf_counter() - started, 'pool_size': len(pool),
            'active_candidates': len(instance.active), 'pattern_types': len(instance.types),
            'scenarios': len(scenarios), 'reduction': reduction, 'traces': traces,
            'symmetry': symmetry, 'nominal_floor': nominal_floor,
            'all_fixed_pairs_in_master': instance.p,
            'objective_scope': 'Full fixed-E0 XOR coverage under positive annotation completions; not biological truth.'}
