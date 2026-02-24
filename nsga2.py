from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import random
import math
import numpy as np


# -----------------------------
# Data structures
# -----------------------------
@dataclass(frozen=True)
class Customer:
    cid: int
    x: float
    y: float
    demand: float


@dataclass
class Instance:
    iid: int
    best_known: Optional[float]
    n: int
    p: int
    capacity: float
    customers: List[Customer]
    dist: np.ndarray  # shape (n, n), distance between customer i and j


@dataclass
class Individual:
    medians: List[int]                 # indices [0..n-1] of chosen medians, length p
    objectives: Tuple[float, float]    # (cost, violation)
    rank: int = 10**9
    crowding: float = 0.0


# -----------------------------
# Parsing
# -----------------------------
def parse_instances_from_text(text):

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    i = 0
    instances: List[Instance] = []

    while i < len(lines):
        a = lines[i].split()
        if len(a) < 2:
            raise ValueError(f"Bad header line: {lines[i]}")
        iid = int(a[0])
        best_known = float(a[1])
        i += 1

        b = lines[i].split()
        if len(b) < 3:
            raise ValueError(f"Bad size line: {lines[i]}")
        n = int(b[0])
        p = int(b[1])
        cap = float(b[2])
        i += 1

        customers: List[Customer] = []
        for _ in range(n):
            c = lines[i].split()
            if len(c) < 4:
                raise ValueError(f"Bad customer line: {lines[i]}")
            cid = int(c[0])
            x = float(c[1])
            y = float(c[2])
            d = float(c[3])
            customers.append(Customer(cid=cid, x=x, y=y, demand=d))
            i += 1

        dist = compute_distance_matrix(customers, metric="euclidean", rounding="floor")
        instances.append(Instance(iid=iid, best_known=best_known, n=n, p=p, capacity=cap,
                                  customers=customers, dist=dist))
    return instances


def parse_instances_from_file(path: str) -> List[Instance]:
    with open(path, "r", encoding="utf-8") as f:
        return parse_instances_from_text(f.read())


#distance /config
def compute_distance_matrix(customers, metric="euclidean", rounding="round"):
    n = len(customers)
    coords = np.array([(c.x, c.y) for c in customers], dtype=float)
    dx = coords[:, None, 0] - coords[None, :, 0]
    dy = coords[:, None, 1] - coords[None, :, 1]

    if metric == "euclidean":
        dist = np.sqrt(dx * dx + dy * dy)
    elif metric == "manhattan":
        dist = np.abs(dx) + np.abs(dy)
    else:
        raise ValueError("metric must be 'euclidean' or 'manhattan'")

    if rounding == "none":
        return dist
    if rounding == "floor":
        return np.floor(dist)
    if rounding == "ceil":
        return np.ceil(dist)
    if rounding == "round":
        return np.floor(dist + 0.5)  # standard .5 up

    raise ValueError("rounding must be none/floor/ceil/round")


def assignment_cost(dist_ij: float, demand: float, demand_weighted: bool = True) -> float:
    return dist_ij * demand if demand_weighted else dist_ij


def decode_and_evaluate(
    inst: Instance,
    medians: List[int],
    demand_weighted: bool = True,
    customer_order: str = "desc_demand"  # "desc_demand" | "id"
):
    """
    Greedy assignment with capacity tracking.
    Always assigns every customer; if no capacity remains, assigns anyway and counts violation.
    Returns (cost, total_violation).
    """
    n, p, cap = inst.n, inst.p, inst.capacity
    if len(medians) != p or len(set(medians)) != p:
        # invalid genotype -> huge penalty
        return (1e18, 1e18)

    loads = {m: 0.0 for m in medians}
    cost = 0.0

    # choose assignment order (helps feasibility)
    if customer_order == "desc_demand":
        order = sorted(range(n), key=lambda i: inst.customers[i].demand, reverse=True)
    else:
        order = list(range(n))

    for ci in order:
        dmd = inst.customers[ci].demand

        # try nearest feasible median first
        # (p is small, simple scan is fine)
        best_feasible = None
        best_feasible_dist = float("inf")
        best_any = None
        best_any_dist = float("inf")

        for m in medians:
            dij = float(inst.dist[ci, m])

            if dij < best_any_dist:
                best_any_dist = dij
                best_any = m

            if loads[m] + dmd <= cap and dij < best_feasible_dist:
                best_feasible_dist = dij
                best_feasible = m

        chosen = best_feasible if best_feasible is not None else best_any
        loads[chosen] += dmd
        cost += assignment_cost(float(inst.dist[ci, chosen]), dmd, demand_weighted)

    violation = 0.0
    for m in medians:
        violation += max(0.0, loads[m] - cap)

    return (cost, violation)


# NSGA-II core, dominance, sorting, crowding
def constraint_dominates(a, b):

    cost_a, viol_a = a.objectives
    cost_b, viol_b = b.objectives

    feasible_a = (viol_a <= 1e-12)
    feasible_b = (viol_b <= 1e-12)

    if feasible_a and not feasible_b:
        return True
    if feasible_b and not feasible_a:
        return False
    if not feasible_a and not feasible_b:
        return viol_a < viol_b

    # both feasible: minimize cost (and violation ties)
    return (cost_a <= cost_b and viol_a <= viol_b) and (cost_a < cost_b or viol_a < viol_b)


def fast_non_dominated_sort(pop):
    S: Dict[int, List[int]] = {}
    n_dom = [0] * len(pop)
    fronts: List[List[int]] = []

    for i in range(len(pop)):
        S[i] = []
        n_dom[i] = 0
        for j in range(len(pop)):
            if i == j:
                continue
            if constraint_dominates(pop[i], pop[j]):
                S[i].append(j)
            elif constraint_dominates(pop[j], pop[i]):
                n_dom[i] += 1

        if n_dom[i] == 0:
            pop[i].rank = 0

    current = [i for i in range(len(pop)) if n_dom[i] == 0]
    fronts.append(current)

    r = 0
    while fronts[r]:
        next_front: List[int] = []
        for i in fronts[r]:
            for j in S[i]:
                n_dom[j] -= 1
                if n_dom[j] == 0:
                    pop[j].rank = r + 1
                    next_front.append(j)
        r += 1
        fronts.append(next_front)

    fronts.pop()  # last empty
    return [[pop[i] for i in f] for f in fronts]


def crowding_distance(front):
    if not front:
        return
    m = 2  # number of objectives: (cost, violation)
    for ind in front:
        ind.crowding = 0.0

    for k in range(m):
        front.sort(key=lambda ind: ind.objectives[k])
        front[0].crowding = float("inf")
        front[-1].crowding = float("inf")
        minv = front[0].objectives[k]
        maxv = front[-1].objectives[k]
        if abs(maxv - minv) < 1e-18:
            continue
        for i in range(1, len(front) - 1):
            prevv = front[i - 1].objectives[k]
            nextv = front[i + 1].objectives[k]
            front[i].crowding += (nextv - prevv) / (maxv - minv)


def binary_tournament(pop):
    a, b = random.sample(pop, 2)
    if a.rank < b.rank:
        return a
    if b.rank < a.rank:
        return b
    # same rank: higher crowding wins
    return a if a.crowding > b.crowding else b


def random_individual(inst: Instance) -> Individual:
    medians = random.sample(range(inst.n), inst.p)
    return Individual(medians=medians, objectives=(1e18, 1e18))


def crossover_set(parent1, parent2, inst):

    p = inst.p
    s1 = parent1.medians[:]
    s2 = parent2.medians[:]
    k = random.randint(1, p - 1)

    child1 = random.sample(s1, k) + random.sample(s2, p - k)
    child2 = random.sample(s2, k) + random.sample(s1, p - k)

    child1 = repair_unique(child1, inst.n, p)
    child2 = repair_unique(child2, inst.n, p)
    return child1, child2


def mutate_swap(medians, inst, pm):
    if random.random() > pm:
        return medians
    p = inst.p
    chosen = medians[:]
    idx = random.randrange(p)
    current = set(chosen)
    candidates = [i for i in range(inst.n) if i not in current]
    if not candidates:
        return chosen
    chosen[idx] = random.choice(candidates)
    return repair_unique(chosen, inst.n, p)


def repair_unique(medians, n, p):

    med = medians[:p]
    seen = set()
    cleaned: List[int] = []
    for x in med:
        if x not in seen and 0 <= x < n:
            cleaned.append(x)
            seen.add(x)
    while len(cleaned) < p:
        cand = random.randrange(n)
        if cand not in seen:
            cleaned.append(cand)
            seen.add(cand)
    return cleaned


class NSGA2Solver:
    def __init__(
        self,
        pop_size: int = 200,
        generations: int = 400,
        pc: float = 0.9,
        pm: float = 0.2,
        demand_weighted: bool = True,
        customer_order: str = "desc_demand",
        seed: int = 1234
    ):
        self.pop_size = pop_size
        self.generations = generations
        self.pc = pc
        self.pm = pm
        self.demand_weighted = demand_weighted
        self.customer_order = customer_order
        self.seed = seed

    def evaluate(self, inst, ind):
        ind.objectives = decode_and_evaluate(
            inst,
            ind.medians,
            demand_weighted=self.demand_weighted,
            customer_order=self.customer_order,
        )

    def run(self, inst):
        random.seed(self.seed)
        np.random.seed(self.seed)

        # init
        pop = [random_individual(inst) for _ in range(self.pop_size)]
        for ind in pop:
            self.evaluate(inst, ind)

        # rank/crowding init
        fronts = fast_non_dominated_sort(pop)
        for f in fronts:
            crowding_distance(f)

        for gen in range(self.generations):
            # offspring
            offspring: List[Individual] = []
            while len(offspring) < self.pop_size:
                p1 = binary_tournament(pop)
                p2 = binary_tournament(pop)

                if random.random() < self.pc:
                    c1_med, c2_med = crossover_set(p1, p2, inst)
                else:
                    c1_med, c2_med = p1.medians[:], p2.medians[:]

                c1_med = mutate_swap(c1_med, inst, self.pm)
                c2_med = mutate_swap(c2_med, inst, self.pm)

                c1 = Individual(medians=c1_med, objectives=(1e18, 1e18))
                c2 = Individual(medians=c2_med, objectives=(1e18, 1e18))
                self.evaluate(inst, c1)
                self.evaluate(inst, c2)
                offspring.append(c1)
                if len(offspring) < self.pop_size:
                    offspring.append(c2)

            # combine + select next gen
            combined = pop + offspring
            fronts = fast_non_dominated_sort(combined)

            new_pop: List[Individual] = []
            for f in fronts:
                crowding_distance(f)
                if len(new_pop) + len(f) <= self.pop_size:
                    new_pop.extend(f)
                else:
                    # fill remainder by crowding
                    f.sort(key=lambda ind: ind.crowding, reverse=True)
                    need = self.pop_size - len(new_pop)
                    new_pop.extend(f[:need])
                    break

            pop = new_pop

        # final nondominated front
        fronts = fast_non_dominated_sort(pop)
        pareto_front = fronts[0]
        crowding_distance(pareto_front)

        best_feasible = min(
            (ind for ind in pop if ind.objectives[1] <= 1e-12),
            key=lambda ind: ind.objectives[0],
            default=min(pop, key=lambda ind: (ind.objectives[1], ind.objectives[0]))
        )
        return pareto_front, best_feasible

def run_nsga2(
    path: str,
    instance_id: Optional[int] = None,
    **solver_kwargs
) -> None:
    instances = parse_instances_from_file(path)
    if instance_id is not None:
        instances = [inst for inst in instances if inst.iid == instance_id]
        if not instances:
            raise ValueError(f"No instance with id={instance_id} in file.")

    solver = NSGA2Solver(**solver_kwargs)



    for inst in instances:
        front, best = solver.run(inst)
        cost, viol = best.objectives

        print(f"\nInstance {inst.iid} (best-known: {inst.best_known})")
        print(f"  n={inst.n}, p={inst.p}, cap={inst.capacity}")
        print(f"  Best found: cost={cost:.4f}, violation={viol:.4f}")
        print(f"  Medians (0-based indices): {best.medians}")
        print(f"  Pareto front size: {len(front)}")

        # show a few front points
        front_sorted = sorted(front, key=lambda ind: (ind.objectives[1], ind.objectives[0]))
        print("  Front sample (cost, violation):")
        for ind in front_sorted[:min(10, len(front_sorted))]:
            print(f"    ({ind.objectives[0]:.4f}, {ind.objectives[1]:.4f})")


