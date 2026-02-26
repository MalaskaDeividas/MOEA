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
    dist: np.ndarray  # shape (n, n)


@dataclass
class Individual:
    medians: List[int]
    objectives: Tuple[float, float]          # (cost, violation)
    fitness: float = float("inf")            # SPEA2 fitness (smaller is better)


# -----------------------------
# Parsing
# -----------------------------
def parse_instances_from_text(text: str, metric="euclidean", rounding="floor") -> List[Instance]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    i = 0
    instances: List[Instance] = []

    while i < len(lines):
        a = lines[i].split()
        iid = int(a[0])
        best_known = float(a[1])
        i += 1

        b = lines[i].split()
        n = int(b[0])
        p = int(b[1])
        cap = float(b[2])
        i += 1

        customers: List[Customer] = []
        for _ in range(n):
            c = lines[i].split()
            customers.append(Customer(int(c[0]), float(c[1]), float(c[2]), float(c[3])))
            i += 1

        dist = compute_distance_matrix(customers, metric=metric, rounding=rounding)
        instances.append(Instance(iid, best_known, n, p, cap, customers, dist))

    return instances


def parse_instances_from_file(path: str, metric="euclidean", rounding="floor") -> List[Instance]:
    with open(path, "r", encoding="utf-8") as f:
        return parse_instances_from_text(f.read(), metric=metric, rounding=rounding)


# -----------------------------
# Distances
# -----------------------------
def compute_distance_matrix(customers: List[Customer], metric="euclidean", rounding="floor") -> np.ndarray:
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
        return np.floor(dist + 0.5)
    raise ValueError("rounding must be none/floor/ceil/round")


# -----------------------------
# Decode / evaluate
# -----------------------------
def assignment_cost(dist_ij: float, demand: float, demand_weighted: bool) -> float:
    return dist_ij * demand if demand_weighted else dist_ij


def decode_and_evaluate(
    inst: Instance,
    medians: List[int],
    demand_weighted: bool = False,
    customer_order: str = "desc_demand",
) -> Tuple[float, float]:
    n, p, cap = inst.n, inst.p, inst.capacity
    if len(medians) != p or len(set(medians)) != p:
        return (1e18, 1e18)

    loads = {m: 0.0 for m in medians}
    cost = 0.0

    if customer_order == "desc_demand":
        order = sorted(range(n), key=lambda i: inst.customers[i].demand, reverse=True)
    else:
        order = list(range(n))

    for ci in order:
        dmd = inst.customers[ci].demand

        best_feas = None
        best_feas_d = float("inf")
        best_any = None
        best_any_d = float("inf")

        for m in medians:
            dij = float(inst.dist[ci, m])

            if dij < best_any_d:
                best_any_d = dij
                best_any = m

            if loads[m] + dmd <= cap and dij < best_feas_d:
                best_feas_d = dij
                best_feas = m

        chosen = best_feas if best_feas is not None else best_any
        loads[chosen] += dmd
        cost += assignment_cost(float(inst.dist[ci, chosen]), dmd, demand_weighted)

    violation = sum(max(0.0, loads[m] - cap) for m in medians)
    return (cost, violation)


# -----------------------------
# Constraint domination (Deb)
# -----------------------------
def constraint_dominates(a: Individual, b: Individual) -> bool:
    ca, va = a.objectives
    cb, vb = b.objectives

    fa = (va <= 1e-12)
    fb = (vb <= 1e-12)

    if fa and not fb:
        return True
    if fb and not fa:
        return False
    if not fa and not fb:
        return va < vb

    # both feasible: minimize cost (and violation tie)
    return (ca <= cb and va <= vb) and (ca < cb or va < vb)


# -----------------------------
# Variation operators
# -----------------------------
def repair_unique(medians: List[int], n: int, p: int) -> List[int]:
    med = medians[:p]
    seen = set()
    out: List[int] = []
    for x in med:
        if 0 <= x < n and x not in seen:
            out.append(x)
            seen.add(x)
    while len(out) < p:
        cand = random.randrange(n)
        if cand not in seen:
            out.append(cand)
            seen.add(cand)
    return out


def random_individual(inst: Instance) -> Individual:
    medians = random.sample(range(inst.n), inst.p)
    return Individual(medians=medians, objectives=(1e18, 1e18))


def crossover_set(p1: Individual, p2: Individual, inst: Instance) -> Tuple[List[int], List[int]]:
    p = inst.p
    k = random.randint(1, p - 1)
    c1 = random.sample(p1.medians, k) + random.sample(p2.medians, p - k)
    c2 = random.sample(p2.medians, k) + random.sample(p1.medians, p - k)
    return repair_unique(c1, inst.n, p), repair_unique(c2, inst.n, p)


def mutate_swap(medians: List[int], inst: Instance, pm: float) -> List[int]:
    if random.random() > pm:
        return medians
    p = inst.p
    out = medians[:]
    idx = random.randrange(p)
    current = set(out)
    candidates = [i for i in range(inst.n) if i not in current]
    if candidates:
        out[idx] = random.choice(candidates)
    return repair_unique(out, inst.n, p)


# -----------------------------
# SPEA2 fitness assignment
# -----------------------------
def objective_distance(a: Individual, b: Individual) -> float:
    # Euclidean distance in objective space
    (c1, v1) = a.objectives
    (c2, v2) = b.objectives
    return math.sqrt((c1 - c2) ** 2 + (v1 - v2) ** 2)


def spea2_fitness(pop: List[Individual]) -> None:
    """
    Assign SPEA2 fitness to each individual in pop (in-place).
    """
    N = len(pop)
    # Dominance matrix
    dom = [[False] * N for _ in range(N)]
    strength = [0] * N

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if constraint_dominates(pop[i], pop[j]):
                dom[i][j] = True
                strength[i] += 1

    raw = [0] * N
    for i in range(N):
        s = 0
        for j in range(N):
            if j != i and dom[j][i]:
                s += strength[j]
        raw[i] = s

    # Density
    k = int(math.sqrt(N))
    k = max(1, k)
    # pairwise distances
    dmat = [[0.0] * N for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            d = objective_distance(pop[i], pop[j])
            dmat[i][j] = d
            dmat[j][i] = d

    density = [0.0] * N
    for i in range(N):
        ds = sorted(dmat[i][j] for j in range(N) if j != i)
        sigma_k = ds[min(k - 1, len(ds) - 1)] if ds else 0.0
        density[i] = 1.0 / (sigma_k + 2.0)

    for i in range(N):
        pop[i].fitness = raw[i] + density[i]


def spea2_truncate(archive: List[Individual], target_size: int) -> List[Individual]:
    """
    SPEA2 truncation: iteratively remove individuals in the most crowded region.
    Tie-break by lexicographic comparison of sorted distances.
    """
    A = archive[:]
    while len(A) > target_size:
        m = len(A)
        # distance lists
        dist_lists: List[List[float]] = []
        for i in range(m):
            ds = [objective_distance(A[i], A[j]) for j in range(m) if j != i]
            ds.sort()
            dist_lists.append(ds)

        # find index to remove: smallest lexicographic distance list
        remove_idx = 0
        for i in range(1, m):
            if dist_lists[i] < dist_lists[remove_idx]:
                remove_idx = i

        A.pop(remove_idx)

    return A


# -----------------------------
# SPEA2 solver
# -----------------------------
class SPEA2Solver:
    def __init__(
        self,
        pop_size: int = 200,
        archive_size: int = 200,
        generations: int = 400,
        pc: float = 0.9,
        pm: float = 0.25,
        demand_weighted: bool = False,
        customer_order: str = "desc_demand",
        seed: int = 1234,
    ):
        self.pop_size = pop_size
        self.archive_size = archive_size
        self.generations = generations
        self.pc = pc
        self.pm = pm
        self.demand_weighted = demand_weighted
        self.customer_order = customer_order
        self.seed = seed

    def evaluate(self, inst: Instance, ind: Individual) -> None:
        ind.objectives = decode_and_evaluate(
            inst,
            ind.medians,
            demand_weighted=self.demand_weighted,
            customer_order=self.customer_order,
        )

    def environmental_selection(self, union: List[Individual]) -> List[Individual]:
        spea2_fitness(union)

        # 1) take all with fitness < 1 into archive
        archive = [ind for ind in union if ind.fitness < 1.0]

        # 2) if too many, truncate
        if len(archive) > self.archive_size:
            archive = spea2_truncate(archive, self.archive_size)

        # 3) if too few, fill with best fitness individuals
        if len(archive) < self.archive_size:
            rest = [ind for ind in union if ind not in archive]
            rest.sort(key=lambda ind: ind.fitness)
            archive.extend(rest[: self.archive_size - len(archive)])

        return archive

    def tournament(self, archive: List[Individual]) -> Individual:
        a, b = random.sample(archive, 2)
        return a if a.fitness < b.fitness else b
    
    

    def run(self, inst):
        random.seed(self.seed)
        np.random.seed(self.seed)

        pop = [random_individual(inst) for _ in range(self.pop_size)]
        for ind in pop:
            self.evaluate(inst, ind)

        archive = []
        history = []

        def best_feasible_cost(arch):
            feas = [ind for ind in arch if ind.objectives[1] <= 1e-12]
            if feas:
                return min(feas, key=lambda ind: ind.objectives[0]).objectives[0]
            return None

        # record gen 0 (no archive yet -> use pop)
        feas0 = [ind for ind in pop if ind.objectives[1] <= 1e-12]
        history.append(min(feas0, key=lambda ind: ind.objectives[0]).objectives[0] if feas0 else None)

        for _ in range(self.generations):
            union = pop + archive
            archive = self.environmental_selection(union)

            # record after selection
            history.append(best_feasible_cost(archive))

            offspring = []
            while len(offspring) < self.pop_size:
                p1 = self.tournament(archive)
                p2 = self.tournament(archive)

                if random.random() < self.pc:
                    c1_med, c2_med = crossover_set(p1, p2, inst)
                else:
                    c1_med, c2_med = p1.medians[:], p2.medians[:]

                c1_med = mutate_swap(c1_med, inst, self.pm)
                c2_med = mutate_swap(c2_med, inst, self.pm)

                c1 = Individual(c1_med, (1e18, 1e18))
                c2 = Individual(c2_med, (1e18, 1e18))
                self.evaluate(inst, c1)
                self.evaluate(inst, c2)
                offspring.append(c1)
                if len(offspring) < self.pop_size:
                    offspring.append(c2)

            pop = offspring

        best = min(
            (ind for ind in archive if ind.objectives[1] <= 1e-12),
            key=lambda ind: ind.objectives[0],
            default=min(archive, key=lambda ind: (ind.objectives[1], ind.objectives[0]))
        )

        archive_sorted = sorted(archive, key=lambda ind: (ind.objectives[1], ind.objectives[0], ind.fitness))
        return archive_sorted, best, history


# -----------------------------
# Convenience runner
# -----------------------------
def run_spea2(path, instance_id, metric, rounding, **solver_kwargs):
    instances = parse_instances_from_file(path, metric=metric, rounding=rounding)
    if instance_id is not None:
        instances = [inst for inst in instances if inst.iid == instance_id]
        if not instances:
            raise ValueError(f"No instance with id={instance_id} in file.")

    solver = SPEA2Solver(**solver_kwargs)

    histories = {}  # iid -> history list

    for inst in instances:
        archive, best, history = solver.run(inst)   # <-- now 3 returns
        cost, viol = best.objectives

        print(f"\nInstance {inst.iid} (best-known: {inst.best_known})")
        print(f"  n={inst.n}, p={inst.p}, cap={inst.capacity}")
        print(f"  Best found: cost={cost:.4f}, violation={viol:.4f}")
        print(f"  Medians (0-based indices): {best.medians}")
        print(f"  Archive size: {len(archive)}")
        print("  Archive sample (cost, violation, fitness):")
        for ind in archive[: min(10, len(archive))]:
            print(f"    ({ind.objectives[0]:.4f}, {ind.objectives[1]:.4f}, F={ind.fitness:.4f})")

        histories[inst.iid] = history

    # If you run only one instance_id, return its history directly (nice for plotting)
    if instance_id is not None:
        return histories[instance_id]

    return histories
