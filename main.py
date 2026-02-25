from nsga2 import run_nsga2
from spea2 import run_spea2
import matplotlib.pyplot as plt


def fill_none(hist, fallback=None):
    """
    Replace None values (no feasible yet) with a fallback.
    If fallback is None, use the first non-None value, else a big number.
    """
    if fallback is None:
        non_none = [v for v in hist if v is not None]
        fallback = non_none[0] if non_none else 1e9
    return [fallback if v is None else v for v in hist]


if __name__ == "__main__":
    data_path   = "p_median_capacitated.txt"
    instance_id = 1

    pop_size    = 200
    generations = 400
    pc          = 0.9
    pm          = 0.25

    demand_weighted = False
    customer_order  = "desc_demand"
    seed            = 1234

    # SPEA2 config
    metric = "euclidean"
    rounding = "floor"
    archive_size = 200

    # --- run NSGA-II (returns history list) ---
    nsga_hist = run_nsga2(
        data_path,
        instance_id=instance_id,
        pop_size=pop_size,
        generations=generations,
        pc=pc,
        pm=pm,
        demand_weighted=demand_weighted,
        customer_order=customer_order,
        seed=seed
    )

    # --- run SPEA2 (returns history list) ---
    spea_hist = run_spea2(
        data_path,
        instance_id,
        metric,
        rounding,
        pop_size=pop_size,
        archive_size=archive_size,
        generations=generations,
        pc=pc,
        pm=pm,
        demand_weighted=demand_weighted,
        customer_order=customer_order,
        seed=seed
    )

    # Make them plottable (replace None if feasibility isn't immediate)
    nsga_y = fill_none(nsga_hist)
    spea_y = fill_none(spea_hist)

    # Align lengths just in case (should both be generations+1)
    L = min(len(nsga_y), len(spea_y))
    nsga_y = nsga_y[:L]
    spea_y = spea_y[:L]
    x = list(range(L))

    plt.figure()
    plt.plot(x, nsga_y, label="NSGA-II best feasible cost")
    plt.plot(x, spea_y, label="SPEA2 best feasible cost")
    plt.xlabel("Generation")
    plt.ylabel("Best feasible cost")
    plt.title(f"Instance {instance_id}: NSGA-II vs SPEA2")
    plt.legend()
    plt.tight_layout()

    out = f"nsga2_vs_spea2_instance{instance_id}.png"
    plt.savefig(out, dpi=200)
    plt.show()

    print(f"\nSaved plot to: {out}")
    print(f"Final NSGA-II best feasible cost: {nsga_y[-1]:.4f}")
    print(f"Final SPEA2  best feasible cost: {spea_y[-1]:.4f}")