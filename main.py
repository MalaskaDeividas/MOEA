from nsga2 import run_nsga2












if __name__ == "__main__":

    # IMPORTANT:
    # If your best-known values don't match, try:
    #   - metric="manhattan" in compute_distance_matrix
    #   - demand_weighted=False

    data_path = "p_median_capacitated.txt"

    run_nsga2(
        data_path,
        instance_id=1,          # or None to solve all
        pop_size=200,
        generations=400,
        pc=0.9,
        pm=0.25,
        demand_weighted=False,
        customer_order="desc_demand",
        seed=1234
    )

