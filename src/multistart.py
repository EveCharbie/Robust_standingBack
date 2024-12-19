from bioptim import Solver, MultiStart


def prepare_multi_start(
    prepare_ocp,
    save_results,
    combinatorial_parameters: dict,
    save_folder: str = None,
    n_pools: int = 10,
    solver: Solver = None,
):
    """
    The initialization of the multi-start
    """

    return MultiStart(
        combinatorial_parameters=combinatorial_parameters,
        prepare_ocp_callback=prepare_ocp,
        post_optimization_callback=(save_results, {"save_folder": save_folder}),
        should_solve_callback=(should_solve, {"save_folder": save_folder}),
        n_pools=n_pools,
        solver=solver,
    )


def should_solve(*combinatorial_parameters, **extra_parameters):
    return True
