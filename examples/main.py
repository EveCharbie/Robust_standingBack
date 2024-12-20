from typing import Callable
from bioptim import (
    Solver,
)
from src.constants import (
    PATH_MODEL_1_CONTACT,
    PATH_MODEL,
)

from sommersault_taudot import prepare_ocp as prepare_ocp_ntc
from sommersault_htc_taudot import prepare_ocp as prepare_ocp_with_htc
from sommersault_ktc_taudot import prepare_ocp as prepare_ocp_with_ktc
from src.save_results import save_results_taudot
from src.save_results import save_results_holonomic_taudot

from src.multistart import prepare_multi_start


def main(prepare_ocp: Callable, save_results: Callable, multi_start: bool = False, condition: str = ""):
    # --- Parameters --- #
    movement = "backflip"
    version = "post_submission"

    WITH_MULTI_START = multi_start
    save_folder = f"../results/{str(movement)}_V{version}/{condition}"

    biorbd_model_path = (PATH_MODEL_1_CONTACT, PATH_MODEL, PATH_MODEL, PATH_MODEL, PATH_MODEL_1_CONTACT)
    phase_time = (0.2, 0.2, 0.3, 0.3, 0.3)
    n_shooting = (20, 20, 30, 30, 30)

    # Solver options
    solver = Solver.IPOPT(show_options=dict(show_bounds=True), _linear_solver="MA57", show_online_optim=False)
    solver.set_maximum_iterations(10000)
    solver.set_bound_frac(1e-8)
    solver.set_bound_push(1e-8)
    solver.set_tol(1e-6)

    if WITH_MULTI_START:

        combinatorial_parameters = {
            "bio_model_path": [biorbd_model_path],
            "phase_time": [phase_time],
            "n_shooting": [n_shooting],
            "WITH_MULTI_START": [True],
            "seed": list(range(0, 20)),
        }

        multi_start = prepare_multi_start(
            prepare_ocp,
            save_results,
            combinatorial_parameters=combinatorial_parameters,
            save_folder=save_folder,
            solver=solver,
            n_pools=1,
        )

        multi_start.solve()
    else:
        ocp = prepare_ocp(biorbd_model_path, phase_time, n_shooting, WITH_MULTI_START=False)
        # ocp.add_plot_penalty()

        solver.show_online_optim = False
        sol = ocp.solve(solver)
        sol.print_cost()

        # --- Save results --- #
        sol.graphs(show_bounds=True, save_name=str(movement) + "_V" + version)
        sol.animate()

        combinatorial_parameters = [biorbd_model_path, phase_time, n_shooting, WITH_MULTI_START, "no_seed"]
        save_results(sol, *combinatorial_parameters, save_folder=save_folder)


if "__main__" == __name__:
    main(prepare_ocp_ntc, save_results_taudot, multi_start=True, condition="ntc")
    main(prepare_ocp_with_ktc, save_results_taudot, multi_start=True, condition="ktc")
    main(prepare_ocp_with_htc, save_results_holonomic_taudot, multi_start=True, condition="htc")
