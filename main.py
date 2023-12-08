import utils, hw6, importlib
import numpy as np

if __name__ == "__main__":
    model = utils.load_MDP("model.json")
    importlib.reload(utils)

    # 문제 1. Transition Probability
    P = hw6.compute_transition_matrix(model)
    sol_P = np.load("solution_P.npy")
    if not np.array_equal(P, sol_P):
        raise ValueError(
            "The computed transition matrix P does not match the ground truth."
        )

    # 문제 2. Update utility
    U_current = np.zeros((model.M, model.N))
    U_next = hw6.update_utility(model, P, U_current)
    model.visualize(U_next, save_path=True, figname="U_next")

    # 문제 3. Value iteration
    U = hw6.value_iteration(model)
    model.visualize(U, save_path=True, figname="result")
