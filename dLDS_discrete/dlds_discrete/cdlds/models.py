import numpy as np
from pydmd import DMDc
import util_controls as uc
from dataclasses import dataclass
from datasets import CdLDSDataGenerator


@dataclass
class DLDSwithControlModel:
    """_summary_

    Attributes
    ----------
    datasets : CdLDSDataGenerator
        A class that generates data from a DLDS with controls system
    X : np.ndarray
        The data
    coeffs_ : np.ndarray
        The coefficients of the model
    F_ : np.ndarray
        The sub-dynamics of the model
    """

    datasets: CdLDSDataGenerator = CdLDSDataGenerator()
    X: np.ndarray = None
    coeffs_: np.ndarray = None
    F_: np.ndarray = None

    def fit(self, max_time=500, dt=0.1, num_subdyns=3, error_reco=np.inf, data=[], step_f=30, GD_decay=0.85, max_error=1e-3,
            max_iter=3000, F=[], coefficients=[], params={'update_c_type': 'inv', 'reg_term': 0, 'smooth_term': 0},
            epsilon_error_change=10**(-5), D=[], x_former=[], latent_dim=None, include_D=False, step_D=30, reg1=0, reg_f=0,
            max_data_reco=1e-3, sigma_mix_f=0.1, action_along_time='median', seed=0, seed_f=0,
            normalize_eig=True, init_distant_F=False, max_corr=0.1, decaying_reg=1, other_params_c={}, include_last_up=False):
        """
        This is the main function to train the model! 
        Inputs:
            max_time      = Number of time points for the dynamics. Relevant only if data is empty;
            dt            =  time interval for the dynamics
            num_subdyns   = number of sub-dynamics
            error_reco    = intial error for the reconstruction (do not touch)
            data          = if one wants to use a pre define groud-truth dynamics. If not empty - it overwrites max_time, dt, and dynamics_type
            step_f        = initial step size for GD on the sub-dynamics
            GD_decay      = Gradient descent decay rate
            max_error     = Threshold for the model error. If the model arrives at a lower reconstruction error - the training ends.
            max_iter      = # of max. iterations for training the model
            F             = pre-defined sub-dynamics. Keep empty if random.
            coefficients  = pre-defined coefficients. Keep empty if random.
            params        = dictionary that includes info about the regularization and coefficients solver. e.g. {'update_c_type':'inv','reg_term':0,'smooth_term':0}
            epsilon_error_change = check if the sub-dynamics do not change by at least epsilon_error_change, for at least 5 last iterations. Otherwise - add noise to f
            D             = pre-defined D matrix (keep empty if D = I)
            latent_dim    =  If D != I, it is the pre-defined latent dynamics.
            include_D     = If True -> D !=I; If False -> D = I
            step_D        = GD step for updating D, only if include_D is true
            reg1          = if include_D is true -> L1 regularization on D
            reg_f         = if include_D is true ->  Frobenius norm regularization on D
            max_data_reco = if include_D is true -> threshold for the error on the reconstruction of the data (continue training if the error (y - Dx)^2 > max_data_reco)
            sigma_mix_f            = std of noise added to mix f
            action_along_time      = the function to take on the error over time. Can be 'median' or 'mean'
            seed                   = random seed
            seed_f                 = random seed for initializing f
            normalize_eig          = whether to normalize each sub-dynamic by dividing by the highest abs eval
            init_distant_F         = when initializing F -> make sure that the correlation between each pair of {f}_i does not exeed a threshold
            max_corr               = max correlation between each pair of initial sub-dyns (relevant only if init_distant_F is True)
            decaying_reg           = decaying factor for the l1 regularization on the coefficients. If 1 - there is no decay. (should be a scalar in (0,1])
            other_params_c         = additional parameters for the update step of c
            include_last_up        = add another update step of the coefficients at the end
        """
        self.F_ = F
        self.coeffs_ = coefficients
        self.X = data

    def __str__(self):
        # we just want to print all attributes
        return str(self.__dict__.keys())


class ControlModel:
    """ control model 

    """

    def __init__(self, control_density=0.02):
        self.all_U = None
        self.all_A = None
        self.all_B = None
        self.sparsity_pattern = None
        self.control_density = control_density

    def fit(self, X, Y, num_iter, U_0=None, D_control=1):
        """Fit the control model from the data X and Y
        This function is directly translated from the MATLAB version of learn_control_signals from the repository https://github.com/Charles-Fieseler/Learning_Control_Signals_MATLAB


        Args:
            X (np.ndarray): Data matrix X
            Y (np.ndarray): Data matrix Y
            num_iter (int): Number of iterations
            U_0 (np.ndarray, optional): Initial control input. Defaults to None.
            D_control (int, optional): Number of control inputs. Defaults to 1.
        """

        if U_0 is None:
            U_0 = uc.init_U(X, Y, D_control)  # initial control input

        D_control, T = U_0.shape
        D_obs = X.shape[0]

        self.all_U = np.zeros((num_iter, D_control, T))  # all control inputs
        self.all_U[0] = U_0

        self.all_A = np.zeros((num_iter, D_obs, D_obs))  # all A matrices
        self.all_B = np.zeros((num_iter, D_obs, D_control))  # all B matrices
        self.sparsity_pattern = np.zeros(
            (D_control, T), dtype=bool)  # sparsity pattern of U

        all_err = np.zeros((num_iter, 2))  # error of the model

        for i in range(num_iter):
            print(f'Iteration {i + 1}/{num_iter}')
            # solve for A and B

            AB = Y @ np.linalg.pinv(np.vstack([X, self.all_U[i]]))

            A = AB[:, :D_obs]
            B = AB[:, D_obs:]

            # compute error
            all_err[i] = np.linalg.norm(A @ X + B @ self.all_U[i] - Y, 2)

            if i > 0:
                self.all_A[i+1] = A
                self.all_B[i+1] = B

            # solve for U using the current A and B
            U = np.linalg.lstsq(B, Y - A @ X, rcond=None)[0]

            # update sparsity pattern
            U_nonsparse = U.copy()
            U_nonsparse[~self.sparsity_pattern] = 0

            U[self.sparsity_pattern] = 0
            # re-updated sparsity pattern
            re_up_pattern = np.zeros_like(self.sparsity_pattern)
            has_converged = False

            for j in range(D_control):
                tmp = U[j, :]
                try:
                    threshold = max(np.quantile(tmp[tmp > 0], self.control_density),
                                    np.min(tmp[tmp > 0]))  # threshold for sparsity pattern, if control_density is 0.1, then we take the 0.1 quantile of the positive values of the control input as the threshold unless it is smaller than the smallest positive value
                except IndexError:
                    pass
                above_threshold = tmp > threshold
                if np.sum(above_threshold) == 0:
                    # if all control inputs are equal to or below the threshold, then the algorithm has converged
                    has_converged = True
                    print(
                        f'No values above threshold {threshold} in control input {j}')
                else:
                    has_converged = False
                # update sparsity pattern, if control input is below threshold
                self.sparsity_pattern[j, :] = ~(above_threshold)

            # is this correct
            if has_converged:
                break
            # set control inputs below threshold to 0
            U[self.sparsity_pattern] = 0

            # U = np.abs(U)
            self.all_U[i+1] = U
            # compute error
            all_err[i+1, 1] = np.linalg.norm(A @ X + B @ U - Y, 'fro')

        if has_converged:
            for n in range(i, num_iter):
                self.all_U[n] = self.all_U[i]
                self.all_A[n] = self.all_A[i]
                self.all_B[n] = self.all_B[i]
            pass
        else:

            print('not converged - calculate exact dmdc')
            dmdc = DMDc(svd_rank=2)
            dmdc.fit(X=X.T, I=self.all_U[num_iter-1].T)
            modes = dmdc.modes
            eigs = np.power(
                dmdc.eigs, dmdc.dmd_time["dt"] // dmdc.original_time["dt"]
            )
            A = np.linalg.multi_dot(
                [modes, np.diag(eigs), np.linalg.pinv(modes)]
            )
            self.all_A[-1] = A
            self.all_B[-1] = dmdc.B

    def transform(self, x, u):
        pass