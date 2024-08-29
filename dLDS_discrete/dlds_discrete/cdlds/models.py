from inspect import stack
from locale import normalize
from mimetypes import init
from turtle import st
from attrs import field
from networkx import sigma
import numpy as np
from pydmd import DMDc
from dataclasses import dataclass, field
from datasets import CdLDSDataGenerator
from scipy import linalg
from scipy.linalg import expm
from sklearn import linear_model
import util as uc
import plotly.express as px
import pylops
import torch
import slim
from sklearn.linear_model import RANSACRegressor


class DeepDLDS(torch.nn.Module):

    def __init__(self, input_size, output_size, num_subdyn, time_points, softmax_temperature=1):
        super().__init__()

        self.softmax_temperature = softmax_temperature

        self.F = torch.nn.ParameterList()  # can't be a simple list

        # self.coeffs = torch.nn.Parameter(torch.tensor(
        #    np.random.rand(num_subdyn, time_points), requires_grad=True))

        # initialize coefficients with just ones with a little noise
        self.coeffs = torch.nn.Parameter(torch.ones(
            num_subdyn, time_points), requires_grad=True)  # adding noise: + 0.05*torch.randn(num_subdyn, time_points)

        self.Bias = torch.nn.Parameter(torch.tensor(
            np.random.rand(num_subdyn, time_points), requires_grad=True))

        self.Bias.data = torch.nn.functional.sigmoid(self.Bias.data)

        for i in range(num_subdyn):

            # torch init

            f_i = slim.linear.BoundedNormLinear(
                input_size, input_size, bias=False, sigma_max=1.0, sigma_min=0)

            # initialize F
            # f_i.(torch.nn.init.xavier_normal)

            self.F.append(f_i)

    def forward(self, x_t, t):
        # print(f'coefficients:{self.coeffs[0, t]}')
        # print(f'F output:{self.F[0].weight}')
        # print(f'x input:{x_t}')
        # print(f'F output:{self.F[0](x_t)}')
        y_t = torch.stack([self.coeffs[i, t]*f_i(x_t.unsqueeze(0)) + self.Bias[i, t]
                           for i, f_i in enumerate(self.F)]).sum(dim=0)
        return y_t

    @property
    def soft_coeffs(self):
        return torch.nn.functional.softmax(self.coeffs / self.softmax_temperature, dim=0)

    def multi_step(self):

        _coeffs = self.soft_coeffs
        Y = torch.zeros((self.step_ahead, self.input_size))
        y0 = self.batch
        Y[0, :] = y0

        for t in range(self.step_ahead):
            # combination of all f_i * c_i
            y = torch.stack([_coeffs[i, t]*f_i(y0)
                            for i, f_i in enumerate(self.F)]).sum(dim=0)
            Y[t, :] = y
            y0 = y

        return Y


class SimpleNN(torch.nn.Module):
    """a simple fully connected neural network with ReLU activation functions and a linear output layer
    """

    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()

        in_size = input_size
        self.layers = []

        # define multiple layers in a loop
        for hidden_size in hidden_sizes:
            # linear layer computes output = input * weights + bias
            self.layers.append(torch.nn.Linear(in_size, hidden_size))
            self.layers.append(torch.nn.ReLU())
            in_size = hidden_size

        self.layers.append(torch.nn.Linear(in_size, output_size))
        # no final activation function because we have a regression problem

        self.network = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        # this forward function is always called when we call the model (it's somewhere in __call__ method)
        return self.network(x)


@dataclass
class DLDSwithControl:
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

    datasets: CdLDSDataGenerator = field(default_factory=CdLDSDataGenerator)
    X: np.ndarray = None
    num_true_dim: int = None
    num_subdyns: int = None
    num_time_points: int = None
    num_control_inputs: int = None
    coeffs_: np.ndarray = None
    control_density: float = 0.2
    F_: np.ndarray = None
    U_: np.ndarray = None
    B_: np.ndarray = None

    # %% Basic Model Functions
    # %% Main Model Training

    def fit(self, max_time=500, dt=0.1, dynamics_type='cyl', num_subdyns=3,
            error_reco=np.inf,  data=[], step_f=30, GD_decay=0.85, max_error=1e-3,
            max_iter=3000, F=[], coefficients=[], params={'update_c_type': 'inv', 'reg_term': 0, 'smooth_term': 0},
            epsilon_error_change=10**(-5), D=[], x_former=[], latent_dim=None, include_D=False, step_D=30, reg1=0, reg_f=0,
            max_data_reco=1e-3,  sigma_mix_f=0.1,  action_along_time='median', to_print=True, seed=0, seed_f=0,
            normalize_eig=True,  params_ex={'radius': 1, 'num_cyls': 5, 'bias': 0, 'exp_power': 0.2},
            init_distant_F=False, max_corr=0.1, decaying_reg=1, other_params_c={}, include_last_up=False, gradient_learning=False):
        """
        This is the main function to train the model!
        Inputs:
            max_time      = Number of time points for the dynamics. Relevant only if data is empty;
            dt            =  time interval for the dynamics
            dynamics_type = type of the dynamics. Can be 'cyl', 'lorenz', 'multi_cyl', 'torus', 'circ2d', 'spiral'
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
            to_print               = to print error value while training? (boolean)
            seed                   = random seed
            seed_f                 = random seed for initializing f
            normalize_eig          = whether to normalize each sub-dynamic by dividing by the highest abs eval
            params_ex              = parameters related to the creation of the ground truth dynamics. e.g. {'radius':1, 'num_cyls': 5, 'bias':0,'exp_power':0.2}
            init_distant_F         = when initializing F -> make sure that the correlation between each pair of {f}_i does not exeed a threshold
            max_corr               = max correlation between each pair of initial sub-dyns (relevant only if init_distant_F is True)
            decaying_reg           = decaying factor for the l1 regularization on the coefficients. If 1 - there is no decay. (should be a scalar in (0,1])
            other_params_c         = additional parameters for the update step of c
            include_last_up        = add another update step of the coefficients at the end
        """
        self.X = data

        latent_dyn = self.X
        # number of dimensions, number of time points
        self.n_true_dim, self.num_time_points = self.X.shape

        mat_shape = (self.n_true_dim, self.n_true_dim)

        """
        Initialize F
        """

        self.F_ = [self.initialize(mat_shape, normalize=True,
                                   r_seed=seed_f+i) for i in range(num_subdyns)]

        """
        Initialize coefficients
        """

        # self.coeffs_ = self.initialize((num_subdyns, self.num_time_points-1))

        # initialize coefficients with just ones
        self.coeffs_ = np.ones((num_subdyns, self.num_time_points-1))

        """
        Initialize U
        """

        X1 = self.X[:, :-1]
        X2 = self.X[:, 1:]

        if not self.num_control_inputs:
            self.num_control_inputs = 1
        # initial control input
        # -1 because we don't have a control input for the last time point
        self.U_ = uc.init_U(X1, X2, self.num_control_inputs)
        # all control inputs
        self.all_U = np.zeros(
            (max_iter, self.num_control_inputs, self.num_time_points-1))
        self.all_U[0] = self.U_
        self.all_B = np.zeros(
            (max_iter, self.n_true_dim, self.num_control_inputs))  # all B matrices

        self.B_ = (X2 - self.create_reco(X1, self.coeffs_,
                                         self.F_, controls=None, B=None)) @ np.linalg.pinv(self.U_)  # refactor

        self.all_B[0] = self.B_

        self.sparsity_pattern = np.zeros(
            (self.num_control_inputs, self.num_time_points-1), dtype=bool)

        cur_reco = self.create_reco(latent_dyn=latent_dyn,
                                    coefficients=self.coeffs_, F=self.F_, controls=self.U_, B=self.B_)

        data_reco_error = np.inf

        counter = 1

        error_reco_array = []

        while data_reco_error > max_data_reco and (counter < max_iter):
            """
            Decay reg
            """
            if params['update_c_type'] == 'lasso':
                params['reg_term'] = params['reg_term']*decaying_reg

            """
            Update coefficients and U
            """

            if counter != 1:
                # if counter % 10 == 0:  # update c only every 10 iterations
                self.coeffs_ = self.update_c(
                    self.F_, latent_dyn, params, random_state=seed, other_params=other_params_c, cofficients=self.coeffs_, controls=self.U_, B=self.B_, gradient_learning=gradient_learning)

            # we want to directly solve for B given F, U, X1, X2 and the current coefficients
            self.B_ = (X2 - self.create_reco(X1, self.coeffs_,
                                             self.F_, controls=None, B=None)) @ np.linalg.pinv(self.U_)  # + 10**-10 * np.eye(self.U_.shape[0]))  # add a small value to the diagonal to make the matrix invertible

            self.all_B[counter] = self.B_

            """
            Update F
            """

            self.F_ = self.update_f_all(latent_dyn, self.F_, self.coeffs_, self.U_, self.B_, step_f, normalize=False,
                                        action_along_time=action_along_time, normalize_eig=normalize_eig)

            step_f *= GD_decay

            mid_reco = self.create_reco(
                latent_dyn, self.coeffs_, self.F_, self.U_, self.B_)
            error_reco = np.mean((latent_dyn - mid_reco)**2)

            error_reco_array.append(error_reco)

            if np.mean(np.abs(np.diff(error_reco_array[-5:]))) < epsilon_error_change:
                self.F_ = [f_i + sigma_mix_f *
                           np.random.randn(f_i.shape[0], f_i.shape[1]) for f_i in self.F_]

            """
            Update U
            """

            if counter < 0:  # % (max_iter * 0.1) == 0:

                # solve for U given F, B, X1, X2 and the current coefficients
                # delta_U = np.linalg.lstsq(self.B_, X2 - self.create_reco(
                #    X1, self.coeffs_, self.F_, controls=self.U_, B=self.B_), rcond=None)[0]
                # self.U_ = self.U_ + delta_U

                # update U via NMF
                self.U_ = uc.init_U(X1, X2, self.num_control_inputs, err=X2 - self.create_reco(
                    X1, self.coeffs_, self.F_, controls=None, B=None))

                # but keep the sparsity pattern
                self.U_[self.sparsity_pattern] = 0

                # self.U_ = self.U_ + 0.1 * (new_U - self.U_) # take a gradient step

                """
                threshold for the control input sparsity
                """

                for j in range(self.num_control_inputs):
                    tmp = self.U_[j, :]
                    threshold = max(np.quantile(tmp[tmp > 0], self.control_density),
                                    np.min(tmp[tmp > 0]))  # threshold for sparsity pattern, if control_density is 0.1, then we take the 0.1 quantile of the positive values of the control input as the threshold unless it is smaller than the smallest positive value

                    above_threshold = tmp > threshold

                    # print of tmp those values that are above the threshold
                    # print(
                    #    f'Values under threshold {threshold} in control input {j}: {tmp[~above_threshold]}')

                    if np.sum(above_threshold) == 0:
                        # if all control inputs are equal to or below the threshold, then the algorithm has converged
                        has_converged = True
                    else:
                        has_converged = False
                    # update sparsity pattern, if control input is below threshold
                    self.sparsity_pattern[j, :] = ~(above_threshold)

                # is this correct
                if has_converged:
                    break
                # set control inputs below threshold to 0
                self.U_[self.sparsity_pattern] = 0

            # U = np.abs(U)
            self.all_U[counter] = self.U_

            counter += 1
            if counter == max_iter:
                print('Arrived to max iter')

        # Post training adjustments
        if include_last_up:
            self.coeffs_ = self.update_c(self.F_, latent_dyn, params,  {
                'reg_term': 0, 'update_c_type': 'inv', 'smooth_term': 0, 'num_iters': 10, 'threshkind': 'soft'}, controls=self.U_)
        else:
            self.coeffs_ = self.update_c(self.F_, latent_dyn, params,
                                         other_params=other_params_c, controls=self.U_)

        return error_reco_array

    def create_reco(self, latent_dyn, coefficients, F, controls, B, type_find='median', min_far=10, smooth_coeffs=False,
                    smoothing_params={'wind': 5}):
        """
        This function creates the reconstruction
        Inputs:
            latent_dyn   = the ground truth latent dynamics
            coefficients = the operators coefficients (c(t)_i)
            F            = a list of transport operators (a list with M transport operators,
                                                        each is a square matrix, kXk, where k is the latent dynamics
                                                        dimension )
            type_find    = 'median'
            min_far      = 10
            smooth_coeffs= False
            smoothing_params = {'wind':5}

        Outputs:
            cur_reco    = dLDS reconstruction of the latent dynamics

        """

        cur_reco = np.hstack([self.create_next(latent_dyn, coefficients, F, controls, B, time_point)
                              for time_point in range(latent_dyn.shape[1]-1)])
        cur_reco = np.hstack([latent_dyn[:, 0].reshape((-1, 1)), cur_reco])

        return cur_reco

    def initialize(self, size_mat, r_seed=0, dist_type='norm', init_params={'loc': 0, 'scale': 1}, normalize=False):
        """
        !!!TODO: TAKEN FROM main_functions, need to revisit this

        This is an initialization function to initialize matrices like G_i and c.
        Inputs:
        size_mat    = 2-element tuple or list, describing the shape of the mat
        r_seed      = random seed (should be integer)
        dist_type   = distribution type for initialization; can be 'norm' (normal dist), 'uni' (uniform dist),'inti', 'sprase', 'regional'
        init_params = a dictionary with params for initialization. The keys depends on 'dist_type'.
                        keys for norm -> ['loc','scale']
                        keys for inti and uni -> ['low','high']
                        keys for sparse -> ['k'] -> number of non-zeros in each row
                        keys for regional -> ['k'] -> repeats of the sub-dynamics allocations
        normalize   = whether to normalize the matrix
        Output:
            the random matrix with size 'size_mat'
        """
        np.random.seed(r_seed)
        if dist_type == 'norm':
            # print('init_params:', init_params)
            # print('size_mat:', size_mat)
            rand_mat = np.random.normal(
                loc=init_params['loc'], scale=init_params['scale'], size=size_mat)

        if normalize:
            rand_mat = self.norm_mat(rand_mat)

        return rand_mat

    def norm_mat(self, mat, type_norm='evals', to_norm=True):
        """
        This function comes to norm matrices.
        Inputs:
            mat       = the matrix to norm
            type_norm = what type of normalization to apply. Can be:
                - 'evals' - normalize by dividing by the max eigen-value
                - 'max'   - divide by the maximum abs value in the matrix
                - 'exp'   -  normalization using matrix exponential (matrix exponential)
            to_norm   = whether to norm or not to.
        Output:
            the normalized matrix
        """
        if to_norm:
            if type_norm == 'evals':
                eigenvalues, _ = linalg.eig(mat)
                mat = mat / np.max(np.abs(eigenvalues))
            elif type_norm == 'max':
                mat = mat / np.max(np.abs(mat))
            elif type_norm == 'exp':
                mat = np.exp(-np.trace(mat))*expm(mat)
        return mat

    def update_c(self, F, latent_dyn,
                 params_update_c={'update_c_type': 'inv', 'reg_term': 0, 'smooth_term': 0, 'to_norm_fx': False}, clear_dyn=[],
                 direction='c2n', other_params={'warm_start': False}, random_state=0, skip_error=False, cofficients=[], controls=[], B=None, gradient_learning=False):
        """
        The function comes to update the coefficients of the sub-dynamics, {c_i}, by solving the inverse or solving lasso.
        Inputs:
            F               = list of sub-dynamics. Should be a list of k X k arrays.
            latent_dyn      = latent_dynamics (dynamics dimensions X time)
            params_update_c = dictionary with keys:
                update_c_type  = options:
                    - 'inv' (least squares)
                    - 'lasso' (sklearn lasso)
                    - 'fista' (https://pylops.readthedocs.io/en/latest/api/generated/pylops.optimization.sparsity.FISTA.html)
                    - 'omp' (https://pylops.readthedocs.io/en/latest/gallery/plot_ista.html#sphx-glr-gallery-plot-ista-py)
                    - 'ista' (https://pylops.readthedocs.io/en/latest/api/generated/pylops.optimization.sparsity.ISTA.html)
                    - 'IRLS' (https://pylops.readthedocs.io/en/latest/api/generated/pylops.optimization.sparsity.IRLS.html)
                    - 'spgl1' (https://pylops.readthedocs.io/en/latest/api/generated/pylops.optimization.sparsity.SPGL1.html)


                    - . Refers to the way the coefficients should be claculated (inv -> no l1 regularization)
                reg_term       = scalar between 0 to 1, describe the reg. term on the cofficients
                smooth_term    = scalar between 0 to 1, describe the smooth term on the cofficients (c_t - c_(t-1))
            direction      = can be c2n (clean to noise) OR  n2c (noise to clean)
            other_params   = additional parameters for the lasso solver (optional)
            random_state   = random state for reproducability (optional)
            skip_error     = whether to skip an error when solving the inverse for c (optional)
            cofficients    = needed only if smooth_term > 0. This is the reference coefficients matrix to apply the constraint (c_hat_t - c_(t-1)) on.

        Outputs:
            coefficients matrix (k X T), type = np.array

        example:
        coeffs = update_c(np.random.rand(2,2), np.random.rand(2,15),{})
        """

        if isinstance(latent_dyn, list):
            if len(latent_dyn) == 1:
                several_dyns = False
            else:
                several_dyns = True
        else:
            several_dyns = False  # we always hit this condition because latent_dyn is a ndarray
        if several_dyns:
            n_times = latent_dyn[0].shape[1]-1
        else:
            n_times = latent_dyn.shape[1]-1

        params_update_c = {**{'update_c_type': 'inv',
                              'smooth_term': 0, 'reg_term': 0}, **params_update_c}
        if len(clear_dyn) == 0:
            clear_dyn = latent_dyn
        if direction == 'n2c':
            latent_dyn, clear_dyn = clear_dyn,  latent_dyn
        if isinstance(F, np.ndarray):
            F = [F]
        coeffs_list = []

        # deltas = []

        for time_point in np.arange(n_times):
            if not several_dyns:
                cur_dyn = clear_dyn[:, time_point]
                next_dyn = latent_dyn[:, time_point+1]
                total_next_dyn = next_dyn
                f_x_mat = []
                for f_i in F:
                    f_x_mat.append(f_i @ cur_dyn)
                stacked_fx = np.vstack(f_x_mat).T
                stacked_fx[stacked_fx > 10**8] = 10**8

            else:
                # we never hit this condition
                total_next_dyn = []
                for dyn_num in range(len(latent_dyn)):
                    cur_dyn = clear_dyn[dyn_num][:, time_point]
                    next_dyn = latent_dyn[dyn_num][:, time_point+1]
                    total_next_dyn.extend(next_dyn.flatten().tolist())
                    f_x_mat = []
                    for f_num, f_i in enumerate(F):
                        f_x_mat.append(f_i @ cur_dyn)  # maybe add U?
                    if dyn_num == 0:
                        stacked_fx = np.vstack(f_x_mat).T
                    else:
                        stacked_fx = np.vstack(
                            [stacked_fx, np.vstack(f_x_mat).T])
                    stacked_fx[stacked_fx > 10**8] = 10**8

                total_next_dyn = np.reshape(np.array(total_next_dyn), (-1, 1))

            if B is not None:
                total_next_dyn = np.array(
                    total_next_dyn) - (B @ controls[:, time_point])

                # px.line(total_next_dyn-latent_dyn,
                #        title='total_next_dyn').show()

                # raise ValueError('B is not None')

            if len(F) == 1:
                stacked_fx = np.reshape(stacked_fx, [-1, 1])
            # print(
            #    f'Update c at time point {time_point} with {params_update_c} and coefficients shape {len(cofficients)}')
            if params_update_c['smooth_term'] > 0 and time_point > 0:
                if len(coeffs_list) == 0:
                    print(
                        "Warning: you called the smoothing option without defining coefficients")
            if params_update_c['smooth_term'] > 0 and time_point > 0 and len(coeffs_list) > 0:
                # c_former = cofficients[:, time_point-1]  # .reshape((-1, 1))
                c_former = coeffs_list[-1]
                total_next_dyn_full = np.hstack(
                    [total_next_dyn, np.sqrt(params_update_c['smooth_term'])*c_former])

                # stacked_fx_full = np.hstack([stacked_fx, np.sqrt(
                # params_update_c['smooth_term'])*np.eye(len(stacked_fx))])

                stacked_fx_full = np.vstack([stacked_fx, np.sqrt(
                    params_update_c['smooth_term'])*np.eye(len(c_former))])  # .T # EDIT maybe instead of stacked_fx use c_former and instead of np.hstack vstack

            else:
                total_next_dyn_full = total_next_dyn
                stacked_fx_full = stacked_fx

            if params_update_c['update_c_type'] == 'inv' or (params_update_c['reg_term'] == 0 and params_update_c['smooth_term'] == 0):
                try:

                    # if first iteration, extend stacked_fx_full and total_next_dyn_full to be the same size as the rest of the iterations

                    if gradient_learning:

                        learning_rate = 0.1

                        if len(coeffs_list) == 0:
                            coeffs = np.ones((len(F), 1))

                        else:
                            solved_coeffs = (linalg.pinv(
                                stacked_fx_full) @ total_next_dyn_full.reshape((-1, 1)))

                            # coeffs = self.coeffs_[time_point]

                            coeffs = self.coeffs_[:, time_point].reshape((-1, 1)) + learning_rate * \
                                (solved_coeffs -
                                 self.coeffs_[:, time_point].reshape((-1, 1)))
                    else:
                        coeffs = (linalg.pinv(stacked_fx_full) @
                                  total_next_dyn_full.reshape((-1, 1)))

                except Exception as e:
                    if not skip_error:
                        raise e
                        # raise NameError(
                        #    'A problem in taking the inverse of fx when looking for the model coefficients')
                    else:
                        return np.nan*np.ones((len(F), latent_dyn.shape[1]))
            elif params_update_c['update_c_type'] == 'lasso':

                clf = linear_model.Lasso(
                    alpha=params_update_c['reg_term'], random_state=random_state, **other_params)
                clf.fit(stacked_fx_full, total_next_dyn_full.T)

                coeff_step = np.array(clf.coef_)

                # take gradient
                if len(coeffs_list) > 0:
                    coeffs = coeffs_list[-1] + 0.4 * \
                        (coeff_step - coeffs_list[-1])
                else:
                    coeffs = np.ones((len(F), 1))

            elif params_update_c['update_c_type'].lower() == 'spgl1':
                # print('spgl1')
                Aop = pylops.MatrixMult(stacked_fx_full)
                coeffs = pylops.optimization.sparsity.SPGL1(Aop, total_next_dyn_full.flatten(), iter_lim=params_update_c['num_iters'],
                                                            tau=params_update_c['reg_term'])[0]

            else:

                raise NameError('Unknown update c type')
            coeffs_list.append(coeffs.flatten())
        coeffs_final = np.vstack(coeffs_list)

        return coeffs_final.T

    def create_next(self, latent_dyn, coefficients, F, controls, B, time_point):
        """
        This function evaluate the dynamics at t+1 given the value of the dynamics at time t, the sub-dynamics, and other model parameters
        Inputs:
            latent_dyn    = the latent dynamics (can be either ground truth or estimated). [k X T]
            coefficients  = the sub-dynamics coefficients (used by the model)
            F             = a list of np.arrays, each np.array is a sub-dynamic with size kXk
            time_point    = current time point
            order         = how many time points in the future we want to estimate
        Outputs:
            k X 1 np.array describing the dynamics at time_point+1

        """
        if isinstance(F[0], list):
            F = [np.array(f_i) for f_i in F]

        if latent_dyn.shape[1] > 1:
            if controls is not None:
                cur_A = np.dstack([coefficients[i, time_point]*f_i @
                                   latent_dyn[:, time_point] + B @ controls[:, time_point] for i, f_i in enumerate(F)]).sum(2).T
            else:
                cur_A = np.dstack([coefficients[i, time_point]*f_i @
                                   latent_dyn[:, time_point] for i, f_i in enumerate(F)]).sum(2).T
        else:
            cur_A = np.dstack([coefficients[i, time_point]*f_i @
                               latent_dyn + B @ controls[:, time_point] for i, f_i in enumerate(F)]).sum(2).T
        return cur_A

    def create_ci_fi_xt(self, latent_dyn, F, coefficients, controls, B, cumulative=False, mute_infs=10**50,
                        max_inf=10**60):
        """
        An intermediate step for the reconstruction -
        Specifically - It calculated the error that should be taken in the GD step for updating f:
        f - eta * output_of(create_ci_fi_xt)
        output:
            3d array of the gradient step (unweighted): [k X k X time]
        """

        if max_inf <= mute_infs:
            raise ValueError('max_inf should be higher than mute-infs')
        curse_dynamics = latent_dyn

        all_grads = []
        for time_point in np.arange(latent_dyn.shape[1]-1):
            if cumulative:
                if time_point > 0:
                    previous_A = cur_A
                else:
                    previous_A = curse_dynamics[:, 0]
                cur_A = self.create_next(np.reshape(
                    previous_A, [-1, 1]), coefficients, F, controls, B, time_point)
            else:
                cur_A = self.create_next(
                    curse_dynamics, coefficients, F, controls, B, time_point)
            next_A = latent_dyn[:, time_point+1]

            """
        The actual step
        """

            if cumulative:
                gradient_val = (next_A - cur_A) @ previous_A.T
            else:
                gradient_val = (
                    next_A - cur_A) @ curse_dynamics[:, time_point].T
            all_grads.append(gradient_val)
        return np.dstack(all_grads)

    def update_f_all(self, latent_dyn, F, coefficients, controls, B, step_f, normalize=False, acumulated_error=False,
                     action_along_time='mean', weights_power=1.2, normalize_eig=True):
        """
        Update all the sub-dynamics {f_i} using GD
        """

        if action_along_time == 'mean':

            all_grads = self.create_ci_fi_xt(
                latent_dyn, F, coefficients, controls, B)  # this is the gradient step
            new_f_s = [self.norm_mat(f_i-2*step_f*self.norm_mat(np.mean(all_grads[:, :, :]*np.reshape(coefficients[i, :], [
                1, 1, -1]), 2), to_norm=normalize), to_norm=normalize_eig) for i, f_i in enumerate(F)]  # we multiply all the gradients by the coefficients because
        elif action_along_time == 'median':
            all_grads = self.create_ci_fi_xt(
                latent_dyn, F, coefficients, controls, B)

            new_f_s = [self.norm_mat(f_i-2*step_f*self.norm_mat(np.median(all_grads[:, :, :]*np.reshape(coefficients[i, :], [
                1, 1, -1]), 2), to_norm=normalize), to_norm=normalize_eig) for i, f_i in enumerate(F)]

        else:
            raise NameError(
                'Unknown action along time. Should be mean or median')
        for f_num in range(len(new_f_s)):
            rand_mat = np.random.rand(
                new_f_s[f_num].shape[0], new_f_s[f_num].shape[1])
            new_f_s[f_num][np.isnan(new_f_s[f_num])] = rand_mat[np.isnan(
                new_f_s[f_num])] .flatten()

        return new_f_s

    def spec_corr(self, v1, v2):
        """
        absolute value of correlation
        """
        corr = np.corrcoef(v1[:], v2[:])
        return np.abs(corr[0, 1])

    # %% Plotting functions

    def __str__(self):
        # we just want to print all attributes
        return str(self.__dict__.keys())

    def initialize(self, size_mat, r_seed=0, dist_type='norm', init_params={'loc': 0, 'scale': 1}, normalize=False):
        """
        !!!TODO: TAKEN FROM main_functions, need to revisit this

        This is an initialization function to initialize matrices like G_i and c.
        Inputs:
        size_mat    = 2-element tuple or list, describing the shape of the mat
        r_seed      = random seed (should be integer)
        dist_type   = distribution type for initialization; can be 'norm' (normal dist), 'uni' (uniform dist),'inti', 'sprase', 'regional'
        init_params = a dictionary with params for initialization. The keys depends on 'dist_type'.
                        keys for norm -> ['loc','scale']
                        keys for inti and uni -> ['low','high']
                        keys for sparse -> ['k'] -> number of non-zeros in each row
                        keys for regional -> ['k'] -> repeats of the sub-dynamics allocations
        normalize   = whether to normalize the matrix
        Output:
            the random matrix with size 'size_mat'
        """
        np.random.seed(r_seed)
        if dist_type == 'norm':
            # print('init_params:', init_params)
            # print('size_mat:', size_mat)
            rand_mat = np.random.normal(
                loc=init_params['loc'], scale=init_params['scale'], size=size_mat)

        if normalize:
            rand_mat = self.norm_mat(rand_mat)

        return rand_mat

    def norm_mat(self, mat, type_norm='evals', to_norm=True):
        """
        This function comes to norm matrices.
        Inputs:
            mat       = the matrix to norm
            type_norm = what type of normalization to apply. Can be:
                - 'evals' - normalize by dividing by the max eigen-value
                - 'max'   - divide by the maximum abs value in the matrix
                - 'exp'   -  normalization using matrix exponential (matrix exponential)
            to_norm   = whether to norm or not to.
        Output:
            the normalized matrix
        """
        if to_norm:
            if type_norm == 'evals':
                eigenvalues, _ = linalg.eig(mat)
                mat = mat / np.max(np.abs(eigenvalues))
            elif type_norm == 'max':
                mat = mat / np.max(np.abs(mat))
            elif type_norm == 'exp':
                mat = np.exp(-np.trace(mat))*linalg.expm(mat)
        return mat


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
