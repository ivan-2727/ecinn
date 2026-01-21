import tensorflow as tf
import numpy as np
from misc import exp_flux_sampling
from copy import deepcopy as copy 

class DataGenerator(tf.keras.utils.Sequence):
    """Data generator to generate training data on the fly, refactored with arrays/lists."""

    def __init__(self, outerBoundary,experiments,derived_params,
                 PortionAnalyzed=0.75, Lambda=3.0, num_train_samples=10**6, batch_size=250):
        super().__init__()

        self.num_train_samples = num_train_samples
        self.batch_size = batch_size
        self.Lambda = Lambda
        self.PortionAnalyzed = PortionAnalyzed
        self.theta_i = derived_params.theta_i
        self.theta_v = derived_params.theta_v
        self.outerBoundary = outerBoundary

        self.num_datasets = len(experiments)

        # --- 1. Load experimental data and parameters into lists ---
        self.experiments = experiments

        self.FullScanT = [2.0 * abs(self.theta_v - self.theta_i) / e.sigma for e in self.experiments]
        self.maxT = [T * PortionAnalyzed for T in self.FullScanT]
        self.time_array = [np.linspace(0, T, num=num_train_samples) for T in self.maxT]
        for arr in self.time_array:
            np.random.shuffle(arr)
        self.maxX = [Lambda * np.sqrt(T) for T in self.maxT]

    # --- 2. Helper getters ---
    def getFullScanT(self):
        return copy(self.FullScanT)

    def getMaxT(self):
        return copy(self.maxT)

    def getMaxX(self):
        return copy(self.maxX)

    def __len__(self):
        return int(np.floor(self.num_train_samples / self.batch_size))

    # --- 3. Generate batch ---
    def __getitem__(self, index):
        TX_eqn = []
        TX_ini = []
        TX_bnd0 = []
        TX_bnd1 = []
        theta_aux = []

        # --- 3a. Inputs for each dataset ---
        for i in range(self.num_datasets):
            # Equation domain
            tx = np.random.rand(self.batch_size, 2)
            tx[..., 0] *= self.maxT[i]
            tx[..., 1] *= self.maxX[i]
            TX_eqn.append(tx)

            # Initial condition at T=0
            tx_ini = np.random.rand(self.batch_size, 2)
            tx_ini[..., 0] = 0.0
            tx_ini[..., 1] *= self.maxX[i]
            TX_ini.append(tx_ini)

            # Electrode surface (X=0)
            tx_bnd0_i = np.random.rand(self.batch_size, 2)
            tx_bnd0_i[..., 0] = np.sort(self.time_array[i][index * self.batch_size:(index + 1) * self.batch_size])
            tx_bnd0_i[..., 1] = 0.0
            TX_bnd0.append(tx_bnd0_i)

            # Outer boundary
            tx_bnd1_i = np.random.rand(self.batch_size, 2)
            tx_bnd1_i[..., 0] *= self.maxT[i]
            tx_bnd1_i[..., 1] = self.maxX[i]
            TX_bnd1.append(tx_bnd1_i)

            # Applied potentials
            th = np.random.rand(self.batch_size, 1)
            th[:, 0] = np.where(
                tx_bnd0_i[..., 0] < self.FullScanT[i] / 2.0,
                self.theta_i - self.experiments[i].sigma * tx_bnd0_i[..., 0],
                self.theta_v + self.experiments[i].sigma * (tx_bnd0_i[..., 0] - self.FullScanT[i] / 2.0)
            )
            theta_aux.append(th)

        # --- 3b. Interpolated fluxes from experimental data ---
        interpolated_flux = [
            exp_flux_sampling(TX_bnd0[i][..., 0], self.experiments[i].pot_flux, self.FullScanT[i], self.PortionAnalyzed)
            for i in range(self.num_datasets)
        ]

        # --- 3c. Targets ---
        C_eqn = [np.zeros((self.batch_size, 1)) for _ in range(self.num_datasets)]
        C_ini = [np.ones((self.batch_size, 1)) for _ in range(self.num_datasets)]
        C_BV_bnd0 = [np.zeros((self.batch_size, 1)) for _ in range(self.num_datasets)]

        if self.outerBoundary == 'SI':
            C_bnd1 = [np.ones((self.batch_size, 1)) for _ in range(self.num_datasets)]
        elif self.outerBoundary == 'TL':
            C_bnd1 = [np.zeros((self.batch_size, 1)) for _ in range(self.num_datasets)]
        else:
            raise ValueError("Unknown outerBoundary")

        # --- 4. Flatten inputs and outputs for Keras ---
        x_train = []
        y_train = []

        for i in range(self.num_datasets):
            x_train.extend([TX_eqn[i], TX_ini[i], TX_bnd0[i], TX_bnd1[i], theta_aux[i]])
            y_train.extend([
                C_eqn[i],
                C_ini[i],
                C_BV_bnd0[i],
                interpolated_flux[i],
                C_bnd1[i]
            ])
         
        # for i, xi in enumerate(x_train):
        #     xi = np.asarray(xi)
        #     print(
        #         f"Input {i}: shape={xi.shape}, "
        #         f"std={xi.std():.3e}, min={xi.min():.3e}, max={xi.max():.3e}"
        #     )
        # for i, yi in enumerate(y_train):
        #     yi = np.asarray(yi)
        #     print(
        #         f"Output {i}: shape={yi.shape}, "
        #         f"std={yi.std():.3e}, min={yi.min():.3e}, max={yi.max():.3e}"
        #     )
        # input()

        return tuple(x_train), tuple(y_train)
