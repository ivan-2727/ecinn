from dataclasses import dataclass
import numpy as np
import tensorflow as tf
import pandas as pd 
from layer import GradientLayer, BVLayerCathodic, DiffusionCoefficientLayer

@dataclass
class Prediction:
    lambda_K0: float 
    lambda_alpha: float 
    lambda_dA: float 
    pot_flux: pd.DataFrame 

class ECINN:

    def __init__(self, networks, generator, weights_list, outerBoundary, saved):

        self.networks = networks
        self.grads = [GradientLayer(network) for network in networks]
        self.BVLayer = BVLayerCathodic()
        self.DLayer = DiffusionCoefficientLayer()
        self.generator = generator         

        num_networks = len(self.networks)

        # --- 1. Create inputs ---
        tx_eqn = []
        tx_ini = []
        tx_bnd0 = []
        tx_bnd1 = []
        theta_aux = []

        for i in range(num_networks):
            tx_eqn.append(tf.keras.layers.Input(shape=(2,), name=f'Domain_Input_{i+1}'))
            tx_ini.append(tf.keras.layers.Input(shape=(2,), name=f'Ini_Input_{i+1}'))
            tx_bnd0.append(tf.keras.layers.Input(shape=(2,), name=f'Bnd0_Input_{i+1}'))
            tx_bnd1.append(tf.keras.layers.Input(shape=(2,), name=f'Bnd1_Input_{i+1}'))
            theta_aux.append(tf.keras.layers.Input(shape=(1,), name=f'Potential_Input_{i+1}'))

        u = []
        du_dt = []
        du_dx = []
        d2u_dx2 = []

        for i in range(num_networks):
            u_i, du_dt_i, du_dx_i, d2u_dx2_i = self.grads[i](tx_eqn[i])
            u.append(u_i)
            du_dt.append(du_dt_i)
            du_dx.append(du_dx_i)
            d2u_dx2.append(d2u_dx2_i)

        u_ini = [self.networks[i](tx_ini[i]) for i in range(num_networks)]

        u_bnd0 = []
        du_dt_bnd0 = []
        du_dx_bnd0 = []
        d2u_dx2_bnd0 = []

        for i in range(num_networks):
            u0, d0_dt, d0_dx, d02_dx2 = self.grads[i](tx_bnd0[i])
            u_bnd0.append(u0)
            du_dt_bnd0.append(d0_dt)
            du_dx_bnd0.append(d0_dx)
            d2u_dx2_bnd0.append(d02_dx2)

        diffusion_flux = []
        u_eqn = []
        predicted_flux = []

        for i in range(num_networks):
            flux, eqn = self.DLayer(du_dx_bnd0[i], du_dt[i], d2u_dx2[i])
            diffusion_flux.append(flux)
            u_eqn.append(eqn)
            predicted_flux.append(-flux)

        BV_flux = [self.BVLayer(theta_aux[i], u_bnd0[i]) for i in range(num_networks)]
        u_BV_bnd0 = [
            (diffusion_flux[i] - BV_flux[i]) for i in range(num_networks)
        ]

        if outerBoundary == 'SI':
            u_bnd1 = [self.networks[i](tx_bnd1[i]) for i in range(num_networks)]
        elif outerBoundary == 'TL':
            u_bnd1 = []
            for i in range(num_networks):
                u_i, du_dt_i, du_dx_i, d2u_dx2_i = self.grads[i](tx_bnd1[i])
                u_bnd1.append(u_i)
        else:
            raise ValueError(f"Unknown outerBoundary: {outerBoundary}")

        inputs, outputs = [], []
        for i in range(num_networks):
            inputs.extend([tx_eqn[i], tx_ini[i], tx_bnd0[i], tx_bnd1[i], theta_aux[i]])
            outputs.extend([
                u_eqn[i],
                u_ini[i],
                u_BV_bnd0[i],
                predicted_flux[i],
                u_bnd1[i]
            ])
        
        self.model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        if saved:
            self.model.load_weights(saved)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=4e-4,
        )
        self.model.compile(optimizer=optimizer, loss=tuple('mse' for _ in range(len(weights_list))), loss_weights=weights_list)         

    def fit(self, epochs, verbose, callbacks):
        self.model.fit(self.generator, epochs=epochs, verbose=verbose, callbacks=callbacks)

    def get_lambdas(self):
        return self.BVLayer.lambda_K0.numpy(), self.BVLayer.lambda_alpha.numpy(), self.DLayer.lambda_dA.numpy()

    def predict(self, num_test_samples, derived_params, sigmas):
        predictions = []
        maxT = self.generator.getMaxT()
        fullScanT = self.generator.getFullScanT()
        maxX = self.generator.getMaxX()
        for i in range(len(self.networks)):
            t_flat = np.linspace(0, maxT[i], num_test_samples)
            cv_flat = np.where(t_flat<fullScanT[i]/2.0, derived_params.theta_i-sigmas[i]*t_flat, derived_params.theta_v+sigmas[i]*(t_flat-fullScanT[i]/2.0))
            x_flat = np.linspace(0, maxX[i], num_test_samples) 
            t, x = np.meshgrid(t_flat, x_flat)
            tx = np.stack([t.flatten(), x.flatten()], axis=-1)
            c = self.networks[i].predict(tx, batch_size=num_test_samples)
            c = c.reshape(t.shape)
            TX_flux = np.zeros((len(t_flat),2))
            TX_flux[:,0] = t_flat
            TX_flux = tf.convert_to_tensor(TX_flux)
            with tf.GradientTape() as g:
                g.watch(TX_flux)
                C = self.networks[i](TX_flux)
            dC_dX = g.batch_jacobian(C,TX_flux)[...,1]
            flux = -self.DLayer.lambda_dA.numpy()*dC_dX.numpy().reshape(-1)
            predictions.append(pd.DataFrame({'Potential':cv_flat,'Flux':flux}))
        return predictions