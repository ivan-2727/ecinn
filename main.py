from network import make_network
from myio import read_experiments
from data_generator import DataGenerator
import tensorflow as tf
from model import ECINN 
from misc import schedule
from draw import draw_comparison
from adaptive_weights import AdaptiveWeights
import numpy as np 
from network import make_network
import pandas as pd 
from tensorflow.keras.callbacks import LambdaCallback

num_test_samples = 1000
tot_var = 5
scan_rates = ["10"]
params, derived_params, experiments = read_experiments("data", "Fe-GCE", scan_rates)

networks = [make_network() for _ in range(len(scan_rates))]

generator = DataGenerator(
    outerBoundary="SI", 
    experiments=experiments, 
    derived_params=derived_params)

def frozen(i):
    return i%tot_var == 2

weights_list = [tf.Variable(0.0 if frozen(i) else 1.0, trainable=(not frozen(i))) for i in range(tot_var*len(networks))]

ecinn = ECINN(
    networks=networks, 
    generator=generator, 
    weights_list=weights_list, 
    outerBoundary="SI",
    saved='model.weights.h5')

tf.keras.utils.plot_model(ecinn.model, to_file='ECINN_model.png', rankdir="LR",show_shapes=True,expand_nested=True,dpi=200) #Requires installation of Graphviz and pydot

adaptive_weights = AdaptiveWeights(ecinn=ecinn, 
experiments=experiments, 
derived_params=derived_params, 
num_test_samples=num_test_samples, weights = [weights_list[i] for i in range(len(weights_list)) if frozen(i)])

ecinn.fit(
    epochs=300, verbose=2, 
    callbacks=[
        tf.keras.callbacks.LearningRateScheduler(schedule, verbose=0),
        adaptive_weights
    ])

predictions = ecinn.predict(
    num_test_samples = num_test_samples,
    derived_params = derived_params,
    sigmas = [e.sigma for e in experiments]
)
print("Predictions", predictions[0])
print("Experiment", experiments[0].pot_flux)
 