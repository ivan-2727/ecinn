import tensorflow as tf
import numpy as np
from draw import draw_comparison
from tensorflow.keras.callbacks import LambdaCallback

class AdaptiveWeights(tf.keras.callbacks.LambdaCallback):
    def __init__(self, ecinn, experiments, derived_params, num_test_samples, weights):
        super().__init__()
        self.ecinn = ecinn
        self.num_test_samples = num_test_samples
        self.derived_params = derived_params
        self.experiments = experiments
        self.weights = weights
        self.frames = []

    def on_epoch_end(self, epoch, logs):
        self.ecinn.model.save_weights("model.weights.h5")
        predictions = self.ecinn.predict(
            num_test_samples = self.num_test_samples,
            derived_params = self.derived_params,
            sigmas = [e.sigma for e in self.experiments]
        )
        draw_comparison(epoch, predictions, self.experiments)
        if epoch > 50:
            for w in self.weights:
                if w < 1:
                    w.assign_add(5e-2)