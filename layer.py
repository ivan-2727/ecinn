import tensorflow as tf

class GradientLayer(tf.keras.layers.Layer):

    def __init__(self, model, **kwargs):
        self.model = model
        super().__init__(**kwargs)

    def call(self, x):

        with tf.GradientTape() as g:
            g.watch(x)
            with tf.GradientTape() as gg:
                gg.watch(x)
                u = self.model(x)
            dc_dtx = gg.batch_jacobian(u, x)
            dc_dt = dc_dtx[..., 0]
            dc_dx = dc_dtx[..., 1]
        d2c_dx2 = g.batch_jacobian(dc_dx, x)[..., 1]
        return u, dc_dt, dc_dx, d2c_dx2

class BVLayerCathodic(tf.keras.layers.Layer):
    """
    Custom layer for BV kinetics for the cathodic scan. Only reduction matters.
    """
    def __init__(self,name='FluxLayer'):
        super().__init__(name=name)
        self.lambda_K0 = self.add_weight(
            name='lambda_K0',
            shape=(),
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=True,
            constraint=tf.keras.constraints.NonNeg()
        )

        self.lambda_alpha = self.add_weight(
            name='lambda_alpha',
            shape=(),
            initializer=tf.keras.initializers.Constant(0.4),
            trainable=True,
            constraint=tf.keras.constraints.NonNeg()
        )

    def call(self,theta_aux,u_bnd0):
        return self.lambda_K0*tf.exp(-self.lambda_alpha*theta_aux)*u_bnd0

class DiffusionCoefficientLayer(tf.keras.layers.Layer):
    
    def __init__(self):
        super().__init__()
        self.lambda_dA = self.add_weight(
            name='lambda_dA',
            shape=(),
            initializer=tf.keras.initializers.Constant(0.4),
            trainable=True,
            constraint=tf.keras.constraints.NonNeg()
        )

    def call(self,du_dx_bnd0,du_dt,d2u_dx2):
        flux = self.lambda_dA * du_dx_bnd0
        u_eqn = du_dt - self.lambda_dA*d2u_dx2
        return flux, u_eqn

