import tensorflow as tf

class GradientLayer(tf.keras.layers.Layer):

    def __init__(self, model, **kwargs):
        self.model = model
        super().__init__(**kwargs)

    def call(self, x):
        # with tf.GradientTape(persistent=True) as tape2:
        #     tape2.watch(x)
        #     with tf.GradientTape() as tape1:
        #         tape1.watch(x)
        #         u = self.model(x)
        #     du_dtx = tape1.gradient(u, x)
        #     du_dt = du_dtx[:, 0:1]
        #     du_dx = du_dtx[:, 1:2]
        # d2u_dx2 = tape2.gradient(du_dx, x)[:, 1:2]
        # return u, du_dt, du_dx, d2u_dx2

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
        self.lambda_K0 = tf.Variable(initial_value=1.0,trainable=True,name='lambda_K0',constraint=tf.keras.constraints.non_neg())
        self.lambda_alpha = tf.Variable(initial_value=0.4,trainable=True,name='lambda_alpha',constraint=tf.keras.constraints.non_neg())

    def call(self,theta_aux,u_bnd0):
        return self.lambda_K0*tf.exp(-self.lambda_alpha*theta_aux)*u_bnd0

class DiffusionCoefficientLayer(tf.keras.layers.Layer):
    
    def __init__(self):
        super().__init__()
        self.lambda_dA = tf.Variable(initial_value=0.4,trainable=True,name ='lambda_d_A',constraint=tf.keras.constraints.non_neg())

    def call(self,du_dx_bnd0,du_dt,d2u_dx2):
        flux = self.lambda_dA * du_dx_bnd0
        u_eqn = du_dt - self.lambda_dA*d2u_dx2
        return flux, u_eqn

