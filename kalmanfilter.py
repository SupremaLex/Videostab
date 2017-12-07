# file kalmanfilter.py
from numpy import dot, zeros
from numpy.linalg import inv


class KalmanFilter1D:

    def __init__(self):
        self.x = 0            # posteriori state estimate
        self.covariance_P = 1   # posteriori estimate error covariance
        self.covariance_Q = 0.001   # process noise covariance
        self.covariance_R = 0.1**2   # measurement noise covariance

    def predict(self):
        self.x = self.x
        self.covariance_P += self.covariance_Q

    def update(self, z):
        K = self.covariance_P / (self.covariance_P + self.covariance_R)
        self.x += K * (z - self.x)
        self.covariance_P *= 1 - K

    def predict_and_update(self, z):
        self.predict()
        self.update(z)
        return self.x


class KalmanFilterND:

    def __init__(self, X, F, H, P, Q, R):
        self.x = X              # posteriori state estimate
        self.x_min = zeros(X.shape)
        self.y = zeros(X.shape)
        self.transition = F     # transition matrix
        self.measurement = H    # measurement function
        self.covariance_P = P   # posteriori estimate error covariance
        self.covariance_Q = Q   # process noise covariance
        self.covariance_R = R   # measurement noise covariance

    def predict(self):
        self.x = dot(self.transition, self.x)
        # P(k) = F(k) * P(k) * (F(k))^T + Q
        self.covariance_P = dot(self.transition, self.covariance_P).dot(self.transition.T) + self.covariance_Q

    def update(self, z):
        # prediction update
        # H*P*H^T + R
        S = dot(self.measurement, self.covariance_P).dot(self.measurement.T) + self.covariance_R
        # measurement update
        K = dot(self.covariance_P, self.measurement.T).dot(inv(S))  # gain K(k) = P(k)*H^T*S^-1
        # y(k) = z(k) - H*x_k|k-1
        y = z - dot(self.measurement, self.x)
        # x(k) = x_k|k-1 + K(k) * y(k)
        self.x += dot(K, y)
        # P(k) = (I - K(k)*H) * P(k)    delta update
        self.covariance_P -= dot(K, self.measurement).dot(self.covariance_P)

    def predict_and_update(self, z):
        self.predict()
        self.update(z)
        return self.x
