from numpy import dot, zeros, ones
from numpy.linalg import inv


class KalmanFilter:

    def __init__(self, X, F, H, P, Q, R, model='multivariate'):
        self.model = model
        self.x = X              # posteriori state estimate
        self.z = zeros(X.shape) #actual measurement
        self.d = zeros(X.shape)
        self.transition = F     # transition matrix
        self.measurement = H    # measurement function
        self.covariance_P = P   # posteriori estimate error covariance
        self.covariance_Q = Q   # process noise covariance
        self.covariance_R = R   # measurement noise covariance

    def predict(self):
        if self.model == 'multivariate':
            self.x = dot(self.transition, self.x)
            # P(k) = F(k) * P(k) * (F(k))^T + Q
            self.covariance_P = dot(self.transition, self.covariance_P).dot(self.transition.T) + self.covariance_Q
        else:
            self.x = self.x
            self.covariance_P += self.covariance_Q

    def update(self, d):
        # prediction update
        if self.model == 'multivariate':
            # H*P*H^T + R
            S = dot(self.measurement, self.covariance_P).dot(self.measurement.T) + self.covariance_R
            # measurement update
            K = dot(self.covariance_P, self.measurement.T).dot(inv(S))  # gain K(k) = P(k)*H^T*S^-1
            # y(k) = z(k) - H*x_k|k-1
            y = d - dot(self.measurement, self.x)
            # x(k) = x_k|k-1 + K(k) * y(k)
            self.x += dot(K, y)
            # P(k) = (I - K(k)*H) * P(k)    delta update
            self.covariance_P -= dot(K, self.measurement).dot(self.covariance_P)
        else:
            self.z += d
            K = self.covariance_P / (self.covariance_P + self.covariance_R)
            self.x += K * (self.z - self.x)
            self.covariance_P *= ones(3) - K
            self.d = self.x - self.z
