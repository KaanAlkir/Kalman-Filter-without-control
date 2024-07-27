import numpy as np
import matplotlib.pyplot as plt

# Implemented Matrices
A = np.array([[0.8, 0, 0, 0 , 0.2],
              [0, 0.1, 0.1, 0, 0],
              [0, 0, 0.3, 0, 0.1],
              [0, 0, 0, 0.1, 0],
              [0.1 ,0.2 ,0 ,0 ,0 ]])

C = np.array([0.1, 0, 0, 0, 0.2])

# Defining Parameters for the given conditions
I = np.eye(np.size(A, 0))
x = np.random.normal(0, 1, (np.size(A, 0), 1))
sigma = np.eye(np.size(A, 0))
M = np.zeros((np.size(A, 0), 1))

# Defining arrays to track the norms
norm_x = []
norm_M = []
norm_xm = []
norm_sigma = []

# Loop limitation
T = 100
N = 0

# Transpose of C
C_T = C.reshape(-1, 1)

# Kalman Filter Upgrade
while N < T:
    N = N + 1
    v_t = np.random.normal(0, 1, 1)
    w_t = np.random.normal(0, 1, (np.size(A, 0), 1))
    y = C @ x + v_t
    x = A @ x + w_t
    M = A @ M + sigma @ C_T * ((y - C @ A @ M) / (C @ sigma @ C_T + 1))
    sigma = A @ sigma @ A.T + I - A @ sigma @ C_T * ((C * sigma @ A.T) / (C @ sigma @ C_T + 1))
    norm_x.append((np.linalg.norm(x)).item())
    norm_M.append((np.linalg.norm(M)).item())
    norm_xm.append((np.linalg.norm(M - x)).item())
    norm_sigma.append(np.linalg.norm(sigma))

# following is for plotting
plt.plot(range(T), norm_sigma, color='black')
plt.plot(range(T), norm_xm, color='darkblue')
plt.plot(range(T), norm_x, color='orange')
plt.plot(range(T), norm_M, color='mediumseagreen')
plt.title('Norms')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.show()