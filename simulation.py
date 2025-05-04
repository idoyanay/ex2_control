import numpy as np
import matplotlib.pyplot as plt


num_steps = 50
delta_t = 1.0
u = np.array([[0]])

# Define system matrices
F = np.array([[1, delta_t],
              [0, 1]])               # State transition matrixB = np.array([[0.5], [1]])     
H = np.array([[1, 0]])               # Measurement matrix
Q = np.eye(2)                        # Process noise covariance
R = np.array([[3]])                  # Measurement noise covariance
B = np.array([[0], [1]])
G = np.eye(2)                        # Noise input matrix
S = np.zeros((2, 1))                 # Cross-covariance (assumed zero)


# Initial state and covariance
x0 = np.array([[0], [1]]) 
P0 = np.eye(2)                        # x0 covariance
u = np.array([[1]])  

# init arrays for saving the process
x_true_hist = []
x_hat_hist = []
y_hist = []

x_true_hist.append(x0.flatten()) 
x_hat_hist.append(x0.flatten())
x_hat_i = x0        # in the kalman filter senario, the initial state is known
true_x = x0

Pi = P0     # initial condition for the error covariance
for i in range(num_steps):
    Yi = H @ true_x + np.random.normal(0, np.sqrt(R))   # taking the current output of the system
    ei = Yi - H @ x_hat_i                       # updaiting the error i
    y_hist.append(Yi.item())                
    Ki = (F @ Pi @ H.T + G @ S) @ np.linalg.inv(H @ Pi @ H.T + R) # updaiting the K(p, i)

    x_hat_i = F @ x_hat_i + Ki @ ei                                   # updaiting the x_hat_i
    x_hat_hist.append(x_hat_i.flatten())                          # saving the x_hat_i

    w = np.random.multivariate_normal([0, 0], Q).reshape(2, 1)    # simulate the noise w
    true_x = F @ true_x + B @ u + G @ w                           # updaiting the true x_i
    x_true_hist.append(true_x.flatten())        # saving the true x_i


    Rei = H @ Pi @ H.T + R                                        # updaiting the R(i)
    Pi = F @ Pi @ F.T + G.T @ Q @ G.T - Ki @ Rei @ Ki.T           # updaiting the error covariance


x_true_hist = np.array(x_true_hist)
x_hat_hist = np.array(x_hat_hist)
y_hist = np.array(y_hist)
# --- Plotting ---
plt.figure(figsize=(12, 5))

# Position plot
plt.subplot(1, 2, 1)
plt.plot(x_true_hist[:, 0], label='True Position')
plt.plot(x_hat_hist[:, 0], label='Estimated Position')
plt.plot(y_hist, 'o', label='Measurements', markersize=3, alpha=0.5)
plt.title('Position')
plt.xlabel('Time step')
plt.ylabel('Position')
plt.legend()

# Velocity plot
plt.subplot(1, 2, 2)
plt.plot(x_true_hist[:, 1], label='True Velocity')
plt.plot(x_hat_hist[:, 1], label='Estimated Velocity')
plt.title('Velocity')
plt.xlabel('Time step')
plt.ylabel('Velocity')
plt.legend()

plt.tight_layout()
plt.savefig('kalman_filter_q_b.png')
plt.show()