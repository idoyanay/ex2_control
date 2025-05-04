import numpy as np
import matplotlib.pyplot as plt
import argparse


## --- defaulte values for the matrices and parameters (each question will change the relevant parts if needed) --- ##
## ------------------------------------------------------------------------------------------------------------------- ##
num_steps = 50
delta_t = 1.0

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
u = np.array([[0]])  
## ------------------------------------------------------------------------------------------------------------------- ##





def parse_args():
    # get the question number from the command line
    parser = argparse.ArgumentParser(description='Kalman Filter')
    parser.add_argument('-q', '--question', type=int, default=1, help='Question number') # number the question from 1 to 4 for b-e questions and 5 for question g
    args = parser.parse_args()
    if args.question >5 or args.question < 1:
        print("Question number must be between 1 and 4")
    return args


def run_q_b():
    
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

def run_q_c():
        # init arrays for saving the process
    trace_P_hist = []
    trace_Kp_hist = []

    x_hat_i = x0        # in the kalman filter senario, the initial state is known
    true_x = x0

    Pi = P0     # initial condition for the error covariance
    for i in range(num_steps):
        Yi = H @ true_x + np.random.normal(0, np.sqrt(R))   # taking the current output of the system
        ei = Yi - H @ x_hat_i                       # updaiting the error i
        Ki = (F @ Pi @ H.T + G @ S) @ np.linalg.inv(H @ Pi @ H.T + R) # updaiting the K(p, i)

        x_hat_i = F @ x_hat_i + Ki @ ei                                   # updaiting the x_hat_i

        w = np.random.multivariate_normal([0, 0], Q).reshape(2, 1)    # simulate the noise w
        true_x = F @ true_x + B @ u + G @ w                           # updaiting the true x_i


        Rei = H @ Pi @ H.T + R                                        # updaiting the R(i)
        Pi = F @ Pi @ F.T + G.T @ Q @ G.T - Ki @ Rei @ Ki.T           # updaiting the error covariance

        trace_P_hist.append(np.trace(Pi))
        trace_Kp_hist.append(np.trace(Ki))


    trace_P_hist = np.array(trace_P_hist)
    trace_Kp_hist = np.array(trace_Kp_hist)
    # --- Plotting ---
    plt.figure(figsize=(12, 5))

    # Plot trace of Pi
    plt.subplot(1, 2, 1)
    plt.plot(trace_P_hist, label='Trace of P')
    plt.title('Trace of Error Covariance (P)')
    plt.xlabel('Time step')
    plt.ylabel('Trace(P)')
    plt.legend()

    # Plot trace of Kalman gain Kp
    plt.subplot(1, 2, 2)
    plt.plot(trace_Kp_hist, label='Trace of Kp')
    plt.title('Trace of Kalman Gain (Kp)')
    plt.xlabel('Time step')
    plt.ylabel('Trace(Kp)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('question_c.png')
    plt.show()

def run_q_d():
    num_runs = 1000
    # To store position estimation errors across runs
    kalman_errors = np.zeros((num_runs, num_steps))
    naive_errors = np.zeros((num_runs, num_steps))

    for run in range(num_runs):
        # Initial states
        x0 = np.array([[0], [1]])  # true initial state
        x_hat = x0.copy()          # initial Kalman estimate
        P = np.eye(2)              # initial error covariance

        true_x = x0.copy()

        for t in range(num_steps):
            # Simulate measurement
            y = H @ true_x + np.random.normal(0, np.sqrt(R))

            # --- Kalman Predictor ---
            e = y - H @ x_hat
            Re = H @ P @ H.T + R
            Kp = (F @ P @ H.T + G @ S) @ np.linalg.inv(Re)
            x_hat = F @ x_hat + Kp @ e
            P = F @ P @ F.T + G.T @ Q @ G.T - Kp @ Re @ Kp.T

            # --- Naive Estimator ---
            naive_pos = y.item()  # just take measurement as position estimate

            # --- Store squared errors (only for position) ---
            true_pos = true_x[0, 0]
            kalman_errors[run, t] = (x_hat[0, 0] - true_pos) ** 2
            naive_errors[run, t] = (naive_pos - true_pos) ** 2

            # --- True system update ---
            w = np.random.multivariate_normal([0, 0], Q).reshape(2, 1)
            true_x = F @ true_x + B @ u + G @ w

        # Compute error variance over all runs
        kalman_var = np.mean(kalman_errors, axis=0)
        naive_var = np.mean(naive_errors, axis=0)

    # --- Plotting ---
    plt.figure(figsize=(10, 5))
    plt.plot(kalman_var, label='Kalman Filter Error Variance (position)')
    plt.plot(naive_var, label='Naive Estimator Error Variance (position)')
    plt.title('Monte Carlo: Position Estimation Error Variance')
    plt.xlabel('Time step')
    plt.ylabel('Variance')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('question_d.png')
    plt.show()


def main(args):
    # get the question number from the command line
    question = args.question
    if question == 1:
        run_q_b()
    elif question == 2:
        run_q_c()
    elif question == 3:
        run_q_d()
    elif question == 4:
        run_q_e()
    elif question == 5:
        run_q_g()


if __name__ == "__main__":
    # parse arguments
    args = parse_args()
    main(args)