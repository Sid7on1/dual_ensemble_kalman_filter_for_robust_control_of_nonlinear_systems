import numpy as np
from typing import List, Tuple, Dict
from dual_enkf import DualEnKF
from lyapunov_redesign import LyapunovRedesign
from pde_simulator import PDESimulator
from controlgym.utils import qplot
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustController:
    """
    Robust Controller class for nonlinear systems using dual EnKF approach.

    Attributes:
        system_dynamics (callable): Function that defines the system dynamics.
        measurement_function (callable): Function that defines the measurement process.
        process_noise (float): Standard deviation of process noise.
        measurement_noise (float): Standard deviation of measurement noise.
        num_ensembles (int): Number of ensembles for the dual EnKF.
        lambda_param (float): Weighting parameter for Lyapunov redesign.
        max_iterations (int): Maximum number of control iterations.
        tolerance (float): Tolerance for control convergence.
        control_bounds (List[float]): Lower and upper bounds on the control input.
    """

    def __init__(self, system_dynamics, measurement_function, process_noise, measurement_noise,
                 num_ensembles, lambda_param, max_iterations, tolerance, control_bounds):
        self.system_dynamics = system_dynamics
        self.measurement_function = measurement_function
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.num_ensembles = num_ensembles
        self.lambda_param = lambda_param
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.control_bounds = control_bounds
        self.enkf = DualEnKF(system_dynamics, measurement_function, process_noise, measurement_noise, num_ensembles)
        self.lyapunov = LyapunovRedesign(system_dynamics, lambda_param)

    def _update_state(self, x, u, t):
        """
        Update the system state using the given control input at time t.

        Args:
            x (numpy.ndarray): Current state vector.
            u (float): Control input.
            t (float): Current time.

        Returns:
            numpy.ndarray: Updated state vector.
        """
        # Simulate the system dynamics with the given control input
        x_dot = self.system_dynamics(x, u, t)
        x = x + x_dot * self.enkf.dt
        return x

    def _compute_optimal_control(self, x, t):
        """
        Compute the optimal control input using dual EnKF and Lyapunov redesign.

        Args:
            x (numpy.ndarray): Current state vector.
            t (float): Current time.

        Returns:
            float: Optimal control input.
        """
        # Initialize control input
        u = 0.0

        # Perform dual EnKF prediction and update steps
        self.enkf.predict(x, u, t)
        self.enkf.update(x, t)

        # Compute the innovation sequence
        innovation_sequence = self.enkf.get_innovation_sequence()

        # Compute the optimal control using Lyapunov redesign
        u = self.lyapunov.compute_control(x, innovation_sequence, self.enkf.P, t)

        # Apply control bounds
        u = np.clip(u, self.control_bounds[0], self.control_bounds[1])

        return u

    def run_robust_control(self, x0, t0, tf):
        """
        Run the robust control algorithm for the given system and time interval.

        Args:
            x0 (numpy.ndarray): Initial state vector.
            t0 (float): Initial time.
            tf (float): Final time.

        Returns:
            numpy.ndarray: Time history of states.
            numpy.ndarray: Time history of control inputs.
        """
        # Initialize time history arrays
        t = np.arange(t0, tf + self.enkf.dt, self.enkf.dt)
        num_steps = len(t)
        x_history = np.zeros((num_steps, x0.shape[0]))
        u_history = np.zeros(num_steps)

        # Set initial state
        x = x0

        # Main control loop
        for i, current_time in enumerate(t):
            # Compute optimal control input
            u = self._compute_optimal_control(x, current_time)

            # Update state using the computed control input
            x = self._update_state(x, u, current_time)

            # Store state and control history
            x_history[i, :] = x
            u_history[i] = u

            # Check for convergence
            if i > 0 and np.all(np.abs(x_history[i, :] - x_history[i-1, :]) < self.tolerance):
                logger.info("Convergence achieved at time step %d", i)
                break

            if i == num_steps - 1:
                logger.info("Maximum iterations reached without convergence.")

        return t, x_history, u_history

    def plot_results(self, t, x_history, u_history):
        """
        Plot the state and control history.

        Args:
            t (numpy.ndarray): Time history.
            x_history (numpy.ndarray): State history.
            u_history (numpy.ndarray): Control input history.
        """
        # Plot the state and control trajectories
        fig, axs = plt.subplots(2, 1, figsize=(10, 6))
        axs[0].plot(t, x_history)
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('State')
        axs[0].grid(True)
        axs[1].plot(t, u_history)
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Control Input')
        axs[1].grid(True)
        plt.tight_layout()
        plt.show()

def main():
    # Define system dynamics and measurement function
    system_dynamics = lambda x, u, t: np.array([x[1], -x[0] - np.sin(t) - u])
    measurement_function = lambda x, t: x

    # Set simulation parameters
    process_noise = 0.1
    measurement_noise = 0.1
    num_ensembles = 500
    lambda_param = 0.5
    max_iterations = 1000
    tolerance = 1e-4
    control_bounds = [-5.0, 5.0]

    # Initial conditions
    x0 = np.array([1.0, 0.0])
    t0 = 0.0
    tf = 10.0

    # Instantiate the robust controller
    controller = RobustController(system_dynamics, measurement_function, process_noise, measurement_noise,
                                 num_ensembles, lambda_param, max_iterations, tolerance, control_bounds)

    # Run the robust control algorithm
    t, x_history, u_history = controller.run_robust_control(x0, t0, tf)

    # Plot the results
    controller.plot_results(t, x_history, u_history)

if __name__ == '__main__':
    main()