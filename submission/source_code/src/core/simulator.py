import numpy as np


class SIRSimulator:
    def __init__(self, N, beta, gamma):
        self.N = N
        self.beta = beta
        self.gamma = gamma

    def simulate_gillespie(self, S0, I0, R0, max_time=100):
        t = 0
        S, I, R = S0, I0, R0

        times = [t]
        s_counts = [S]
        i_counts = [I]
        r_counts = [R]

        while t < max_time and I > 0:
            infection_rate = self.beta * S * I / self.N
            recovery_rate = self.gamma * I
            total_rate = infection_rate + recovery_rate

            if total_rate == 0:
                break

            dt = np.random.exponential(1 / total_rate)
            t += dt

            if np.random.random() < infection_rate / total_rate:
                S -= 1
                I += 1
            else:
                I -= 1
                R += 1

            times.append(t)
            s_counts.append(S)
            i_counts.append(I)
            r_counts.append(R)

        return np.array(times), np.array(s_counts), np.array(i_counts), np.array(r_counts)

    def simulate_deterministic(self, S0, I0, R0, times):
        """Solve the deterministic SIR ODEs using scipy for comparison/validation."""
        from scipy.integrate import odeint

        def sir_ode(y, t, N, beta, gamma):
            S, I, R = y
            dSdt = -beta * S * I / N
            dIdt = beta * S * I / N - gamma * I
            dRdt = gamma * I
            return [dSdt, dIdt, dRdt]

        solution = odeint(sir_ode, [S0, I0, R0], times, args=(self.N, self.beta, self.gamma))
        return solution[:, 0], solution[:, 1], solution[:, 2]

    def interpolate_simulation(self, times, s_counts, i_counts, r_counts, num_points=100):
        t_uniform = np.linspace(0, times[-1], num_points)
        s_interp = np.interp(t_uniform, times, s_counts)
        i_interp = np.interp(t_uniform, times, i_counts)
        r_interp = np.interp(t_uniform, times, r_counts)
        return t_uniform, s_interp, i_interp, r_interp
