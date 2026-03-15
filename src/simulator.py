import numpy as np

class SIRSimulator:
    def __init__(self, N, beta, gamma):
        """
        Initializes the SIR simulator.
        
        Args:
            N (int): Total population size.
            beta (float): Infection rate.
            gamma (float): Recovery rate.
        """
        self.N = N
        self.beta = beta
        self.gamma = gamma

    def simulate_gillespie(self, S0, I0, R0, max_time=100):
        """
        Simulates an SIR epidemic using the Gillespie algorithm (stochastic).
        
        Args:
            S0 (int): Initial susceptible population.
            I0 (int): Initial infected population.
            R0 (int): Initial recovered population.
            max_time (float): Maximum simulation time.
            
        Returns:
            tuple: (times, S, I, R) lists.
        """
        t = 0
        S, I, R = S0, I0, R0
        
        times = [t]
        s_counts = [S]
        i_counts = [I]
        r_counts = [R]
        
        while t < max_time and I > 0:
            # Rates of events
            infection_rate = self.beta * S * I / self.N
            recovery_rate = self.gamma * I
            total_rate = infection_rate + recovery_rate
            
            if total_rate == 0:
                break
                
            # Time to next event
            dt = np.random.exponential(1 / total_rate)
            t += dt
            
            # Which event happens?
            p = np.random.random()
            if p < infection_rate / total_rate:
                # Infection event
                S -= 1
                I += 1
            else:
                # Recovery event
                I -= 1
                R += 1
                
            times.append(t)
            s_counts.append(S)
            i_counts.append(I)
            r_counts.append(R)
            
        return np.array(times), np.array(s_counts), np.array(i_counts), np.array(r_counts)

    def simulate_deterministic(self, S0, I0, R0, times):
        """
        Solves the deterministic SIR ODEs for comparison.
        dS/dt = -beta * S * I / N
        dI/dt = beta * S * I / N - gamma * I
        dR/dt = gamma * I
        """
        from scipy.integrate import odeint
        
        def sir_ode(y, t, N, beta, gamma):
            S, I, R = y
            dSdt = -beta * S * I / N
            dIdt = beta * S * I / N - gamma * I
            dRdt = gamma * I
            return [dSdt, dIdt, dRdt]
        
        y0 = [S0, I0, R0]
        solution = odeint(sir_ode, y0, times, args=(self.N, self.beta, self.gamma))
        return solution[:, 0], solution[:, 1], solution[:, 2]

    def interpolate_simulation(self, times, s_counts, i_counts, r_counts, num_points=100):
        """
        Interpolates stochastic simulation results onto a uniform time grid.
        """
        t_uniform = np.linspace(0, times[-1], num_points)
        s_interp = np.interp(t_uniform, times, s_counts)
        i_interp = np.interp(t_uniform, times, i_counts)
        r_interp = np.interp(t_uniform, times, r_counts)
        return t_uniform, s_interp, i_interp, r_interp

if __name__ == "__main__":
    # Quick test
    N = 1000
    beta = 0.3
    gamma = 0.1
    S0, I0, R0 = 999, 1, 0
    
    sim = SIRSimulator(N, beta, gamma)
    t, s, i, r = sim.simulate_gillespie(S0, I0, R0)
    
    print(f"Simulation finished at t={t[-1]:.2f}")
    print(f"Final counts: S={s[-1]}, I={i[-1]}, R={r[-1]}")
