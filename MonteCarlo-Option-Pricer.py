import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class MonteCarloOptionPricer:
    """
    European option pricer using Monte Carlo simulation.
    """
    def __init__(self, S0: float, K: float, T: float, r: float, sigma: float, num_simulations: int = 100000):
        self.S0 = S0                # Initial asset price
        self.K = K                  # Strike price
        self.T = T                  # Time to maturity (in years)
        self.r = r                  # Risk-free interest rate
        self.sigma = sigma          # Asset volatility
        self.num_simulations = num_simulations

    def price_european_option(self, option_type: str = 'call') -> tuple:
        """
        Calculates the price of a European option and its standard error.
        """
        # Generate standard normal random variables
        Z = np.random.standard_normal(self.num_simulations)
        
        # Simulate price at maturity (vectorized)
        ST = self.S0 * np.exp((self.r - 0.5 * self.sigma**2) * self.T + self.sigma * np.sqrt(self.T) * Z)
        
        # Calculate payoff
        if option_type.lower() == 'call':
            payoffs = np.maximum(ST - self.K, 0)
        elif option_type.lower() == 'put':
            payoffs = np.maximum(self.K - ST, 0)
        else:
            raise ValueError("Option type must be 'call' or 'put'")
            
        # Discount to present value
        discount_factor = np.exp(-self.r * self.T)
        price = discount_factor * np.mean(payoffs)
        
        # Calculate standard error (for confidence interval)
        std_error = discount_factor * np.std(payoffs) / np.sqrt(self.num_simulations)
        
        return price, std_error

    def plot_simulations(self, num_paths_to_plot: int = 100, steps: int = 252):
        """
        Simulates and plots the time evolution of multiple paths.
        """
        dt = self.T / steps
        paths = np.zeros((steps + 1, num_paths_to_plot))
        paths[0] = self.S0
        
        for t in range(1, steps + 1):
            Z = np.random.standard_normal(num_paths_to_plot)
            paths[t] = paths[t-1] * np.exp((self.r - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * Z)
            
        plt.figure(figsize=(10, 6))
        plt.plot(paths)
        plt.axhline(y=self.K, color='r', linestyle='--', label=f'Strike (K={self.K})')
        plt.title(f'Monte Carlo Simulation: {num_paths_to_plot} Geometric Brownian Motion Paths')
        plt.xlabel('Time Steps')
        plt.ylabel("Underlying Asset Price")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


class LongstaffSchwartzPricer:
    """
    American option pricer via Monte Carlo simulation 
    using the Longstaff-Schwartz algorithm (LSMC).
    """
    def __init__(self, S0: float, K: float, T: float, r: float, sigma: float, 
                 num_simulations: int = 50000, steps: int = 100):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.num_simulations = num_simulations
        self.steps = steps

    def _generate_paths(self) -> np.ndarray:
        """
        Generates underlying asset paths (Geometric Brownian Motion).
        Returns a matrix of size (steps + 1, num_simulations).
        """
        dt = self.T / self.steps
        paths = np.zeros((self.steps + 1, self.num_simulations))
        paths[0] = self.S0
        
        # Vectorized generation of all random shocks
        Z = np.random.standard_normal((self.steps, self.num_simulations))
        
        for t in range(1, self.steps + 1):
            paths[t] = paths[t-1] * np.exp((self.r - 0.5 * self.sigma**2) * dt + 
                                           self.sigma * np.sqrt(dt) * Z[t-1])
        return paths

    def price_american_put(self, poly_degree: int = 2) -> float:
        """
        Calculates the price of an American Put via Longstaff-Schwartz.
        
        :param poly_degree: Polynomial degree for continuation value regression.
        """
        paths = self._generate_paths()
        dt = self.T / self.steps
        discount_factor = np.exp(-self.r * dt)
        
        # Initialize cash flow vector (payoffs at maturity)
        # V will hold the optimal future cash flow for each path
        V = np.maximum(self.K - paths[-1], 0)
        
        # Backward induction from T-1 to 1 (excluding t=0 which is discounted at the end)
        for t in range(self.steps - 1, 0, -1):
            # Discount future cash flow by one time step
            V = V * discount_factor
            
            # Immediate exercise value at time t
            immediate_exercise = np.maximum(self.K - paths[t], 0)
            
            # We only perform regression on In-The-Money (ITM) paths
            # because exercising Out-Of-The-Money makes no financial sense.
            itm_indices = np.where(immediate_exercise > 0)[0]
            
            if len(itm_indices) > 0:
                X_itm = paths[t, itm_indices]
                Y_itm = V[itm_indices]
                
                # Polynomial regression: Y ~ a + bX + cX^2 ...
                # to estimate the conditional expectation E[V | X]
                coeffs = np.polyfit(X_itm, Y_itm, poly_degree)
                continuation_value = np.polyval(coeffs, X_itm)
                
                # Decision rule: exercise if Immediate Value > Continuation Value
                exercise_decision = immediate_exercise[itm_indices] > continuation_value
                
                # For paths where we exercise, the optimal future cash flow becomes the immediate payoff
                exercised_paths = itm_indices[exercise_decision]
                V[exercised_paths] = immediate_exercise[exercised_paths]
                
        # Final discount to t=0 and average
        V = V * discount_factor
        american_price = np.mean(V)
        
        return american_price


def calculate_price_and_greeks_mc(S0, K, T, r, sigma, option_type='call', num_simulations=500000, eps=0.01):
    """
    Calculates price, Delta, and Gamma using central finite differences 
    with Common Random Numbers (CRN) for variance reduction.
    """
    # The shock (bump): e.g., 1% of initial price
    h = S0 * eps
    
    # 1. Generate a SINGLE matrix of shocks to reduce variance (CRN)
    Z = np.random.standard_normal(num_simulations)
    
    # 2. Internal function to calculate vectorized payoff based on starting point
    def get_payoffs(S_start):
        ST = S_start * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        if option_type == 'call':
            return np.maximum(ST - K, 0)
        else:
            return np.maximum(K - ST, 0)

    # 3. Simulate the 3 starting scenarios using the SAME dynamic Z
    payoffs_up = get_payoffs(S0 + h)
    payoffs_base = get_payoffs(S0)
    payoffs_down = get_payoffs(S0 - h)
    
    # 4. Discount and average
    discount = np.exp(-r * T)
    V_up = discount * np.mean(payoffs_up)
    V_base = discount * np.mean(payoffs_base)
    V_down = discount * np.mean(payoffs_down)
    
    # 5. Apply central finite differences
    delta = (V_up - V_down) / (2 * h)
    gamma = (V_up - 2 * V_base + V_down) / (h**2)
    
    return V_base, delta, gamma


# ==========================================
# Execution and Examples
# ==========================================
if __name__ == "__main__":
    # Market parameters
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    
    print("="*50)
    print("1. EUROPEAN PRICER (Standard Monte Carlo)")
    print("="*50)
    N_euro = 100000
    euro_pricer = MonteCarloOptionPricer(S0, K, T, r, sigma, num_simulations=N_euro)
    call_price, call_err = euro_pricer.price_european_option('call')
    print(f"European Call Price : {call_price:.4f} (Std Error: ±{call_err:.4f})")
    
    print("\n" + "="*50)
    print("2. AMERICAN PRICER (Longstaff-Schwartz)")
    print("="*50)
    N_amer, steps = 100000, 100
    ls_pricer = LongstaffSchwartzPricer(S0, K, T, r, sigma, num_simulations=N_amer, steps=steps)
    american_put = ls_pricer.price_american_put(poly_degree=2)
    
    # Calculate standard European Put on the same terminal paths for comparison
    paths = ls_pricer._generate_paths()
    european_put = np.mean(np.maximum(K - paths[-1], 0)) * np.exp(-r * T)
    print(f"European Put Price  : {european_put:.4f}")
    print(f"American Put Price  : {american_put:.4f}")
    print(f"Early Exercise Premium : {american_put - european_put:.4f}")
    
    print("\n" + "="*50)
    print("3. GREEKS COMPUTATION (Finite Differences with CRN)")
    print("="*50)
    price, delta, gamma = calculate_price_and_greeks_mc(S0, K, T, r, sigma, 'call')
    
    # Analytical Black-Scholes formulas for verification
    d1 = (np.log(S0/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    bs_delta = norm.cdf(d1)
    bs_gamma = norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
    
    print(f"MC Price  : {price:.4f}")
    print(f"MC Delta  : {delta:.4f}  (BS Theoretical: {bs_delta:.4f})")
    print(f"MC Gamma  : {gamma:.5f} (BS Theoretical: {bs_gamma:.5f})")

    # Uncomment below to visualize the paths
    # euro_pricer.plot_simulations(num_paths_to_plot=100, steps=252)