import numpy as np

class RankSelector:
    @staticmethod
    def get_optimal_k(singular_values, m, n):
        """
        Determines optimal rank k using Gavish-Donoho method with unknown noise level.
        Args:
            singular_values (np.array): Array of singular values (descending).
            m (int): Number of rows of original matrix.
            n (int): Number of cols of original matrix.
        Returns:
            int: Optimal k.
        """
        if len(singular_values) == 0:
            return 1
            
        beta = min(m, n) / max(m, n)
        
        # Approximate omega(beta)
        # Based on Gavish & Donoho 2014, "The Optimal Hard Threshold for Singular Values is 4/sqrt(3)"
        # Formula 10 for unknown sigma: lambda_star * median(singular_values)
        # But we need the coefficient for the median scaling.
        # A common approximation is:
        omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
        
        # Threshold
        tau = omega * np.median(singular_values)
        
        # Count values above threshold
        k = np.sum(singular_values > tau)
        
        # Ensure at least 1 and not more than len
        return max(1, int(k))

