import unittest
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.linear_algebra.lanczos_svd import LanczosSVD

class TestLanczosSVD(unittest.TestCase):
    def setUp(self):
        # Create a random sparse matrix
        np.random.seed(42)
        m, n = 50, 30
        self.A = sp.rand(m, n, density=0.1, format='csr')
        self.k = 5
        
    def test_singular_values(self):
        print("\nTesting Singular Values...")
        # 1. Custom SVD
        # Use more iterations to ensure convergence for testing correctness
        lanczos = LanczosSVD(n_components=self.k, max_iter=80) 
        lanczos.fit(self.A)
        s_custom = lanczos.singular_values_
        
        # 2. Scipy SVDs
        u, s_scipy, vt = svds(self.A, k=self.k)
        # svds returns ascending
        s_scipy = s_scipy[::-1]
        
        print(f"Custom S: {s_custom}")
        print(f"Scipy S: {s_scipy}")
        
        # Check if values are close
        # Note: Lanczos might not converge perfectly for all values with few iterations,
        # but for top k with sufficient iterations it should be close.
        # Allow some tolerance.
        np.testing.assert_allclose(s_custom, s_scipy, rtol=1e-1, err_msg="Singular values mismatch")
        
    def test_reconstruction(self):
        print("\nTesting Reconstruction...")
        lanczos = LanczosSVD(n_components=self.k, max_iter=30) # More iter for better precision
        lanczos.fit(self.A)
        
        U = lanczos.U_
        S = np.diag(lanczos.singular_values_)
        Vt = lanczos.components_
        
        # A_approx = U * S * Vt
        A_approx = U.dot(S).dot(Vt)
        
        # Calculate reconstruction error on the projected subspace
        # Note: A_approx is rank-k approx.
        # Let's compare with scipy rank-k approx
        
        u_s, s_s, vt_s = svds(self.A, k=self.k)
        A_scipy = u_s.dot(np.diag(s_s)).dot(vt_s)
        
        diff = np.linalg.norm(A_approx - A_scipy)
        print(f"Difference between Custom and Scipy reconstruction: {diff}")
        
        self.assertLess(diff, 1.0, "Reconstruction diverges significantly from Reference SVD")

    def test_orthogonality(self):
        print("\nTesting Orthogonality...")
        lanczos = LanczosSVD(n_components=self.k, max_iter=20)
        lanczos.fit(self.A)
        
        U = lanczos.U_
        Vt = lanczos.components_
        
        # U^T U should be Identity
        # V V^T should be Identity (since V^T has orthonormal rows)
        
        I_U = U.T.dot(U)
        I_V = Vt.dot(Vt.T)
        
        identity = np.eye(self.k)
        
        np.testing.assert_allclose(I_U, identity, atol=1e-1, err_msg="U is not orthogonal")
        np.testing.assert_allclose(I_V, identity, atol=1e-1, err_msg="V is not orthogonal")

if __name__ == '__main__':
    unittest.main()

