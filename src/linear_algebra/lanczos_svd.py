import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator

class LanczosSVD:
    def __init__(self, n_components=10, max_iter=None, tol=1e-8):
        self.n_components = n_components
        self.max_iter = max_iter if max_iter else n_components * 2 + 10
        self.tol = tol
        self.components_ = None # Vt
        self.singular_values_ = None
        self.U_ = None

    def _augment_matrix(self, A):
        """
        Constructs the augmented matrix H = [0  A]
                                            [A^T 0]
        """
        m, n = A.shape
        # Create zero blocks
        # We can use LinearOperator to avoid explicit construction if very large,
        # but for text summarization explicit sparse matrix is fine and faster for dev.
        
        # H is (m+n) x (m+n)
        # We use scipy.sparse.bmat
        
        # O_m = sp.csr_matrix((m, m)) # Actually unnecessary to store zeros explicitly if using bmat correctly
        # O_n = sp.csr_matrix((n, n))
        
        # Better to define LinearOperator to save memory if n is large (vocab size)
        # But let's try explicit sparse first as requested by "input: sparse matrix A".
        # Using bmat is efficient.
        
        H = sp.bmat([[None, A], [A.T, None]], format='csr')
        return H

    def fit(self, A):
        m, n = A.shape
        H = self._augment_matrix(A)
        dim = m + n
        
        # Lanczos Iterations
        k = self.max_iter
        
        # Storage for tridiagonal matrix T
        alphas = []
        betas = []
        
        # Storage for Lanczos vectors Q
        # We need to store them for reorthogonalization and final reconstruction
        Q = [] 
        
        # Initial random vector
        np.random.seed(42)
        q = np.random.randn(dim)
        q /= np.linalg.norm(q)
        
        Q.append(q)
        
        beta_prev = 0
        q_prev = np.zeros(dim)
        
        for j in range(k):
            # 1. Apply H
            z = H.dot(q)
            
            # 2. Orthogonalize against previous two (part of standard Lanczos)
            # alpha_j = q_j^T * H * q_j
            alpha = np.dot(q, z)
            alphas.append(alpha)
            
            z = z - alpha * q - beta_prev * q_prev
            
            # 3. Full Reorthogonalization (Gram-Schmidt against ALL previous Q)
            # Critical for numerical stability as requested
            for i, qi in enumerate(Q):
                proj = np.dot(z, qi)
                z = z - proj * qi
                
            # Double reorthogonalization is often used for better precision, 
            # but single pass loop is "full reorthogonalization" logic. 
            # Let's do a second pass if norm drops significantly? 
            # For simplicity and task scope, single full pass is usually sufficient "Full Reorth".
            
            # 4. Calculate new beta
            beta = np.linalg.norm(z)
            
            if beta < self.tol:
                # Invariant subspace found
                break
                
            betas.append(beta)
            
            # Update pointers
            q_prev = q
            q = z / beta
            beta_prev = beta
            
            Q.append(q)
            
        # Build Tridiagonal Matrix T
        # T is size (k)x(k) if we did k iterations.
        # alphas are diag, betas are off-diag.
        
        # We have len(alphas) = iterations. len(betas) = iterations or iterations-1.
        # If loop finished fully: len(betas) == k (last beta is beta_k for q_{k+1})
        # But T uses beta_0..beta_{k-1}. The last beta computed in loop corresponds to connection to q_{k+1}, 
        # which is outside T if we project to subspace Q_k.
        
        n_iter = len(alphas)
        T = np.zeros((n_iter, n_iter))
        np.fill_diagonal(T, alphas)
        np.fill_diagonal(T[1:, :], betas[:n_iter-1])
        np.fill_diagonal(T[:, 1:], betas[:n_iter-1])
        
        # Eigen decomposition of T
        # T is small, use eigh
        eigvals, eigvecs = np.linalg.eigh(T)
        
        # Sort by magnitude (descending)
        # For augmented matrix, eigenvalues are +/- sigma.
        # We want positive ones.
        
        # Filter positive eigenvalues
        # Because H has symmetric spectrum around 0.
        
        positive_indices = [i for i, val in enumerate(eigvals) if val > self.tol]
        # Sort them descending
        positive_indices = sorted(positive_indices, key=lambda i: eigvals[i], reverse=True)
        
        # Take top n_components
        n_keep = min(len(positive_indices), self.n_components)
        if n_keep == 0:
             # Fallback if something weird happens or empty matrix
             self.singular_values_ = np.array([])
             return

        chosen_indices = positive_indices[:n_keep]
        
        self.singular_values_ = eigvals[chosen_indices]
        S_k = eigvecs[:, chosen_indices] # Eigenvectors of T
        
        # Compute Ritz vectors (Approx eigenvectors of H)
        # Y = Q * S_k
        # Q is list of arrays, stack them. Q has n_iter+1 vectors. Use first n_iter.
        Q_mat = np.array(Q[:n_iter]).T # (dim, n_iter)
        
        Y = Q_mat.dot(S_k) # (m+n, n_keep)
        
        # Extract U and V
        # Y = [ u / sqrt(2) ]
        #     [ v / sqrt(2) ]
        # For augmented matrix eigenvectors corresponding to positive sigma:
        # y = [u; v]. But normalization might be different depending on standard SVD def.
        # Actually H [u; v] = s [u; v] => A v = s u, A^T u = s v.
        # Unit eigenvector y implies ||u||^2 + ||v||^2 = 1.
        # Typically ||u|| = ||v|| = 1/sqrt(2).
        
        self.U_ = Y[:m, :] * np.sqrt(2)
        self.components_ = Y[m:, :].T * np.sqrt(2) # V^T is (n_components, n)
        
        # Fix signs if needed (SVD is unique up to sign flips)
        # Not strictly necessary for LSA but good practice? Skip for now.
        
    def transform(self, A):
        # Project data to latent space
        # LSA transform: U_k * Sigma_k ... or just A * V_k^T?
        # Standard sklearn transform returns U * Sigma = A * V^T
        # We have U_ (m x k) and components_ (k x n) (V^T)
        
        # If A is new data (m_new x n):
        # res = A * V^T.T = A * V
        return A.dot(self.components_.T)
        

