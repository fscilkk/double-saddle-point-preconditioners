"""Model non-Hermitian double saddle-point problem (2D MAC discretization) and preconditioners."""
import os
# Restrict BLAS/LAPACK to one thread (must happen before NumPy is imported for the first time).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as sla


class Timer:
    """Measures CPU time (time.process_time) and wall-clock time (time.perf_counter) together.
    Usage:  t = Timer(); ...; cpu, wall = t.stop()"""
    def __init__(self):
        self.c0 = time.process_time(); self.w0 = time.perf_counter()
    def stop(self):
        return time.process_time() - self.c0, time.perf_counter() - self.w0


def lap1d(N, h):  # Dirichlet 1D Laplacian, N interior points
    return sp.diags([-np.ones(N-1), 2*np.ones(N), -np.ones(N-1)], [-1, 0, 1]) / h**2

def conv1d(N, h):  # central difference d/dx, skew-symmetric
    return sp.diags([-np.ones(N-1), np.ones(N-1)], [-1, 1]) / (2*h)


def build(k, stab=True, nu=0.05, w=(1.0, 0.5), omega=1.0, Y=1.0, kappa=1.0, lam_th=0.05, cs2=1.0):
    """Unit square, k x k cells (k even). Velocity on MAC faces, pressure in cells,
    temperature on the (k/2)x(k/2) coarse grid. Returns blocks A,B,C,D,E (sparse, complex).
    Note: the tangential velocity uses a simplified Dirichlet treatment (lap1d on cell centres)."""
    h = 1.0/k
    I = lambda N: sp.identity(N, format='csr')
    # u_x: (k-1) interior x-faces in x, k cells in y ; u_y: k cells in x, (k-1) faces in y
    Lx_f, Lx_c = lap1d(k-1, h), lap1d(k, h)
    Dx_f, Dx_c = conv1d(k-1, h), conv1d(k, h)
    # ordering: x fastest
    Lux = sp.kron(I(k), Lx_f) + sp.kron(Lx_c, I(k-1))
    Luy = sp.kron(I(k-1), Lx_c) + sp.kron(Lx_f, I(k))
    Cux = w[0]*sp.kron(I(k), Dx_f) + w[1]*sp.kron(Dx_c, I(k-1))
    Cuy = w[0]*sp.kron(I(k-1), Dx_c) + w[1]*sp.kron(Dx_f, I(k))
    A1 = nu*Lux + Cux; A2 = nu*Luy + Cuy
    A = sp.block_diag([A1, A2]).tocsr().astype(complex)
    A = A + 1j*omega*sp.identity(A.shape[0])
    # MAC divergence: cells k x k
    Gx = sp.diags([np.ones(k-1), -np.ones(k-1)], [0, -1], shape=(k, k-1)) / h   # faces -> cells
    Bx = sp.kron(I(k), Gx); By = sp.kron(Gx, I(k))
    B = sp.hstack([Bx, By]).tocsr()
    B = B[:-1, :].astype(complex)                 # drop one cell -> full row rank
    m = B.shape[0]
    # restriction fine cells -> coarse cells (aggregation, average of 4)
    kc = k//2
    R1 = sp.kron(I(kc), np.ones((1, 2)))*0.5
    R = sp.kron(R1, R1).tocsr()[:, :-1]
    C = (1j*omega*kappa*R).astype(complex).tocsr()
    p = C.shape[0]
    if stab:
        # compressibility (skew) + boundary admittance (Hermitian PSD)
        xs = np.arange(k); X, Yg = np.meshgrid(xs, xs)
        bnd = ((X == 0) | (X == k-1) | (Yg == 0) | (Yg == k-1)).ravel()[:-1].astype(float)
        D = sp.diags(1j*omega/cs2*np.ones(m) + Y*bnd).tocsr()
        hc = 1.0/kc
        E = (lam_th*(sp.kron(I(kc), lap1d(kc, hc)) + sp.kron(lap1d(kc, hc), I(kc)))
             + 0.5*(sp.kron(I(kc), conv1d(kc, hc)) + sp.kron(conv1d(kc, hc), I(kc)))
             + 1j*omega*sp.identity(kc*kc)).tocsr().astype(complex)
    else:
        D = sp.csr_matrix((m, m), dtype=complex); E = sp.csr_matrix((p, p), dtype=complex)
    return A, B, C, D, E


def assemble(A, B, C, D, E):
    n, m, p = A.shape[0], B.shape[0], C.shape[0]
    K = sp.bmat([[A, B.conj().T, None], [B, -D, C.conj().T], [None, C, E]], format='csc')
    return K, (n, m, p)


class BlockPrec:
    """Applies the inverse of blkdiag(A~,S1~,S2~) (kind='diag') or of the block lower-triangular
    [[A~,0,0],[B,-S1~,0],[0,C,S2~]] (kind='tri')."""
    def __init__(self, solA, solS1, solS2, B, C, dims, kind='diag'):
        self.sA, self.s1, self.s2, self.B, self.C = solA, solS1, solS2, B, C
        self.n, self.m, self.p = dims; self.kind = kind
    def __call__(self, r):
        n, m = self.n, self.m
        r1, r2, r3 = r[:n], r[n:n+m], r[n+m:]
        x1 = self.sA(r1)
        if self.kind == 'diag':
            x2 = self.s1(r2); x3 = self.s2(r3)
        else:
            x2 = -self.s1(r2 - self.B @ x1)
            x3 = self.s2(r3 - self.C @ x2)
        return np.concatenate([x1, x2, x3])


def _givens(a, b):
    """Complex Givens rotation G = [[c, s], [-conj(s), c]] (c real) with G @ [a, b] = [r, 0]."""
    if b == 0:
        return 1.0, 0.0
    if a == 0:
        return 0.0, np.conj(b)/abs(b)
    r = np.hypot(abs(a), abs(b))
    c = abs(a)/r
    s = (a/abs(a))*np.conj(b)/r
    return c, s


def gmres_right(K, b, Minv=None, tol=1e-8, maxit=400):
    """Unrestarted right-preconditioned GMRES.
    Arnoldi: classical Gram-Schmidt with one full reorthogonalisation (CGS2).
    Least-squares problem: updated QR factorisation by complex Givens rotations (O(j) work per step).
    With right preconditioning the GMRES residual is the residual of the ORIGINAL system,
    so res[k] = ||b - K x_k|| / ||b|| after exactly k Arnoldi steps (= k iterations).
    Returns x, res (list, res[0] = 1), true relative residual, converged (bool)."""
    Minv = Minv or (lambda v: v)
    N = b.size; beta = np.linalg.norm(b)
    V = np.zeros((N, maxit+1), complex)
    Z = np.zeros((N, maxit), complex)
    H = np.zeros((maxit+1, maxit), complex)
    cs = np.zeros(maxit); sn = np.zeros(maxit, complex)
    g = np.zeros(maxit+1, complex); g[0] = beta
    V[:, 0] = b/beta; res = [1.0]
    j = -1
    for j in range(maxit):
        z = Minv(V[:, j]); Z[:, j] = z
        w = K @ z
        Vj = V[:, :j+1]
        for _ in range(2):                      # CGS2
            t = Vj.conj().T @ w
            H[:j+1, j] += t
            w -= Vj @ t
        H[j+1, j] = np.linalg.norm(w)
        hnext = H[j+1, j].real
        # apply previous rotations to the new column
        for i in range(j):
            hi, hi1 = H[i, j], H[i+1, j]
            H[i, j] = cs[i]*hi + sn[i]*hi1
            H[i+1, j] = -np.conj(sn[i])*hi + cs[i]*hi1
        # new rotation annihilating H[j+1, j]
        cs[j], sn[j] = _givens(H[j, j], H[j+1, j])
        H[j, j] = cs[j]*H[j, j] + sn[j]*H[j+1, j]; H[j+1, j] = 0.0
        g[j+1] = -np.conj(sn[j])*g[j]; g[j] = cs[j]*g[j]
        res.append(abs(g[j+1])/beta)
        if res[-1] < tol or hnext < 1e-14*beta:
            break
        V[:, j+1] = w/hnext
    k = j + 1
    y = sla.solve_triangular(H[:k, :k], g[:k], lower=False)
    x = Z[:, :k] @ y
    true_res = np.linalg.norm(b - K @ x)/beta
    return x, res, true_res, res[-1] < tol


def practical(A, B, C, D, E, schur='simple', Asolve='lu', nu=0.05):
    """Practical approximations.
    Returns (solA, solS1, solS2, (setup_cpu, setup_wall), S1h, S2h).
    nu must match the viscosity used in build()."""
    t = Timer()
    if Asolve == 'lu':
        luA = spla.splu(A.tocsc()); solA = luA.solve
    else:
        ilu = spla.spilu(A.tocsc(), drop_tol=1e-4, fill_factor=10); solA = ilu.solve
    if schur == 'simple':
        dA = A.diagonal()
        S1h = (D + B @ sp.diags(1/dA) @ B.conj().T).tocsc()
    elif schur == 'mass':
        S1h = (D + (1/nu)*sp.identity(B.shape[0])).tocsc()
    else:
        raise ValueError(schur)
    lu1 = spla.splu(S1h)
    S2h = (E + C @ sp.diags(1/S1h.diagonal()) @ C.conj().T).tocsc()
    lu2 = spla.splu(S2h)
    return solA, lu1.solve, lu2.solve, t.stop(), S1h, S2h
