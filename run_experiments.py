"""Numerical experiments for the revised manuscript.
Produces: fig_eigs_unstab.pdf, fig_eigs_stab.pdf, fig_conv.pdf, results.txt, results.json

All BLAS/LAPACK calls are restricted to ONE thread (see below), and every timing is reported
both as CPU time (time.process_time) and wall-clock time (time.perf_counter).
Set the environment variable KS (e.g. KS=16,32) to run only some grid sizes."""
import os
# ---- single-threaded BLAS: must be set before NumPy/SciPy are imported ----
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import json, platform, sys
import numpy as np
import scipy, scipy.sparse.linalg as spla
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from dsp import build, assemble, BlockPrec, gmres_right, practical, Timer

KS = [int(s) for s in os.environ.get("KS", "16,32,64,128").split(",")]
TOL, MAXIT = 1e-8, 300

rng = np.random.default_rng(2026)
out = open('results.txt', 'w')
def log(*a):
    s = ' '.join(str(x) for x in a); print(s); out.write(s+'\n'); out.flush()

# ---------------- environment -----------------
log(f"[env] Python {sys.version.split()[0]}, NumPy {np.__version__}, SciPy {scipy.__version__}, {platform.platform()}")
log(f"[env] processor: {platform.processor() or 'n/a'}")
try:
    from threadpoolctl import threadpool_info
    for info in threadpool_info():
        log(f"[env] {info.get('internal_api')} ({info.get('user_api')}): num_threads={info.get('num_threads')}")
except ImportError:
    log("[env] threadpoolctl not installed; BLAS thread count set via environment variables to 1")

def dense_ideal(A, B, C, D, E):
    Bd = B.toarray(); Cd = C.toarray()
    luA = spla.splu(A.tocsc())
    S1 = D.toarray() + Bd @ luA.solve(Bd.conj().T)
    S2 = E.toarray() + Cd @ np.linalg.solve(S1, Cd.conj().T)
    return luA, S1, S2

# theoretical eigenvalues of the ideal preconditioned matrix (D=E=0)
GOLD = np.array([(1+np.sqrt(5))/2, (1-np.sqrt(5))/2])
CUBIC = 2*np.cos(np.array([1, 3, 5])*np.pi/7)

# ---------------- 1. eigenvalue plots (k=16) -----------------
for stab in [False, True]:
    A, B, C, D, E = build(16, stab); K, dims = assemble(A, B, C, D, E); n, m, p = dims
    Kd = K.toarray(); luA, S1, S2 = dense_ideal(A, B, C, D, E)
    Ad = A.toarray()
    def Pmat(Ah, S1h, S2h, kind):
        Z = lambda a, b: np.zeros((a, b))
        if kind == 'diag':
            return np.block([[Ah, Z(n, m), Z(n, p)], [Z(m, n), S1h, Z(m, p)], [Z(p, n), Z(p, m), S2h]])
        return np.block([[Ah, Z(n, m), Z(n, p)], [B.toarray(), -S1h, Z(m, p)], [Z(p, n), C.toarray(), S2h]])
    _, _, _, _, S1h, S2h = practical(A, B, C, D, E, 'mass')
    S1h = S1h.toarray(); S2h = S2h.toarray()
    evK  = np.linalg.eigvals(Kd)
    evP  = np.linalg.eigvals(np.linalg.solve(Pmat(Ad, S1, S2, 'diag'), Kd))
    evPD = np.linalg.eigvals(np.linalg.solve(Pmat(Ad, S1h, S2h, 'diag'), Kd))
    evPT = np.linalg.eigvals(np.linalg.solve(Pmat(Ad, S1h, S2h, 'tri'), Kd))
    # Bendixson strip from the 3x3 comparison matrices
    H = lambda X: (X + X.conj().T)/2
    eA = np.linalg.eigvalsh(H(Ad)); eD = np.linalg.eigvalsh(H(D.toarray())); eE = np.linalg.eigvalsh(H(E.toarray()))
    nb = np.linalg.norm(B.toarray(), 2); nc = np.linalg.norm(C.toarray(), 2)
    Mu = np.array([[eA.max(), nb, 0], [nb, -eD.min(), nc], [0, nc, eE.max()]])
    Ml = np.array([[eA.min(), -nb, 0], [-nb, -eD.max(), -nc], [0, -nc, eE.min()]])
    lo, hi = np.linalg.eigvalsh(Ml).min(), np.linalg.eigvalsh(Mu).max()
    eH = np.linalg.eigvalsh(H(Kd))
    log(f"[eig] stab={stab} k=16 (n,m,p)={dims} Bendixson strip [{lo:.3g},{hi:.3g}] ; "
        f"true [lmin(H),lmax(H)]=[{eH.min():.3g},{eH.max():.3g}] ; "
        f"Re sigma(K) in [{evK.real.min():.3g},{evK.real.max():.3g}]")
    log(f"[eig] inertia(H): +{(eH > 0).sum()} -{(eH < 0).sum()} (expected +{n+p} -{m})")
    if not stab:
        cnt = lambda r: int(np.sum(abs(evP - r) < 1e-6))
        c1 = cnt(1.0); cg = [cnt(r) for r in GOLD]; cc = [cnt(r) for r in CUBIC]
        log(f"[eig] ideal P^-1K multiplicities (|lambda-root|<1e-6): 1 -> {c1}, "
            f"(1+-sqrt5)/2 -> {cg}, 2cos(k pi/7), k=1,3,5 -> {cc}; "
            f"total {c1+sum(cg)+sum(cc)} of {len(evP)}; expected {n-m}, {m-p} each, {p} each")
        M = np.linalg.solve(Pmat(Ad, S1, S2, 'diag'), Kd); I = np.eye(n+m+p)
        q = (M-I) @ (M@M - M - I) @ (M@M@M - M@M - 2*M + I)
        log(f"[eig] ||q(M)||_2 = {np.linalg.norm(q, 2):.2e}  (q = (x-1)(x^2-x-1)(x^3-x^2-2x+1))")
        w_, V = np.linalg.eig(M); log(f"[eig] kappa_2(V) of ideal M (numpy eigvec basis) = {np.linalg.cond(V):.2e}")
    else:
        unit = abs(evP - 1) < 1e-7; nu_ = evP[~unit]
        log(f"[eig] ideal: #(lambda=1)={unit.sum()} (n-m={n-m}); non-unit eigenvalues: "
            f"Re in [{nu_.real.min():.3g},{nu_.real.max():.3g}], |Im|max={abs(nu_.imag).max():.3g}")
    # epsilon of practical blocks (Assumption 5.1)
    e1 = np.linalg.norm(np.linalg.solve(S1h, S1) - np.eye(m), 2)
    e2 = np.linalg.norm(np.linalg.solve(S2h, S2) - np.eye(p), 2)
    log(f"[eps] practical blocks: eps_A=0 (sparse LU), eps_1={e1:.3g}, eps_2={e2:.3g}")
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
    for a, ev, t in zip(ax, [evK, evP, evPD, evPT],
                        [r'$K$', r'$\mathcal{P}^{-1}K$ (ideal)',
                         r'$\widetilde{\mathcal{P}}_D^{-1}K$ (practical)',
                         r'$\widetilde{\mathcal{P}}_T^{-1}K$ (practical)']):
        a.plot(ev.real, ev.imag, '.', ms=3); a.set_title(t); a.set_xlabel('Re'); a.grid(alpha=.3)
    ax[0].axvline(lo, color='r', ls='--', lw=1); ax[0].axvline(hi, color='r', ls='--', lw=1)
    ax[0].set_xscale('symlog', linthresh=1); ax[0].set_xlim(1.3*lo, 1.3*hi); ax[0].set_ylabel('Im')
    fig.tight_layout(); fig.savefig(f"fig_eigs_{'stab' if stab else 'unstab'}.pdf"); plt.close(fig)

# ---------------- 2. scaling study ----------------
hist = {}
table = []          # machine-readable results -> results.json
for stab in [False, True]:
    case = 'stabilized' if stab else 'unstabilized'
    log(f"\n[scal] ===== {'stabilized D,E!=0' if stab else 'unstabilized D=E=0'} =====")
    log("[scal]    k      N | method          : iters | setup cpu/wall | solve cpu/wall | total cpu/wall | true_res")
    for k in KS:
        A, B, C, D, E = build(k, stab); K, dims = assemble(A, B, C, D, E); N = K.shape[0]
        b = rng.standard_normal(N) + 1j*rng.standard_normal(N)

        def record(name, it, setup, solve, tr):
            tot = (setup[0]+solve[0], setup[1]+solve[1])
            log(f"[scal] {k:4d} {N:6d} | {name:16s}: {str(it):>5s} | {setup[0]:6.2f} {setup[1]:6.2f} "
                f"| {solve[0]:6.2f} {solve[1]:6.2f} | {tot[0]:6.2f} {tot[1]:6.2f} | {tr:.1e}")
            table.append(dict(case=case, k=k, N=N, method=name, iters=it,
                              setup_cpu=setup[0], setup_wall=setup[1],
                              solve_cpu=solve[0], solve_wall=solve[1],
                              total_cpu=tot[0], total_wall=tot[1], true_res=tr))

        def run(name, setup, Pinv, maxit=MAXIT):
            t = Timer(); x, res, tr, conv = gmres_right(K, b, Pinv, TOL, maxit); solve = t.stop()
            nit = len(res) - 1
            it = nit if conv else (f">{maxit}" if nit == maxit else f"stall{nit}")
            record(name, it, setup, solve, tr)
            hist[(stab, k, name)] = res

        if k <= 32:
            t = Timer(); luA, S1, S2 = dense_ideal(A, B, C, D, E)
            i1 = np.linalg.inv(S1); i2 = np.linalg.inv(S2); st = t.stop()
            run('ideal P (dense)', st, BlockPrec(luA.solve, lambda r: i1 @ r, lambda r: i2 @ r, B, C, dims, 'diag'))
            run('no precond.', (0.0, 0.0), None)
        sA, s1, s2, st, _, _ = practical(A, B, C, D, E, 'mass')
        run('P_D practical', st, BlockPrec(sA, s1, s2, B, C, dims, 'diag'))
        run('P_T practical', st, BlockPrec(sA, s1, s2, B, C, dims, 'tri'))
        sA, s1, s2, st, _, _ = practical(A, B, C, D, E, 'simple')
        run('P_D SIMPLE-type', st, BlockPrec(sA, s1, s2, B, C, dims, 'diag'))
        try:
            t = Timer(); il = spla.spilu(K, drop_tol=1e-4, fill_factor=10); st = t.stop()
            run('ILU(K) global', st, il.solve)
        except Exception as e:
            log(f"[scal] {k:4d} {N:6d} | ILU(K) global   : failed ({e})")
            table.append(dict(case=case, k=k, N=N, method='ILU(K) global', iters='f'))
        t = Timer(); lu = spla.splu(K); x = lu.solve(b); tt = t.stop()
        tr = np.linalg.norm(b - K @ x)/np.linalg.norm(b)
        log(f"[scal] {k:4d} {N:6d} | {'sparse direct LU':16s}:     - |      -      - |      -      - "
            f"| {tt[0]:6.2f} {tt[1]:6.2f} | {tr:.1e}")
        table.append(dict(case=case, k=k, N=N, method='sparse direct LU', iters='-',
                          total_cpu=tt[0], total_wall=tt[1], true_res=tr))

with open('results.json', 'w') as f:
    json.dump(table, f, indent=1, default=str)

# ---------------- 3. convergence histories (k=64) ----------------
kc_ = 64 if 64 in KS else max(KS)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for a, stab in zip(ax, [False, True]):
    for name in ['P_D practical', 'P_T practical', 'P_D SIMPLE-type', 'ILU(K) global']:
        r = hist.get((stab, kc_, name))
        if r: a.semilogy(r, label=name)
    r = hist.get((stab, 32, 'ideal P (dense)'))
    if r: a.semilogy(r, 'k--', label='ideal P (k=32)')
    r = hist.get((stab, 32, 'no precond.'))
    if r: a.semilogy(r, 'k:', label='no precond. (k=32)')
    a.set_title('unstabilized $D=E=0$' if not stab else 'stabilized $D,E\\neq0$')
    a.set_xlabel('GMRES iteration'); a.set_ylabel(r'$\|b-Kx_k\|_2/\|b\|_2$')
    a.grid(alpha=.3); a.set_ylim(1e-10, 2)
ax[1].legend(fontsize=8); fig.tight_layout(); fig.savefig('fig_conv.pdf'); plt.close(fig)
log("done")
