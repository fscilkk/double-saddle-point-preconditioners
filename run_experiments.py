"""Numerical experiments for the revised manuscript.
Produces: fig_eigs_unstab.pdf, fig_eigs_stab.pdf, fig_conv.pdf, results.txt"""
import numpy as np, json
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from dsp import *
rng = np.random.default_rng(2026)
out = open('results.txt','w')
def log(*a):
    s=' '.join(str(x) for x in a); print(s); out.write(s+'\n'); out.flush()

def dense_ideal(A,B,C,D,E):
    Ad=A.toarray(); Bd=B.toarray(); Cd=C.toarray()
    luA=spla.splu(A.tocsc())
    S1=D.toarray()+Bd@luA.solve(Bd.conj().T)
    S2=E.toarray()+Cd@np.linalg.solve(S1,Cd.conj().T)
    return luA, S1, S2

# ---------------- 1. eigenvalue plots (k=16) -----------------
for stab in [False, True]:
    A,B,C,D,E = build(16, stab); K,dims = assemble(A,B,C,D,E); n,m,p = dims
    Kd = K.toarray(); luA,S1,S2 = dense_ideal(A,B,C,D,E)
    Ad = A.toarray()
    def Pmat(Ah,S1h,S2h,kind):
        Z=lambda a,b: np.zeros((a,b))
        if kind=='diag': return np.block([[Ah,Z(n,m),Z(n,p)],[Z(m,n),S1h,Z(m,p)],[Z(p,n),Z(p,m),S2h]])
        return np.block([[Ah,Z(n,m),Z(n,p)],[B.toarray(),-S1h,Z(m,p)],[Z(p,n),C.toarray(),S2h]])
    _,_,_,_,S1h,S2h = practical(A,B,C,D,E,'mass')
    S1h=S1h.toarray(); S2h=S2h.toarray()
    evK  = np.linalg.eigvals(Kd)
    evP  = np.linalg.eigvals(np.linalg.solve(Pmat(Ad,S1,S2,'diag'),Kd))
    evPD = np.linalg.eigvals(np.linalg.solve(Pmat(Ad,S1h,S2h,'diag'),Kd))
    evPT = np.linalg.eigvals(np.linalg.solve(Pmat(Ad,S1h,S2h,'tri'),Kd))
    # Bendixson strip from the 3x3 comparison matrices
    H=lambda X:(X+X.conj().T)/2
    eA=np.linalg.eigvalsh(H(Ad)); eD=np.linalg.eigvalsh(H(D.toarray())); eE=np.linalg.eigvalsh(H(E.toarray()))
    nb=np.linalg.norm(B.toarray(),2); nc=np.linalg.norm(C.toarray(),2)
    Mu=np.array([[eA.max(),nb,0],[nb,-eD.min(),nc],[0,nc,eE.max()]])
    Ml=np.array([[eA.min(),-nb,0],[-nb,-eD.max(),-nc],[0,-nc,eE.min()]])
    lo,hi=np.linalg.eigvalsh(Ml).min(),np.linalg.eigvalsh(Mu).max()
    eH=np.linalg.eigvalsh(H(Kd))
    log(f"[eig] stab={stab} k=16 (n,m,p)={dims} Bendixson strip [{lo:.3g},{hi:.3g}] ; true [lmin(H),lmax(H)]=[{eH.min():.3g},{eH.max():.3g}] ; Re sigma(K) in [{evK.real.min():.3g},{evK.real.max():.3g}]")
    log(f"[eig] inertia(H): +{(eH>0).sum()} -{(eH<0).sum()} (expected +{n+p} -{m})")
    if not stab:
        vals,cnt=np.unique(np.round(evP.real,6),return_counts=True)
        log("[eig] ideal P^-1K distinct eigenvalues/multiplicities:",dict(zip(vals.tolist(),cnt.tolist())),
            f"expected mult: 1->{n-m}, golden pair->{m-p} each, cubic roots->{p} each")
        M=np.linalg.solve(Pmat(Ad,S1,S2,'diag'),Kd); I=np.eye(n+m+p)
        q=(M-I)@(M@M-M-I)@(M@M@M-M@M-2*M+I)
        log(f"[eig] ||q(M)||_2 = {np.linalg.norm(q,2):.2e}  (q = (x-1)(x^2-x-1)(x^3-x^2-2x+1))")
        w,V=np.linalg.eig(M); log(f"[eig] kappa_2(V) of ideal M (numpy eigvec basis) = {np.linalg.cond(V):.2e}")
    else:
        log(f"[eig] ideal: #(lambda=1)={np.sum(abs(evP-1)<1e-7)} (n-m={n-m}); Re range non-unit [{evP.real.min():.3g},{evP.real.max():.3g}], |Im|max={abs(evP.imag).max():.3g}")
    # epsilon of practical blocks (Assumption 5.1)
    e1=np.linalg.norm(np.linalg.solve(S1h,S1)-np.eye(m),2); e2=np.linalg.norm(np.linalg.solve(S2h,S2)-np.eye(p),2)
    log(f"[eps] practical blocks: eps_A=0 (sparse LU), eps_1={e1:.3g}, eps_2={e2:.3g}")
    fig,ax=plt.subplots(1,4,figsize=(15,3.4))
    for a,ev,t in zip(ax,[evK,evP,evPD,evPT],[r'$K$',r'$\mathcal{P}^{-1}K$ (ideal)',r'$\widetilde{\mathcal{P}}_D^{-1}K$ (practical)',r'$\widetilde{\mathcal{P}}_T^{-1}K$ (practical)']):
        a.plot(ev.real,ev.imag,'.',ms=3); a.set_title(t); a.set_xlabel('Re'); a.grid(alpha=.3)
    ax[0].axvline(lo,color='r',ls='--',lw=1); ax[0].axvline(hi,color='r',ls='--',lw=1)
    ax[0].set_xscale('symlog',linthresh=1); ax[0].set_xlim(1.3*lo,1.3*hi); ax[0].set_ylabel('Im')
    fig.tight_layout(); fig.savefig(f"fig_eigs_{'stab' if stab else 'unstab'}.pdf"); plt.close(fig)

# ---------------- 2. scaling study ----------------
hist={}
for stab in [False, True]:
    log(f"\n[scal] ===== {'stabilized D,E!=0' if stab else 'unstabilized D=E=0'} =====")
    log("[scal] k  N  | method : iters  setup(s)  solve(s)  total(s)  true_res")
    for k in [16,32,64,128]:
        A,B,C,D,E = build(k, stab); K,dims = assemble(A,B,C,D,E); N=K.shape[0]
        b = rng.standard_normal(N)+1j*rng.standard_normal(N)
        rows=[]
        def run(name, setup, Pinv, maxit=300):
            t0=time.process_time(); x,res,tr=gmres_right(K,b,Pinv,1e-8,maxit); ts=time.process_time()-t0
            it = len(res)-1 if res[-1]<1e-8 else f">{maxit}"
            log(f"[scal] {k:4d} {N:6d} | {name:16s}: {str(it):>5s} {setup:8.2f} {ts:8.2f} {setup+ts:8.2f} {tr:.1e}")
            hist[(stab,k,name)]=res
        if k<=32:
            t0=time.process_time(); luA,S1,S2=dense_ideal(A,B,C,D,E)
            i1=np.linalg.inv(S1); i2=np.linalg.inv(S2); st=time.process_time()-t0
            run('ideal P (dense)', st, BlockPrec(luA.solve,lambda r:i1@r,lambda r:i2@r,B,C,dims,'diag'))
            run('no precond.', 0.0, None, 300)
        sA,s1,s2,st,_,_ = practical(A,B,C,D,E,'mass')
        run('P_D practical', st, BlockPrec(sA,s1,s2,B,C,dims,'diag'))
        run('P_T practical', st, BlockPrec(sA,s1,s2,B,C,dims,'tri'))
        sA,s1,s2,st,_,_ = practical(A,B,C,D,E,'simple')
        run('P_D SIMPLE-type', st, BlockPrec(sA,s1,s2,B,C,dims,'diag'))
        try:
            t0=time.process_time(); il=spla.spilu(K, drop_tol=1e-4, fill_factor=10); st=time.process_time()-t0
            run('ILU(K) global', st, il.solve)
        except Exception as e: log(f"[scal] {k} ILU(K) failed: {e}")
        t0=time.process_time(); lu=spla.splu(K); x=lu.solve(b); t=time.process_time()-t0
        log(f"[scal] {k:4d} {N:6d} | {'sparse direct LU':16s}:     -          -        - {t:8.2f} {np.linalg.norm(b-K@x)/np.linalg.norm(b):.1e}")

# ---------------- 3. convergence histories (k=64) ----------------
fig,ax=plt.subplots(1,2,figsize=(11,3.8))
for a,stab in zip(ax,[False,True]):
    for name in ['P_D practical','P_T practical','P_D SIMPLE-type','ILU(K) global']:
        r=hist.get((stab,64,name))
        if r: a.semilogy(r,label=name)
    r=hist.get((stab,32,'ideal P (dense)')); a.semilogy(r,'k--',label='ideal P (k=32)')
    r=hist.get((stab,32,'no precond.')); a.semilogy(r,'k:',label='no precond. (k=32)')
    a.set_title('unstabilized $D=E=0$' if not stab else 'stabilized $D,E\\neq0$'); a.set_xlabel('GMRES iteration'); a.set_ylabel(r'$\|b-Kx_k\|_2/\|b\|_2$'); a.grid(alpha=.3); a.set_ylim(1e-10,2)
ax[1].legend(fontsize=8); fig.tight_layout(); fig.savefig('fig_conv.pdf'); plt.close(fig)
log("done")
