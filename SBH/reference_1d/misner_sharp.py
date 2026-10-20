"""Misner-Sharp mass of a single-puncture PINN run: exact value M at every point and time
(the spacetime is Schwarzschild in any slicing). Evaluated along rays from a centre, FD in r."""
import sys, numpy as np
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_pinn as cp, jax.numpy as jnp
from asym_diag import fib_dirs
run=sys.argv[1]
man,wins=cp.load_run(run); ns=cp.module_for(man['model_config']); fo=ns['full_output']; unpack=ns['unpack_sym3']
track=np.loadtxt(run+'/puncture_track.txt')
def ev(w,t,X):
    fn,pr=ns['_wrap_moving_chi_bg_if_needed'](ns['apply_fn'],w['params'],w.get('traj_tt'),w.get('traj_xx'))
    X=jnp.asarray(X); T=jnp.full((X.shape[0],1),float(t))
    return np.asarray(fo(fn,pr,T,X[:,:1],X[:,1:2],X[:,2:3],jnp.array(w['t_start']),jnp.array(w['t_end'])))
dirs=fib_dirs(64)
e=np.cross(dirs,np.where(np.abs(dirs[:,2:3])<0.9,[[0,0,1.0]],[[1.0,0,0]])); e/=np.linalg.norm(e,axis=1,keepdims=True)
def parts(out,n):
    gt=np.asarray(unpack(jnp.asarray(out[:,0:6]))); At=np.asarray(unpack(jnp.asarray(out[:,6:12])))
    a=np.einsum('ni,nij,nj->n',e,gt,e); b=np.einsum('ni,nij,nj->n',n,gt,n)-a
    c=np.einsum('ni,nij,nj->n',e,At,e); chi=out[:,12]; K=out[:,13]+2*out[:,14]
    return a,b,c,chi,K
rs=[0.2,0.3,0.5,1,2,4,8,12,16,20]; h=1e-3
print("rows: t, centre; cols r =",rs,"; entries: mean(m_MS)-0.5 [spread over 64 directions]")
for t in (0.25,2.5,5.0,7.5,10.0):
    w=min(wins,key=lambda w_:abs(w_['t_end']-t)); k=int(np.argmin(abs(track[:,0]-t)))
    for cname,c0 in (("origin",np.zeros(3)),("tracked",np.array([track[k,1],track[k,2],0.0]))):
        row=[]
        for r in rs:
            Rs={}
            for s in (-1,0,1):
                rr=r+s*h; a,b,c,chi,K=parts(ev(w,t,c0+rr*dirs),dirs)
                Rs[s]=rr*np.sqrt(a/chi)
                if s==0: A,B,C,CHI,KK=a,b,c,chi,K
            Rp=(Rs[1]-Rs[-1])/(2*h); R=Rs[0]
            m=0.5*R*(1-Rp**2*CHI/(A+B)+R**2*(C/A+KK/3)**2)
            row.append(f"{m.mean()-0.5:+.1e}[{m.std():.0e}]")
        print(f"t={t:5}, {cname:7}: "+" ".join(row), flush=True)
