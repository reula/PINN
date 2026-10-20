import sys, numpy as np
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_pinn as cp, jax.numpy as jnp
run=sys.argv[1]
man,wins=cp.load_run(run); ns=cp.module_for(man['model_config']); fo=ns['full_output']
ref=np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),'ref_single_m0.5_dr0.005.npz')); F=list(ref['fields'])
def ev(w,t,X):
    fn,pr=ns['_wrap_moving_chi_bg_if_needed'](ns['apply_fn'],w['params'],w.get('traj_tt'),w.get('traj_xx'))
    X=jnp.asarray(X); T=jnp.full((X.shape[0],1),float(t))
    return np.asarray(fo(fn,pr,T,X[:,:1],X[:,1:2],X[:,2:3],jnp.array(w['t_start']),jnp.array(w['t_end'])))
rs=np.array([1,2,4,6,8,10,12,15,18,20,22,23.5])
dirs=np.array([[1,0,0],[0,1,0],[0,0,1],[-1,-1,1]])/np.array([[1],[1],[1],[np.sqrt(3)]])
for t in (2.5,5.0,7.5,10.0):
    w=min(wins,key=lambda w_:abs(w_['t_end']-t)); k=int(np.argmin(abs(ref['t']-t)))
    aref=np.interp(rs,ref['r'],ref['U'][k,F.index('alpha')]); cref=np.interp(rs,ref['r'],ref['U'][k,F.index('chi')])
    da=[]; dc=[]
    for d in dirs:
        o=ev(w,t,rs[:,None]*d[None,:]); da.append(o[:,18]-aref); dc.append(o[:,12]-cref)
    da=np.mean(da,0); dc=np.mean(dc,0)
    print(f"t={t:4}: d_alpha(r) "+" ".join(f"{v:+.1e}" for v in da)); print(f"        d_chi(r)   "+" ".join(f"{v:+.1e}" for v in dc), flush=True)
print("r =", rs)
