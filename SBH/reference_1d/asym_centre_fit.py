import sys, numpy as np
sys.path.insert(0,'/home/claude/bh/reference_1d')
import compare_pinn as cp, jax.numpy as jnp
from asym_diag import fib_dirs
run='/mnt/user-data/uploads/BH/single_m05_jax09_core48k_qn24k_t10_r4'
man,wins=cp.load_run(run); ns=cp.module_for(man['model_config'])
fo=ns['full_output']
track=np.loadtxt(run+'/puncture_track.txt')
def ev(w,t,X):
    fn,pr=ns['_wrap_moving_chi_bg_if_needed'](ns['apply_fn'],w['params'],w.get('traj_tt'),w.get('traj_xx'))
    X=jnp.asarray(X); T=jnp.full((X.shape[0],1),float(t))
    return np.asarray(fo(fn,pr,T,X[:,:1],X[:,1:2],X[:,2:3],jnp.array(w['t_start']),jnp.array(w['t_end'])))
d=fib_dirs(400); h=0.01
for t in (5.0,7.5,10.0):
    w=min(wins,key=lambda w_:abs(w_['t_end']-t)); k=int(np.argmin(abs(track[:,0]-t)))
    for r in (0.1,0.2,0.3):
        res=[]
        for idx,name in ((18,'alpha'),(12,'chi')):
            f=ev(w,t,r*d)[:,idx]; fp=ev(w,t,(r+h)*d)[:,idx].mean(); fm=ev(w,t,(r-h)*d)[:,idx].mean()
            dfdr=(fp-fm)/(2*h)
            A=np.hstack([np.ones((len(d),1)),d]); c,*_=np.linalg.lstsq(A,f,rcond=None)
            res.append((name,-c[1:]/dfdr))
        print(f"t={t} r={r}: track ({track[k,1]:+.4f},{track[k,2]:+.4f}, 0)  centre from "+"  ".join(f"{n}: ({v[0]:+.4f},{v[1]:+.4f},{v[2]:+.4f})" for n,v in res), flush=True)
