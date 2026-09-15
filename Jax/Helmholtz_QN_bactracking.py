######### Import libraries ################
import os
from tqdm import tqdm
import sys
import os
file_path = os.getcwd()
project_root = os.path.dirname(file_path)
print(f"Project root: {project_root}")
if project_root not in sys.path:
    sys.path.append(project_root)
import time
import jax
from jax import flatten_util
import jax.numpy as jnp
import optax
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from flax import linen as nn
from typing import Sequence
from functools import partial
import matplotlib as mpl
from jax import random as randomjax
import random

######### The Optimizers ################
from Crunch.Models.layers import  *
from Crunch.Models.polynomials import  *
from Crunch.Auxiliary.metrics import  *
from Crunch.Optimizers.minimize_backtracking import minimize
jax.config.update("jax_enable_x64", True)
#################################################

############## Added Code ##############3
from typing import Callable
kernel_init: Callable
from functools import partial

# force jax to use one device
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

#Create a colormap
########################################################
cmap = 'RdBu_r'
num_colors=8
# Create a colormap
cmap = plt.get_cmap(cmap)
colors = [cmap(i) for i in np.linspace(0, 1, num_colors)]
colors=colors[:num_colors//4]+colors[3*num_colors//4:]
print(len(colors))
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['font.size'] = 14
#########################################################

#Set hyperparameters
#############################################################################################
import argparse
# Set up argument parser
parser = argparse.ArgumentParser(description='Tuning Parameters')

# File name parameters
parser.add_argument('--Equation', type=str, default='Burgers', help='Name of equation')
parser.add_argument('--Name', type=str, default='SS-Uniform', help='Name of the experiment')
#Batch (training points) parameters
parser.add_argument('--Nint', type=int, default=15000, help='batch_size') #Number of points per batch
parser.add_argument('--k_samp', type=float, default=1.0, help='Enhance outliers smoothing factor') #k factor of Wu et al (adaptive resampling)
parser.add_argument('--c_samp', type=float, default=1.0, help='homogenize') #c factor of Wu et al (adaptive resampling)
parser.add_argument('--N_change', type=int, default=100, help='homogenize') #Resampling epochs
#Seed
parser.add_argument('--SEED', type=int, default=9998, help='Random seed')
#Training epochs
parser.add_argument('--EPOCHS', type=int, default=1000, help='Number of training epochs')
#Architecture hyperparameters (layers, neurons, fourier features if any)
parser.add_argument('--N_LAYERS', type=int, default=4, help='Number of layers in the network') #Number of hidden layers
parser.add_argument('--HIDDEN', type=int, default=30, help='Number of hidden units per layer') #Number of neurons in each hidden layer
parser.add_argument('--FEATURES', type=int, default=1, help='Feature size') 
parser.add_argument('--degree', type=int, default=9, help='Degree of outer') #What is this
parser.add_argument('--degree_T', type=int, default=2, help='Degree of polynomial') #What is this
#Learning rate parameters (Adam)
parser.add_argument('--lr_fact', type=float, default=0.2, help='Scale Lr') #Related with the lr schedule of Adam. Not sure what is this, exactly
parser.add_argument('--decay_rate', type=float, default=0.9, help='Decay rate for learning rate schedule')
parser.add_argument('--LR', type=float, default=1e-3, help='Initial learning rate')
parser.add_argument('--decay_step', type=int,default=5000, help='Decay step size')
#parameters related with RBA (not used here)
#parser.add_argument('--eta', type=float, default=0.01, help='Learning rate or step size for adaptive gamma')
#parser.add_argument('--gamma', type=float, default=0.999, help='Decay rate for adaptive gamma')
#parser.add_argument('--gamma_bfgs', type=float, default=0.1, help='Decay rate for adaptive gamma')
#parser.add_argument('--gamma_grads', type=float, default=0.99, help='Decay rate for adaptive gamma')
#parser.add_argument('--cap_RBA', type=float, default=20, help='Cap limit for RBA')
#parser.add_argument('--max_RBA', type=float, help='Maximum RBA value, default calculated as eta / (1 - gamma)')
#parser.add_argument('--phi', type=float, default=0.95, help='Enhance outliers smoothing factor')
#parser.add_argument('--c_log', type=float, default=1.0, help='homogenize')
parser.add_argument('--Note', type=str, default='', help='In case')

# Parse arguments and display them
args, unknown = parser.parse_known_args()
for arg, value in vars(args).items():
    print(f'{arg}: {value}')

# Initialize parameters with parsed or default values
Nint = args.Nint
SEED = args.SEED
EPOCHS = args.EPOCHS
N_LAYERS = args.N_LAYERS
HIDDEN = args.HIDDEN
FEATURES = args.FEATURES
degree = args.degree
degree_T = args.degree_T

# Optimizer parameters
decay_rate = args.decay_rate
LR = args.LR
lr0 = LR
decay_step = args.decay_step# if args.decay_step is not None else int(EPOCHS * jnp.log(decay_rate) / jnp.log(lrf / lr0))

#resampling
k = args.k_samp
c = args.c_samp
Nchange = args.N_change
args.Name=""
print(args.Name)

# random key
key = jax.random.PRNGKey(SEED)
key, subkey = jax.random.split(key, 2)
np.random.seed(SEED)


#################################################################################

# ------------------------ Sampling points ----------------------
##################################################################
box = [(-1.0, 1.0), (-1.0, 1.0)]  # (x,y)

def generate_inputs(Nint, key,xinit=-1.0,yinit=-1.0,Lx=2.0,Ly=2.0):
    # key: a PRNGKey (jax.random.PRNGKey)
    key_x, key_y, new_key = randomjax.split(key, 3)
    x = Lx * jax.random.uniform(key_x, (Nint, 1)) + xinit
    y = Ly * jax.random.uniform(key_y, (Nint, 1)) + yinit
    X = jnp.hstack((x,y))
    return X, new_key

def generate_test(Nx, Ny, xinit=-1.0,yinit=-1.0,Lx=2.0,Ly=2.0):
    x = np.linspace(xinit, xinit+Lx, Nx)
    y = np.linspace(yinit, yinit+Lx, Ny)
    x, y = np.meshgrid(x, y)
    X = np.hstack((x.flatten()[:, None], y.flatten()[:, None]))
    return X, x, y


#################################################################

#--------------------The model (MLP)------------------------------------
########################################################################
kmax=1
Lx=2.0
Ly=2.0
class MLP(nn.Module):
    degree: int
    features: Sequence[int]
    M:int =10
    def setup(self):
         self.T_funcs = [globals()[f"T{i}"] for i in range(self.degree+1)] 
    @nn.compact
    def __call__(self, x, y):
        init : Callable
        ks=jnp.arange(1,kmax+1)
        Xper = 2*jnp.pi*jnp.matmul(x,ks[None,:])/Lx
        Yper = 2*jnp.pi*jnp.matmul(y,ks[None,:])/Ly
        xcos = jnp.cos(Xper)
        xsin = jnp.sin(Xper)
        ycos = jnp.cos(Yper)
        ysin = jnp.sin(Yper)
        Xper = jnp.concatenate([xcos,xsin],axis=1)
        Yper = jnp.concatenate([ycos,ysin],axis=1)
        Z = jnp.concatenate([Xper, Yper], axis=-1)
        for fs in self.features[:-1]:
            Z = nn.Dense(fs, kernel_init=nn.initializers.xavier_uniform())(Z)
            Z = nn.activation.tanh(Z)
        Z = nn.Dense(self.features[-1], kernel_init=nn.initializers.xavier_uniform())(Z)
        return Z
#####################################################################

#-----------------------The output ----------------------------------
#####################################################################
class PINN(nn.Module):
    degree: int
    degree_T:int
    features: Sequence[int]
    M:int =10
    def setup(self):
         self.MLP = MLP(degree=self.degree,features=self.features,M=self.M)
    @nn.compact
    def __call__(self, x, y):
        u=self.MLP(x,y)
        return u
#####################################################################
    
# optimizer step function
@partial(jax.jit, static_argnums=(0,))
def update_model(optim, gradient, params, state):
    updates, state = optim.update(gradient, state)
    params = optax.apply_updates(params, updates)
    return params, state

# -
@partial(jax.jit, static_argnums=(0, 1))  # key and optimizer are static
def update_model(key, optimizer, gradient, params, state):
    # Perform updates using the specified optimizer and key
    updates, new_state = optimizer.update(gradient['params'][key], state)
    new_params = optax.apply_updates(params['params'][key], updates)
    # Return updated parameters and state for this key only
    params['params'][key] = new_params
    return params, new_state

#------------------------- Loss function computation ------------------------------------
########################################################################################
kval = 1.0
a1 = 1.0
a2 = 4.0
@partial(jax.jit, static_argnums=(0,))
def apply_model(apply_fn, params,lamE,all_grads,*train_data):
    # Unpack data
    x,y = train_data

    # Define residual function
    def r_E(params, x, y):
        u = apply_fn(params, x, y)
        v_x = jnp.ones_like(x)
        v_y = jnp.ones_like(y)
        _,uxx = hvp_fwdfwd(lambda x_val: apply_fn(params, x_val, y), (x,), (v_x,),return_primals=True) 
        _,uyy = hvp_fwdfwd(lambda y_val: apply_fn(params, x, y_val), (y,), (v_y,),return_primals=True) 
        return uxx + uyy + kval**2*u + (jnp.pi**2 * (a1**2 + a2**2)-kval**2)*jnp.sin(jnp.pi*a1*x)*jnp.sin(jnp.pi*a2*y)

    def loss_pde(params):
        # Compute residuals
        residuals = r_E(params, x, y)
        pde_loss = jnp.mean((residuals)**2)
        return pde_loss

    # Compute gradients separately
    pde_loss, gradient_pde = jax.value_and_grad(loss_pde)(params)
    #Store
    all_loss={
        'loss_PDE':pde_loss,
        'loss_BCs':0.0,
        'Loss':pde_loss 
    }
    return all_loss,gradient_pde
##########################################################################################

#Function to generate PDE residuals (necessary for adaptive resampling)
@partial(jax.jit, static_argnums=(0,))
def r_E_external(apply_fn, params, x, y, a1, a2, kval):
    v_x = jnp.ones_like(x)
    v_y = jnp.ones_like(y)

    # ∂u/∂x and ∂²u/∂x² using your forward-forward HVP
    ux, uxx = hvp_fwdfwd(
        lambda x_val: apply_fn(params, x_val, y),
        (x,), (v_x,),
        return_primals=True
    )

    uy, uyy = hvp_fwdfwd(
        lambda y_val: apply_fn(params, x, y_val),
        (y,), (v_y,),
        return_primals=True
    )

    u = apply_fn(params, x, y)

    return uxx + uyy + kval**2*u + (jnp.pi**2 * (a1**2 + a2**2)-kval**2)*jnp.sin(jnp.pi*a1*x)*jnp.sin(jnp.pi*a2*y)

@partial(jax.jit, static_argnames=("apply_fn", "Nint", "rad_args", "Ntest"))
def adaptive_rad(apply_fn, params, a1, a2, kval,
                 Nint, rad_args,
                 Ntest=100000,
                 key=randomjax.PRNGKey(0)):

    # --------------- 1. Generate candidate points -----------------
    Xtest, key = generate_inputs(Ntest, key)
    xtest = Xtest[:, 0:1]
    ytest = Xtest[:, 1:2]

    # --------------- 2. Residuals --------------------------------
    residuals = r_E_external(apply_fn, params, xtest, ytest, a1, a2, kval)
    Y = jnp.abs(residuals).reshape(-1)

    # --------------- 3. Adaptive sampling weights -----------------
    k1, k2 = rad_args
    weights = (Y**k1) / jnp.mean(Y**k1) + k2
    p = weights / jnp.sum(weights)

    # --------------- 4. Weighted sampling -------------------------
    key, key_choice = randomjax.split(key)
    idx = randomjax.choice(key_choice, a=Ntest, shape=(Nint,), replace=False, p=p)

    return Xtest[idx], key

Nx=Ny=200
Xtest, x, y = generate_test(Nx, Ny)
Xtest = jnp.array(Xtest,dtype=jnp.float64)
x = jnp.array(x.flatten(),dtype=jnp.float64)[:,None]
y = jnp.array(y.flatten(),dtype=jnp.float64)[:,None]
u_gt = jnp.sin(a1 * jnp.pi * x) * jnp.sin(a2 * jnp.pi * y)
u_gt = u_gt.astype(jnp.float64)

key, eval_key = randomjax.split(key)
X_c, subkey = generate_inputs(Nint, subkey)
X_c = X_c.astype(jnp.float64)  # <-- CAST HERE
xc = X_c[:, 0:1]
yc = X_c[:, 1:2]

# Boundary Conditions (not used in the 2nd order part, but good practice)
# This data tuple is now consistently float64
train_data = xc, yc
feat_sizes = tuple([HIDDEN for _ in range(N_LAYERS)] + [FEATURES])
# make & init model
model = PINN(degree,degree_T,feat_sizes)
params = model.init(subkey, jnp.ones((Nint, 1), dtype=jnp.float64), jnp.ones((Nint, 1), dtype=jnp.float64))
params = jax.tree_util.tree_map(lambda x: x.astype(jnp.float64), params)
optimizers = {}

for key in params['params'].keys():
    if key=='g_fx':
        print('KART layer')
        optimizers[key]=optax.adam(optax.exponential_decay(lr0*args.lr_fact, decay_step, decay_rate, staircase=False))
    else:
        optimizers[key]=optax.adam(optax.exponential_decay(lr0, decay_step, decay_rate, staircase=False))

# Initialize optimizer states for each parameter group
states = {key: optim.init(params['params'][key]) for key, optim in optimizers.items()}

# forward & loss function
apply_fn = jax.jit(model.apply)
total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
print(total_params)

# +
Nprint = 100
rad_args = (k,c)
if EPOCHS> 0:
    epochs_adam = jnp.arange(Nprint,EPOCHS+Nprint,Nprint)
    l2error_adam = jnp.zeros(len(epochs_adam))
    time_adam = jnp.zeros(len(epochs_adam))
    start_adam = time.time()
    pbar = tqdm(range(1, EPOCHS + 1), desc='Training Progress')
# initialize grads container
    all_grads={
        'grad_bar_PDE':1,
        'grad_bar_BCs':1,
    }
    lamE = 1.
    lamB = 10.

    for e in pbar:
        if (e+1) % Nchange == 0:
            X, subkey = adaptive_rad(apply_fn, params, a1, a2, kval, Nint, rad_args, key=subkey)
            xc = X[:, 0:1]
            yc = X[:, 1:2]
            train_data = xc, yc
        # single run
        if (e+1) % Nprint == 0:
            #Compute errors
            u_pred_it=apply_fn(params, x, y)
            error = relative_l2(u_pred_it, u_gt)
            l2error_adam = l2error_adam.at[e//Nprint].set(error)
            time_adam = time_adam.at[e//Nprint].set(time.time() - start_adam)
        all_loss_it, gradient = apply_model(apply_fn, params,lamE,all_grads,*train_data)
        for key in params['params']:
            params, states[key] = update_model(key, optimizers[key], gradient, params, states[key])

# ### Second Order
cont=0
method='BFGS'
Nprint_bfgs=100
method_bfgs='SSBroyden2'
Nbfgs=100000
initial_weights, unflatten_func = flatten_util.ravel_pytree(params)
params_test=unflatten_func(initial_weights)
time_qn = time.time()

epochs_bfgs = jnp.arange(0,Nbfgs+Nprint_bfgs,Nprint_bfgs) #iterations bfgs list
epochs_bfgs+=EPOCHS
error_list = jnp.zeros(len(epochs_bfgs))
time_list = jnp.zeros(len(epochs_bfgs))

@partial(jax.jit, static_argnums=(0,))
def apply_model_2nd_Order(apply_fn, params, *train_data):
    # Unpack data
    x,y = train_data
    # Define residual function
    def r_E(params, x, y):
        # Compute u
        u = apply_fn(params, x,y)
        # Compute derivatives
        v_x = jnp.ones_like(x)
        v_y = jnp.ones_like(y)
        ux, uxx = hvp_fwdfwd(
        lambda x_val: apply_fn(params, x_val, y),
        (x,), (v_x,),
        return_primals=True
        )

        uy, uyy = hvp_fwdfwd(
        lambda y_val: apply_fn(params, x, y_val),
        (y,), (v_y,),
        return_primals=True
        )
        # Compute residuals using u, ut, and uxx
        return uxx + uyy + kval**2*u + (jnp.pi**2 * (a1**2 + a2**2)-kval**2)*jnp.sin(jnp.pi*a1*x)*jnp.sin(jnp.pi*a2*y)

    def loss_pde(params):
        # Compute residuals
        residuals = r_E(params, x, y)
        pde_loss = jnp.mean((residuals)**2)
        return pde_loss
    
    def loss(params):
        return loss_pde(params)

    loss_value=loss(params)
    return loss_value


@partial(jax.jit, static_argnums=(1, 2))
def loss_and_gradient(weights, N_arg, unflatten_func_arg, *train_data_tuple_arg):
    flat_jax_array = weights 
    params_current = unflatten_func_arg(flat_jax_array)
    loss_val_jax = apply_model_2nd_Order(N_arg, params_current, *train_data_tuple_arg)
    return loss_val_jax


@partial(jax.jit, static_argnames=("apply_fn", "unflatten_func", "Nint", "rad_args", "a1", "a2", "kval", "static_options"))
def bfgs_step(
    # DYNAMIC STATE (changes every iteration)
    initial_weights, H0, key,
    # DYNAMIC DATA (changes every iteration)
    xc, yc,
    # STATIC CONFIGURATION (fixed for the run)
    apply_fn, unflatten_func, Nint, rad_args, a1, a2, kval, static_options
):
    """
    Performs one full, JIT-compiled iteration of the BFGS training loop.
    """
    # 1. Handle the PRNG key state purely
    key, subkey = jax.random.split(key)

    # 2. Update Lambdas based on the current model
    params_it = unflatten_func(initial_weights)

    # 3. Sample a batch using the updated lambdas and the subkey
    # The original 'train_data' is (tc, xc, ti, xi, ui), so we recreate it with the new batch.
    # We assume ti, xi, ui are constants for this problem.
    X, subkey = adaptive_rad(apply_fn, params, a1, a2, kval,  Nint, rad_args, key=subkey)
    xc = X[:, 0:1]
    yc = X[:, 1:2]
    train_data = xc, yc


    current_train_data_tuple = xc, yc
    # 4. Prepare the full options dictionary for minimize
    current_options = dict(static_options)
    current_options['initial_H'] = H0

    # 5. Run the BFGS optimization for Nchange steps
    result = minimize(
        fun=loss_and_gradient,
        x0=initial_weights,
        args=(apply_fn, unflatten_func, *current_train_data_tuple),
        method='BFGS',
        options=current_options
    )

    # 6. Recycle the Hessian purely
    new_H0 = result.hess_inv
    new_H0 = (new_H0 + jnp.transpose(new_H0)) / 2
    try:
        # Cholesky can fail, so we wrap it. JAX's lax.cond is safer inside JIT.
        L = jnp.linalg.cholesky(new_H0)
        is_failed = jnp.any(jnp.isnan(L))
        final_H0 = jax.lax.cond(
            is_failed,
            lambda op: jnp.eye(op.shape[0], dtype=op.dtype),
            lambda op: op,
            operand=new_H0
        )
    except LinAlgError:
        final_H0 = jnp.eye(new_H0.shape[0], dtype=new_H0.dtype)
        jax.debug.print("I failed")

    # 7. Return all pieces of the new state
    return result.x, final_H0, key, result.fun, result.nit

# +
initial_weights, unflatten_func = flatten_util.ravel_pytree(params)
H0 = jnp.eye(len(initial_weights), dtype=jnp.float64)
key = jax.random.PRNGKey(SEED + 1) # Use a new seed to avoid reusing keys

# 2. Prepare Static Configuration
num_outer_iterations = Nbfgs // Nchange

# Create the static options dictionary WITHOUT the dynamic H0
static_options = {
    'maxiter': Nchange,
    'gtol': 1e-9,
    'update_method': "ssbroyden2",
    'initial_scale': False
}
# Convert to a hashable, immutable tuple for JIT
static_options_tuple = tuple(static_options.items())
key = jax.random.PRNGKey(42)
# --- THE JIT-POWERED LOOP ---
pbar = tqdm(range(num_outer_iterations), desc="BFGS Training")
# A. Prepare data for this iteration (This part runs in Python using NumPy)
effective_steps=0

for it in pbar:
    # C. Execute one full, compiled training step
    initial_weights, H0, key, loss_val, nit = bfgs_step(
        # Dynamic State
        initial_weights, H0, key,
        # Dynamic Data
        xc,yc,
        # Static Configuration
        apply_fn, unflatten_func, Nint, rad_args, a1, a2, kval, static_options_tuple
    )
    # Current step
    effective_steps=effective_steps+nit
    # The state variables (initial_weights, H0, key) are now updated for the next iteration.
    current_params_for_eval = unflatten_func(initial_weights)
    # D. Post-step analytics and logging (runs in Python)
    #Compute errors
    u_pred_it=apply_fn(current_params_for_eval, x, y)
    error = relative_l2(u_pred_it, u_gt)
    error_list = error_list.at[(it+1)//Nprint_bfgs].set(error)
    time_list = time_list.at[(it+1)//Nprint_bfgs].set(time.time() - time_qn)
    pbar.set_postfix({
        'It': effective_steps,
        'Loss': f'{loss_val:.3e}',
        'RL2': f'{float(error):.3e}',
    }) 
# -

jnp.save(args.Name + "QN_results",jnp.c_[epochs_bfgs,time_list,error_list])
u_pred_it=apply_fn(current_params_for_eval, x, y)
error_vec = u_pred_it - u_gt
print(jnp.max(jnp.abs(error_vec.flatten()))/jnp.max(jnp.abs(u_gt.flatten())))

