# -*- coding: utf-8 -*-
"""pinn_burgers_class.py
"""

import numpy as np
import tensorflow as tf
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
print('tf.__version__ ',tf.__version__)

default_training_steps=3000

"""##Compute the "exact" solution for the Burgers equation as a function of t and x


"""

# Parameters
nu = 0.01/np.pi  # Viscosity coefficient
xmin=-1.0
xmax=1.0

"""### Approach by finite differences

"""

"""
solve by finite differences Burgers equation
u_t + u u_x = nu u_xx
nu = 0.01/pi
x in [-1,1]
t in [0,1]

use a combination of finite difference schemes for the Burgers' equation:
    the Lax-Friedrichs scheme for the viscosity term
    and the Lax-Friedrichs scheme for the nonlinear term.
    This combination can provide stability and accuracy improvements over
    purely explicit schemes.

"""
use_FD=False
if(use_FD):
  # Parameters
  nx = 401   # Number of spatial grid points
  Tfinal=1.
  nt = 700   # Number of time steps
  dx = 2.0 / (nx - 1)  # Spatial step size
  dt = Tfinal/nt  # Time step size

  print('stability condition dt/dx<1 here dt/dx=',dt/dx)
  assert dt/dx<1.0, "error in the stability condition"
  # Spatial grid
  x = np.linspace(xmin, xmax, nx)
  tspan=np.linspace(0,Tfinal,nt+1)

  # Initial condition (sinusoidal) as a function and then as a vector
  def u0_function(xval):
    return  -np.sin(np.pi * xval)

  u0 = -np.sin(np.pi * x)
  solution=np.zeros((nt+1,nx))
  solution[0,:]=u0.copy()


  r=nu*dt/(dx**2)/2

  # Construct tridiagonal coefficient matrices for Crank-Nicholson scheme
  A = np.diag(np.ones(nx-1), 1) * (-r/2.0)
  A += np.diag(np.ones(nx)* (1 + r))
  A += np.diag(np.ones(nx-1), -1) * (-r/2.0)

  B = np.diag(np.ones(nx-1), 1) * r/2.0
  B += np.diag(np.ones(nx)* (1 - r))
  B += np.diag(np.ones(nx-1), -1) * r/2.0
  # Impose Dirichlet boundary conditions
  A[0, 0] = 1.0
  A[0, 1] = 0.0
  A[-1, -1] = 1.0
  A[-1, -2] = 0.0
  B[0, 0] = 1.0
  B[0, 1] = 0.0
  B[-1, -1] = 1.0
  B[-1, -2] = 0.0

  def crank_nicholsol_dt_over2_step(u):
      un = np.copy(u)  # Copy the solution from the previous time step
      # Apply Crank-Nicholson scheme for viscosity term (nu * d^2u/dx^2)
      # Right-hand side vector
      rhs = np.dot(B, un)
      rhs[0] = 0.0 #set boundary conditions
      rhs[-1] = 0.0    # Solve tridiagonal system
      return  np.linalg.solve(A, rhs)

  for n in range(nt):
      un = crank_nicholsol_dt_over2_step(solution[n,:])#initial dt/2 step CN for heat part

      # Apply Lax-Friedrichs scheme for nonlinear term (u * du/dx = d/dx(u^2/2) )
      f_plus = 0.5 * (un[2:]**2)
      f_minus = 0.5 * (un[:-2]**2)
      solution[n+1,1:-1] = 0.5 * (un[2:] + un[:-2]) - dt*(f_plus - f_minus)/(2*dx)

      # Apply homogeneous boundary conditions (u(-1, t) = u(1, t) = 0)
      solution[n+1,0] = 0.0
      solution[n+1,-1] = 0.0
      solution[n+1,:]=crank_nicholsol_dt_over2_step(solution[n+1,:])#final dt/2 step CN for heat part

  #plot it
  pstep=int(nt/5)
  figa=plt.figure('all solution',figsize=(4*1.5*2,4))
  figa.clf()
  plt.subplot(1,2,1)
  plt.xlabel('x')
  plt.plot(x,solution[::pstep,:].T,linewidth=2)
  plt.legend(['t='+str(np.round(tt,2)) for tt in  tspan[::pstep] ],loc='upper right')
  plt.subplot(1,2,2)
  plt.imshow(solution.T)
  plt.xlabel('t')
  plt.ylabel('x')
  plt.tight_layout()
  plt.savefig('burgers_exact.pdf')

  interp_exact_burgers_solution=RegularGridInterpolator( (tspan,x), solution,fill_value=None)
  interp_exact_burgers_solution=RegularGridInterpolator( (tspan,x), solution,fill_value=None)
  #exact_burgers_solution=lambda t,x : np.array( [interp_exact_burgers_solution((tt,xx)) for tt,xx in zip(np.array(t),np.array(x))]).reshape(t.shape)

  def exact_burgers_solution(t,x):
    if (np.array(t).ndim==0):#single input a float or a 0-dim array
      return float(interp_exact_burgers_solution((t,x)))
    else:
      return np.array( [interp_exact_burgers_solution((tt,xx)) for tt,xx in zip(t.flatten(),x.flatten()) ]).reshape(t.shape)

  #print(exact_burgers_solution(.2,.3))
  #print(exact_burgers_solution(.2*np.ones(3),.3*np.ones(3)).shape)
  initial_data_burgers = lambda x : exact_burgers_solution(x*0,x)

"""### Exact solution for Burgers' equation using Hopf-Cole transormation

We use the HC tranformation and the exact formula computed using Hermite quadrature.


"""

quadrature_order=50
hermite_points,hermite_weights=np.polynomial.hermite.hermgauss(quadrature_order)

def fcf(y):
    return np.exp(-np.cos(y*np.pi)/(2*np.pi*nu))

def f_hermite(x,eta):
    return fcf(x-eta)*u0fun(x-eta)
def f_hermite2(x,eta):
    return fcf(x-eta)

def u0fun(y):
    return -np.sin(np.pi*y)
def Du0fun(x):
    return -np.pi*np.cos(np.pi*x)
def DDu0fun(x):
    return np.pi*np.pi*np.sin(np.pi*x)

def exact_sol(t,x):
    if(np.abs(t)<1.e-10):
        return u0fun(x)+t*(nu*DDu0fun(x)-u0fun(x)*Du0fun(x))
    #    return u0fun(x)
    else:
        c=np.sqrt(4*nu*np.abs(t))
        res=np.sum(f_hermite(x,c*hermite_points)*hermite_weights)
        res2=np.sum(f_hermite2(x,c*hermite_points)*hermite_weights)
        return res/res2

def hopf_cole_quadrature_burgers_solution(t,x):
  if (np.array(t).ndim==0):#single input a float or a 0-dim array
    return float(exact_sol(t,x))
  else:
    return np.array( [exact_sol(tt,xx) for tt,xx in zip(t.flatten(),x.flatten()) ]).reshape(t.shape)
exact_burgers_solution=hopf_cole_quadrature_burgers_solution

# Parameters
nx = 100   # Number of spatial intervals
Tfinal=1.
nt = 5   # Number of time instants
dx = 2.0 /nx  # Spatial step size
dt = Tfinal/nt  # Time step size

xspan = np.linspace(xmin, xmax, nx+1)
tspan=np.linspace(0,Tfinal,nt+1)
Tgrid,Xgrid=np.meshgrid(tspan,xspan,indexing='ij')
solution=hopf_cole_quadrature_burgers_solution(Tgrid,Xgrid)

#plot it
pstep=int(nt/5)
#figa=plt.figure('all solution',figsize=(4*1.5*2,4))
figa=plt.figure('all solution',figsize=(4*1.5,4))
figa.clf()
#plt.subplot(1,2,1)
plt.xlabel('x')
plt.plot(xspan,solution[::pstep,:].T,linewidth=2)
plt.legend(['t='+str(np.round(tt,2)) for tt in  tspan[::pstep] ],loc='upper right')
#plt.subplot(1,2,2)
#plt.imshow(solution.T)
#plt.xlabel('t')
#plt.ylabel('x')
plt.tight_layout()
plt.savefig('burgers_exact.pdf')

"""## Class definition"""

default_eq_param = {'T':1.0,"xmin":-1.0,"xmax":1.0,"nu":0.01/np.pi}

class PhysicsInformedNeuralNetwork:
    def __init__(self, random_seed=123,sampling_lambda = 0.0,
                 eq_param =default_eq_param,
                 nb_hidden_layers = 8,neuron_per_layer = 20,test_case = 'Burgers',
                 input_dim = 2,output_dim = 1,nt = 50,nx=25,epochs = default_training_steps,
                 exact_solution_fun=exact_burgers_solution,
                 verbose=True,learning_rate=0.001,
                 weights_update_type=['lyapunov_lambda','causal','cst_lambda'][0]):
        tf.keras.backend.set_floatx("float64")
        np.random.seed(random_seed)

        self.random_seed = random_seed
        self.nx=nx#number of spacial segments
        self.nu=nu#viscosity in the Burgers' equation
        self.space_batch_size = nx+1#note that this includes extremities so there are nx+1
        self.nt= nt#number of time intervals
        self.time_batch_size = nt+1#note that this includes initial point so there are nt+1
        self.xmin=eq_param['xmin']#
        self.xmax=eq_param['xmax']#
        self.test_case = test_case + '_'+weights_update_type+'_nt'+str(nt)+'_nx'+str(nx)+\
        '_seed'+str(random_seed)+'_lr'+str(learning_rate)+'_T'+str(eq_param['T'])\
        +'_lmbd'+str(np.round(sampling_lambda,3))+'_iter'+str(epochs)

        self.nb_hidden_layers = nb_hidden_layers
        self.neuron_per_layer = neuron_per_layer
        self.eq_param = eq_param
        self.sampling_lambda = sampling_lambda
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.epochs = epochs
        #self.u0_function = u0_function
        self.verbose = verbose
        self.weights_update_type = weights_update_type
        self.learning_rate = learning_rate


        self.dx=(self.xmax-self.xmin)/self.nx
        self.xrange=np.linspace(self.xmin, self.xmax, self.nx+1)
        self.trange=np.linspace(0, self.eq_param['T'], self.nt+1)
        self.dt=self.eq_param['T']/(self.nt)
#        self.timestep=1./(self.nt)

        #self.u0val = np.array([self.u0_function(x) for x in self.xrange]).reshape(nx+1,1)
        if(self.weights_update_type =='lyapunov_lambda'):
          self.weights=np.ones((self.time_batch_size,1))/self.time_batch_size
        elif(self.weights_update_type =='causal'):
          self.weights=np.ones((self.time_batch_size,1))
        elif(self.weights_update_type =='cst_lambda'):
          self.weights=np.exp(-self.sampling_lambda*self.trange)[:,None]
          self.weights /=np.sum(self.weights)

        #grids are in order axis 0 = time, axis 1= space
        self.tgrid_2d=self.trange.reshape(self.nt+1,1)*np.ones((1,self.nx+1))
        self.xgrid_2d=np.ones((self.nt+1,1))* self.xrange.reshape(1,self.nx+1)
        self.t_grid_list=self.tgrid_2d.reshape(-1,1)#flatten()[:,None]
        self.x_grid_list=self.xgrid_2d.reshape(-1,1)#.flatten()[:,None]

        self.x_boundary_mask = self.x_grid_list*0+1.0 #1.0 if x not on boundary, 0.0=boundary
        for jj in range((self.nt+1)*(self.nx+1)):
          if( (np.abs(self.x_grid_list[jj,0]-self.xmin)<=1.e-10) or (np.abs(self.x_grid_list[jj,0]-self.xmax)<=1.e-10) ):
            self.x_boundary_mask[jj,0] = 0

        self.t_grid_list_tensor = tf.Variable(self.t_grid_list,trainable=False)
        self.x_grid_list_tensor = tf.Variable(self.x_grid_list,trainable=False)
        self.x_boundary_mask_tensor = tf.Variable(self.x_boundary_mask,trainable=False)

        self.learned_weights = self.weights.copy()
        self.learned_weights_history =[self.learned_weights.copy()]

        self.tf_variable_weights =tf.Variable(
            (self.weights.copy()+np.zeros((self.nt+1,self.nx+1))).reshape(-1,1),
            trainable=False)#the last one is to ensure good format

        self.weights_history =[self.weights.copy()]
        self.exp_weights=np.exp(-self.sampling_lambda*self.trange)[:,None]
        self.exp_weights /=np.sum(self.exp_weights)
        self.weights_contrast=np.max(self.weights)/(1.0e-10+np.min(self.weights))
        self.exact_solution_fun=exact_solution_fun

        self.exact_sol=self.exact_solution_fun(self.tgrid_2d,self.xgrid_2d)


        self.loss_list=[]

        self.time_sample_tensor = tf.Variable(
            self.trange.reshape(self.time_batch_size,1),trainable=False
        )
        self.initial_time_tensor = tf.Variable(np.zeros((1,1)),trainable=False)
        self.lambda_estimators=None
        self.build_model()

        self.states=self.u(self.t_grid_list_tensor,self.x_grid_list_tensor).numpy().reshape(self.nt+1,self.nx+1)#formatted as the solution
        self.f_of_states=self.tf_generator_funct(self.t_grid_list_tensor,self.x_grid_list_tensor).numpy().reshape(self.states.shape)

        eq_loss = self.equation_loss(self.t_grid_list_tensor,self.x_grid_list_tensor)
        loss=eq_loss[0]
        self.raw_eq_error= eq_loss[1].numpy()
        self.previous_raw_eq_error=self.raw_eq_error.copy()

        self.previous_states=self.states.copy()
        self.previous_f_of_states=self.f_of_states.copy()
        self.previous_raw_eq_error=self.raw_eq_error.copy()

        self.final_state_error=[np.sqrt(self.dx)*np.linalg.norm(self.states[-1,:]-self.exact_sol[-1,:])]
        self.relative_L2_error=[np.sqrt(np.sum((self.states-self.exact_sol)**2))/np.sqrt(np.sum(self.exact_sol**2))]

    def build_model(self):
        activation_function = "tanh"

        input_layer = tf.keras.layers.Input(shape=(self.input_dim,))
        current_map = input_layer

        for _ in range(self.nb_hidden_layers):
            current_map = tf.keras.layers.Dense(
                self.neuron_per_layer,
                activation=activation_function,
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.random_seed)
            )(current_map)

        output_layer = tf.keras.layers.Dense(
            self.output_dim,
            activation=None,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.random_seed)
        )(current_map)

        self.model = tf.keras.Model(input_layer, output_layer)
        self.model_weights_store=[]#create a list to store model weights
        self.model.summary()
        self.store_model_weights()

    @tf.function
    def initial_condition_function(self,x):
      return -tf.math.sin(np.pi * x)


    @tf.function
    def u(self,t, x):
        u = self.model(tf.concat([t, x], axis=1))- self.model(tf.concat([t*0, x], axis=1))+self.initial_condition_function(x)
        return u

    @tf.function
    def equation_loss(self, t_list,x_list):
        u_val = self.u(t_list,x_list)
        u_t = tf.gradients(u_val, t_list)[0]
        u_x = tf.gradients(u_val, x_list)[0]
        u_xx = tf.gradients(u_x, x_list)[0]
        brute_eq_error=  u_t + u_val*u_x - self.nu*u_xx
        E = self.x_boundary_mask_tensor*self.tf_variable_weights * tf.square(brute_eq_error)* self.eq_param['T']#should integrate with dx=T/nx and here is only mean i.e. 1/nx
        BC= tf.square(u_val)*(1-self.x_boundary_mask_tensor)*self.tf_variable_weights#boundary data
        return tf.reduce_mean(E),brute_eq_error,tf.reduce_mean(BC)


    def update_weights(self):
      """set weights :
      a/ depending on brute equation loss (causal using sampling_lambda parameter)
      b/ or Lyapunov (needs fresh states and F(states) )
      c/ or exponential cummulative with constant lambda (using sampling_lambda parameter)
      """
      if(self.weights_update_type =='cst_lambda'):

        self.weights=np.exp(-self.sampling_lambda*self.trange)
        self.weights=self.weights.reshape(self.time_batch_size,1)
        self.weights /=np.sum(self.weights)#normalize if needed

      elif(self.weights_update_type =='causal'):

        errors=-self.sampling_lambda* np.cumsum(np.sum(self.raw_eq_error.reshape(self.nt+1,-1)**2,axis=1))*self.dx
        errors=errors.reshape(self.time_batch_size,1)
        self.weights[0,0]=1.0
        self.weights[1:,0] = np.exp(errors[:-1,0])#shift as per protocol
        self.weights=self.weights.reshape(self.time_batch_size,1)

      elif(self.weights_update_type =='lyapunov_lambda'):

        self.lambda_estimators=np.sum((self.f_of_states-self.previous_f_of_states*0)*
              (self.states-self.previous_states*0),axis=1)/(1.0e-10+np.sum(
                  (self.states-self.previous_states*0)**2,axis=1))#no need for dx parameter this will simplify anyway

        weights=(np.sum(self.lambda_estimators)-np.cumsum(self.lambda_estimators))*self.dt/self.eq_param['T']
        weights=np.exp(weights-np.max(weights))#we substract the max value
        #to have a well defined exponential, no underflow; this does not
        #matter because next line cancels the effect of this overall constant
        weights /=np.sum(weights)#normalization
        weights=weights.reshape(self.time_batch_size,1)#shape (-1,1), ndim=2
        self.learned_weights=weights.copy()

        self.learned_weights_history.append(self.learned_weights.copy())
        self.learned_weights_contrast=np.max(weights)/np.min(weights)
        #here takes place the updating
        self.weights=self.learned_weights.copy()

      self.weights_history.append(self.weights.copy())
      self.tf_variable_weights.assign( (self.weights.copy()+np.zeros((self.nt+1,self.nx+1))).reshape(-1,1))
      self.weights_contrast=np.max(self.weights)/(1.0e-10+np.min(self.weights))

      self.final_state_error.append(np.sqrt(self.dx)*np.linalg.norm(self.states[-1,:]-self.exact_sol[-1,:]))

    @tf.function
    def tf_generator_funct(self,t, x):
        u0 = self.u(t, x)
        u_x = tf.gradients(u0, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        return self.nu*u_xx-u0*u_x

    def train(self,epochs=None,verbose=None,reset_loss_list=False):
        if epochs is None:
          epochs = self.epochs  # Use class attribute as default value
        else:
          self.epochs=epochs
        self.test_case=self.test_case.split(sep='_iter')[0]+'_iter'+str(epochs)
        print('start training ',self.test_case,flush=True)
        if verbose is None:
              verbose = self.verbose  # Use class attribute as default value
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)# Keras default = 0.001
        if(reset_loss_list):
          self.loss_list=[]

        for epoch in range(epochs):
            with tf.GradientTape(persistent=True) as tape:
#                eq_loss = self.equation_loss(self.time_sample_tensor)
                eq_loss = self.equation_loss(self.t_grid_list_tensor,self.x_grid_list_tensor)
                loss=eq_loss[0]+1.0*eq_loss[2]
                self.previous_raw_eq_error=self.raw_eq_error.copy()
                self.raw_eq_error= eq_loss[1].numpy()
            g = tape.gradient(loss, self.model.trainable_weights)
            self.loss_list.append(loss.numpy())

            self.previous_states=self.states.copy()
            self.states=self.u(self.t_grid_list_tensor,self.x_grid_list_tensor).numpy().reshape(self.previous_states.shape)
            self.previous_f_of_states=self.f_of_states.copy()
            self.f_of_states=self.tf_generator_funct(self.t_grid_list_tensor,self.x_grid_list_tensor).numpy().reshape(self.states.shape)
            self.relative_L2_error.append(np.sqrt(np.sum((self.states-self.exact_sol)**2))/np.sqrt(np.sum(self.exact_sol**2)))

            self.update_weights()

            if not epoch % 50 or epoch == epochs - 1:
#                print(f"{epoch:4} {loss.numpy():.9f}"+f" contrast={self.weights_contrast:.2f}"
#                      +f" L2 rel err={self.relative_L2_error[-1]:.9f}",flush=True)
                print(f"{epoch:4} {loss.numpy():.9f}"+f" contrast={self.weights_contrast:.2f}"
                      ,flush=True)

            self.store_model_weights()
            opt.apply_gradients(zip(g, self.model.trainable_weights))
        self.backup_class_data()
        return epoch

    def predict(self, new_input):
        # Make predictions using the trained model
        new_input = np.array(new_input)
        new_input = new_input.reshape(-1,self.input_dim)
        prediction = self.u(new_input)
        return prediction.numpy()

    def plot_loss_convergence(self):
      print('test case: ',self.test_case,'last loss=',self.loss_list[-1])
      if(len(self.loss_list)>0):
        plt.figure('loss',figsize=(8,4))
        plt.clf()
        plt.semilogy(range(len(self.loss_list)), self.loss_list,linewidth=2)
        plt.title(self.test_case)
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.tight_layout()
        plt.savefig('losses_'+self.test_case+'.pdf')

      plt.figure('final_state_error',figsize=(8,4))
      plt.clf()
      plt.semilogy(range(len(self.final_state_error)), self.final_state_error,linewidth=2)
      plt.ylabel("error")
      plt.xlabel("iteration")
      plt.tight_layout()
      plt.savefig('final_error_'+self.test_case+'.pdf')

    def plot_train_results(self):
      print(self.test_case)
      no_snapshots=5
      pstep=int(self.nt/no_snapshots)
      plt.figure('solution',figsize=(3.5*no_snapshots, 3.5), dpi=100)
      plt.clf()
      for ii in range(no_snapshots):
        plt.subplot(1,no_snapshots,ii+1)
        plt.xlabel('x')
        plt.plot(self.xrange,self.states[pstep*ii,:],linewidth=2)
        plt.plot(self.xrange,self.exact_sol[pstep*ii,:],linewidth=2)
        plt.legend(['PINN','exact'],loc='upper right')
      plt.tight_layout()
      plt.savefig('solution_'+self.test_case+'.pdf')

    def plot_last_weights(self):
      plt.figure('last weights',figsize=(6,3), dpi=100)
      plt.clf()
      plt.plot(self.trange, self.weights,'-.',label='weights',linewidth=4)
      plt.legend()
      plt.tight_layout()
      plt.savefig('weights_'+self.test_case+'.pdf')

    def plot_last_estimators(self):
      if(self.lambda_estimators is not None):
        plt.figure('last estimators',figsize=(6,3), dpi=100)
        plt.clf()
        plt.subplot(1,2,1)
        plt.plot(self.lambda_estimators,linewidth=4)
        plt.legend(['estimators'])
        plt.subplot(1,2,2)
        plt.plot(-np.cumsum(self.lambda_estimators)*self.dt/self.eq_param['T'],linewidth=4)
        plt.legend(['cumsum'])
        plt.tight_layout()
        plt.savefig('estimators_'+self.test_case+'.pdf')

    def store_model_weights(self):
      self.model_weights_store.append([a.numpy() for a in self.model.trainable_weights])

    def backup_class_data(self,filename=None):
      if filename is None:
        filename=self.test_case.replace('.','_')
      #get all data
      data_dict={name:getattr(self,name) for name in dir(self) if type(getattr(self,name))
        in [bool,dict, float, int, list, np.float64, np.ndarray, str]
                and name[0] != '_'}
#      data_dict['model_weights']=[a.numpy() for a in self.model.trainable_weights]#already in model_weights_store!
      np.savez(filename+".npz",data_dict)
      #to load:
      #loaded_data = np.load(self.test_case.replace('.','_')+".npz",allow_pickle=True)
      #ald=loaded_data[loaded_data.files[0]]
      #data is in (example xmin):   ald.item()['xmin']

"""## Numerical tests

"""


"""### Lyapunov class (learning)"""

pinn_learning = PhysicsInformedNeuralNetwork(random_seed=1234,
                                             sampling_lambda=0.0,weights_update_type=['lyapunov_lambda','causal','cst_lambda'][0],learning_rate=0.001)#random_seed=1234

import time
start_time = time.process_time()
print("start time",start_time)
pinn_learning.train(default_training_steps)
end_time = time.process_time()
print("end time",end_time)
print('total time ',end_time - start_time, "seconds")

pinn_learning.plot_last_estimators()
pinn_learning.plot_last_weights()
pinn_learning.plot_loss_convergence()
pinn_learning.plot_train_results()

"""#### Plot model weight evolution

set "True" in first line to execute
"""

"""### Causal model class"""

pinn_causal = PhysicsInformedNeuralNetwork(random_seed=1234,weights_update_type=['lyapunov_lambda','causal','cst_lambda'][1],learning_rate=0.01)#random_seed=1234
niter=default_training_steps
eps_list=[0.01,0.1,1,10,100]
for eps in eps_list:
  print('epsilon=',eps)
  pinn_causal.sampling_lambda=eps
  loss_history_causal = pinn_causal.train( int(niter/len(eps_list)))
  pinn_causal.plot_last_weights()
  plt.pause(1.0)
  pinn_causal.plot_loss_convergence()
  plt.pause(1.0)
  pinn_causal.plot_train_results()
  plt.pause(1.0)

iter_per_eps=int((len(pinn_causal.weights_history)-1)/len(eps_list))
plot_per_eps=5
print(len(pinn_causal.weights_history),len(eps_list),iter_per_eps)
plt.figure('weights history',figsize=(3.5*len(eps_list),3.5))
for ii in range(len(eps_list)):
  plt.subplot(1,len(eps_list),ii+1)
  for jj in range(plot_per_eps):
    plot_idx=1+iter_per_eps*ii+jj*int(iter_per_eps/(plot_per_eps-1))
    if(plot_idx <=len(pinn_causal.weights_history)-1):
      plt.plot(pinn_causal.trange,pinn_causal.weights_history[plot_idx],label="iter "+str(jj*int(iter_per_eps/(plot_per_eps-1))),linewidth=2)
  plt.legend()
plt.tight_layout()
plt.savefig("weights_history_"+pinn_causal.test_case+".pdf")

pinn_causal.plot_last_weights()
pinn_causal.plot_loss_convergence()
pinn_causal.plot_train_results()

"""### Lambda constant class l=0"""

pinn0 = PhysicsInformedNeuralNetwork(random_seed=1234,sampling_lambda=0.0,weights_update_type=['lyapunov_lambda','causal','cst_lambda'][-1],
                                           learning_rate=0.01,epochs=default_training_steps)#random_seed=1234
pinn0.train(default_training_steps)
pinn0.plot_last_estimators()
pinn0.plot_last_weights()
pinn0.plot_loss_convergence()
pinn0.plot_train_results()
