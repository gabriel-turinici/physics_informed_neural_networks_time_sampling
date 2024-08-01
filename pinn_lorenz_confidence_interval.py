# -*- coding: utf-8 -*-
"""pinn_lorenz_class_v1_11.py
"""

import numpy as np
import pickle
import tensorflow as tf
from scipy.integrate import odeint
import matplotlib.pyplot as plt
print('tf.__version__ ',tf.__version__)

"""## Class definition"""

default_eq_param = {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0, 'T': 0.5,
                    "state_labels":['x(t)','y(t)','z(t)']}


class PhysicsInformedNeuralNetwork:
    def __init__(self, random_seed=1234,sampling_lambda = 0.0,
                 eq_param = default_eq_param,
                 nb_hidden_layers = 5,neuron_per_layer = 20,test_case = 'Lorenz',
                 input_dim = 1,output_dim = 3,time_batch_size = 256,epochs = 2000,
                 u0=np.ones(3),verbose=True,learning_rate=0.001,
                 weights_update_type=['lyapunov_lambda','causal','cst_lambda'][0]):
        tf.keras.backend.set_floatx("float64")
        np.random.seed(random_seed)

        self.random_seed = random_seed
        self.test_case = test_case + '_'+weights_update_type+'_nt'+str(time_batch_size)+\
        '_seed'+str(random_seed)+'_lr'+str(learning_rate)+'_T'+str(eq_param['T'])\
        +'_lmbd'+str(np.round(sampling_lambda,3))+'_iter'+str(epochs)
        self.nb_hidden_layers = nb_hidden_layers
        self.neuron_per_layer = neuron_per_layer
        self.eq_param = eq_param
        self.sampling_lambda = sampling_lambda
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.time_batch_size = time_batch_size
        self.epochs = epochs
        self.u0 = u0
        self.verbose = verbose
        self.weights_update_type = weights_update_type
        self.learning_rate = learning_rate
        self.trange=np.linspace(0, self.eq_param['T'], self.time_batch_size)
#        self.timestep=self.eq_param['T']/(self.time_batch_size-1)
        self.timestep=1./(self.time_batch_size-1)
        if(self.weights_update_type =='lyapunov_lambda'):
          self.weights=np.ones((self.time_batch_size,1))/self.time_batch_size
        elif(self.weights_update_type =='causal'):
          self.weights=np.ones((self.time_batch_size,1))
        elif(self.weights_update_type =='cst_lambda'):
          self.weights=np.exp(-self.sampling_lambda*self.trange)[:,None]
          self.weights /=np.sum(self.weights)

        self.learned_weights = self.weights.copy()
        self.learned_weights_history =[self.learned_weights.copy()]
        self.tf_variable_weights =tf.Variable(self.weights.copy())
        self.weights_history =[self.weights.copy()]
        self.exp_weights=np.exp(-self.sampling_lambda*self.trange)[:,None]
        self.exp_weights /=np.sum(self.exp_weights)
        self.weights_contrast=np.max(self.weights)/(1.0e-10+np.min(self.weights))
        self.exact_sol=odeint(self.generator_funct,self.u0,t=self.trange,tfirst=True)
        self.f_exact_sol=np.array([self.generator_funct(tt,st)
                    for tt,st in zip(self.trange, self.exact_sol)])

        self.loss_list=[]
        self.state_labels=self.eq_param['state_labels']

        self.time_sample_tensor = tf.Variable(
            self.trange.reshape(self.time_batch_size, self.input_dim)
        )
        self.initial_time_tensor = tf.Variable(np.zeros((1, self.input_dim)))
        self.build_model()

        self.states=self.u(self.time_sample_tensor).numpy()
        self.f_of_states=np.array([self.generator_funct(tt,st)
                    for tt,st in zip(self.trange, self.states)])
#        self.raw_eq_error=self.exact_sol*0

        self.f_of_initial_states=self.f_of_states.copy()
        self.initial_states=self.states.copy()

        eq_loss = self.equation_loss(self.time_sample_tensor)
        loss=eq_loss[0]
        self.raw_eq_error= eq_loss[1].numpy()
        self.previous_raw_eq_error=self.raw_eq_error.copy()

        self.previous_states=self.states.copy()
        self.previous_f_of_states=self.f_of_states.copy()
        self.previous_raw_eq_error=self.raw_eq_error.copy()

        self.raw_lambda_estimators=[]
        self.exact_lambda=[]

        self.final_state_error=[ np.linalg.norm(self.states[-1,:]-self.exact_sol[-1,:])]

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
    def u(self, t):
#        print('t.shape as input to u',t.shape,flush=True)
        u_val = self.model(t)
        return u_val - self.model(self.initial_time_tensor) + self.u0
        #attention at self.u0 : when u0 is not a constant or a constant vector gradient will not flow through this value,
        # do not use for parametric functions of where you need a derivative. Here the derivative with respect to u0
        # will be false (but is not needed in this context)
    @tf.function
    def equation_loss(self, t):
        u_val = self.u(t)
        u_t = tf.concat([tf.gradients(u_val[..., i], t)[0] for i in range(self.output_dim)], axis=1)
        brute_eq_error=u_t - self.tf_generator_funct(t, u_val)
        print(brute_eq_error.shape,flush=True)
        eq_error_sq=tf.square(brute_eq_error)
#        return tf.exp(self.sampling_lambda * tf.reduce_mean(eq_error))

  #      E = tf.exp(-self.sampling_lambda * tf.cumsum(eq_error,axis=0) ) * eq_error
  #      return tf.reduce_mean(E)/self.sampling_lambda

#        E = self.weights * tf.square(eq_error)
        E = self.tf_variable_weights *eq_error_sq
        return tf.reduce_mean(E),brute_eq_error

#        E = tf.exp(-self.sampling_lambda * t / 2.0) * eq_error
#        return tf.reduce_mean(tf.square(E))

    def update_weights(self):
      """set weights :
      a/ depending on brute equation loss (causal using sampling_lambda parameter)
      b/ or Lyapunov (needs fresh states and F(states) )
      c/ or exponential cummulative with constant lambda (using sampling_lambda parameter)
      """
      if(self.weights_update_type =='cst_lambda'):

        self.weights=np.exp(-self.sampling_lambda*self.trange)
        self.weights=self.weights.reshape((self.time_batch_size,1))
        self.weights /=np.sum(self.weights)#normalize if needed

      elif(self.weights_update_type =='causal'):

        errors=-self.sampling_lambda* np.cumsum(np.sum(self.raw_eq_error**2,axis=1))
        errors=errors.reshape((self.time_batch_size,1))
        self.weights[0,0]=1.0
        self.weights[1:,0] = np.exp(errors[:-1,0])#shift as per protocol
        #self.weights = np.exp(errors-np.max(errors))#shift to compute exp stably; will renormalize later
        #note that here no normalization takes place !!!
        self.weights=self.weights.reshape((self.time_batch_size,1))

      elif(self.weights_update_type =='lyapunov_lambda'):

        if(False):
          self.lambda_estimators=np.sum((self.f_of_states-self.f_of_initial_states)*
                (self.states-self.initial_states),axis=1)/(1.0e-10+np.sum(
                    (self.states-self.initial_states)**2,axis=1))
        if(True):
          self.lambda_estimators=np.sum(self.f_of_states*self.states,axis=1)/ \
                                 (1.0e-10+np.sum(self.states**2,axis=1))

        self.raw_lambda_estimators.append(self.lambda_estimators.copy())

        self.exact_lambda.append(np.sum((self.f_of_states-self.f_exact_sol)*
              (self.states-self.exact_sol),axis=1)/(1.0e-10+np.sum(
                  (self.states-self.exact_sol)**2,axis=1)))
        #transformation of lambda estimators
#        self.lambda_estimators = np.maximum(self.lambda_estimators,0)

        weights=(np.sum(self.lambda_estimators)-np.cumsum(self.lambda_estimators))*self.timestep
        weights=np.exp(weights-np.max(weights))#we substract the max value
        #to have a well defined exponential, no underflow; this does not
        #matter because next line cancels the effect of this overall constant
        weights /=np.sum(weights)#normalization
        weights=weights.reshape((self.time_batch_size,1))#shape (...,1), ndim=2
        self.learned_weights=weights.copy()
        ###self.learned_weights +=(weights-self.learned_weights)/(1+epoch)

        #option to make distribution contain points everywhere ... make an average with uniform distribution
  #            self.learned_weights +=0.2*(1.0/self.time_batch_size-self.learned_weights)
        #self.learned_weights /=np.sum(self.learned_weights)

        self.learned_weights_history.append(self.learned_weights.copy())
        self.learned_weights_contrast=np.max(weights)/(1.0e-10+np.min(weights))
        #here takes place the updating
        self.weights=self.learned_weights.copy()

      self.weights_history.append(self.weights.copy())
      self.tf_variable_weights.assign(self.weights.copy())
      self.weights_contrast=np.max(self.weights)/(1.0e-10+np.min(self.weights))

      self.final_state_error.append(np.linalg.norm(self.states[-1,:]
                                                    -self.exact_sol[-1,:]))


    @tf.function
    def tf_generator_funct(self, t, x):
        xl, yl, zl = x[..., 0], x[..., 1], x[..., 2]
        return tf.transpose(tf.stack([
            self.eq_param['sigma'] * (yl - xl),
            xl * (self.eq_param['rho'] - zl) - yl,
            xl * yl - self.eq_param['beta'] * zl
        ]))

    def generator_funct(self,t,x):
      xl,yl,zl=x
      return np.array([self.eq_param['sigma']*(yl-xl) ,xl*(self.eq_param['rho']-zl)-yl,
                        xl*yl-self.eq_param['beta']*zl])


    def train(self,epochs=None,verbose=None,reset_loss_list=False):
        if epochs is None:
          epochs = self.epochs  # Use class attribute as default value
        else:
          self.epochs=epochs
          self.test_case=self.test_case.split(sep='_iter')[0]+'_iter'+str(epochs)
        if verbose is None:
              verbose = self.verbose  # Use class attribute as default value
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)# Keras default = 0.001
        if(reset_loss_list):
          self.loss_list=[]

        for epoch in range(epochs):
            with tf.GradientTape(persistent=True) as tape:
                eq_loss = self.equation_loss(self.time_sample_tensor)
                loss=eq_loss[0]
                self.previous_raw_eq_error=self.raw_eq_error.copy()
                self.raw_eq_error= eq_loss[1].numpy()
            g = tape.gradient(loss, self.model.trainable_weights)
            self.loss_list.append(loss.numpy())

            self.previous_states=self.states.copy()
            self.states=self.u(self.time_sample_tensor).numpy()

            self.previous_f_of_states=self.f_of_states.copy()
            self.f_of_states=np.array([self.generator_funct(tt,st)
                                      for tt,st in zip(self.trange, self.states)])

            self.update_weights()

            if not epoch % 50 or epoch == epochs - 1:
                print(f"{epoch:4} {loss.numpy():.9f} contrast={self.weights_contrast:.2f}",flush=True)

            self.store_model_weights()
            opt.apply_gradients(zip(g, self.model.trainable_weights))
        self.save_object()
        return epoch

    def predict(self, new_input):
        # Make predictions using the trained model
        new_input = np.array(new_input)
        new_input =new_input.reshape(-1,self.input_dim)
        prediction = self.u(new_input)
        return prediction.numpy()

    def save_object(self):
        with open(self.test_case.replace('.','_')+".pkl", 'wb') as file:
          pickle.dump(self,file)

        #with open('file.pkl', 'rb') as file:
        #    # Call load method to deserialze
        #    myvar = pickle.load(file)

    def plot_loss_convergence(self):
      print('test case: ',self.test_case,'last loss=',self.loss_list[-1])
      if(self.loss_list is not None):
        plt.figure('loss',figsize=(8,4))
        plt.clf()
        plt.semilogy(range(len(self.loss_list)), self.loss_list,linewidth=2)
        plt.ylabel("loss")
        plt.xlabel("iteration")
        plt.tight_layout()
        plt.savefig('losses_'+self.test_case+'.pdf')

      if(self.final_state_error is not None):
        plt.figure('final_state_error',figsize=(8,4))
        plt.clf()
        plt.semilogy(range(len(self.final_state_error)), self.final_state_error,linewidth=2)
        plt.ylabel("error")
        plt.xlabel("iteration")
        plt.tight_layout()
        plt.savefig('final_error_'+self.test_case+'.pdf')

    def plot_train_results(self):
      print(self.test_case)
      local_error = [np.sqrt(self.equation_loss(np.array([[tt]]))[0].numpy()) for tt in self.trange]
      plt.figure('solution',figsize=(10.5, 3.5), dpi=300)
      plt.clf()
      sol = self.predict(self.trange)
      plt.subplot(1,3,1)
      plt.xlabel("t")
      for pi in range(self.output_dim):
          plt.plot(self.trange, sol[:,pi],'-.',label=self.state_labels[pi]+' (PINN)',linewidth=4)
      plt.legend()
      plt.subplot(1,3,2)
      plt.xlabel("t")
      for pi in range(self.output_dim):
          plt.plot(self.trange, sol[:,pi],'-.',label=self.state_labels[pi]+' (PINN)',linewidth=4)
      for pi in range(self.output_dim):
          plt.plot(self.trange, self.exact_sol[:,pi],label=self.state_labels[pi]+' (exact)',linewidth=4)
      plt.legend()
      plt.subplot(1,3,3)
      plt.xlabel("t")
      plt.plot(self.trange,local_error,label='equation error',linewidth=4)
      plt.legend()
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
        plt.plot(-np.cumsum(self.lambda_estimators)*self.timestep,linewidth=4)
        plt.legend(['cumsum'])
        plt.tight_layout()
        plt.savefig('estimators_'+self.test_case+'.pdf')

      plt.figure('raw_estimators',figsize=(6,3), dpi=100)
      plt.clf()
      if(self.raw_lambda_estimators is not None):
        plt.plot(self.raw_lambda_estimators[-1],linewidth=4,label='estimators')
      if(self.exact_lambda is not None):
        plt.plot(self.exact_lambda[-1],linewidth=4,label='exact')
        plt.legend()
      plt.tight_layout()
      plt.savefig('estimators_vs_exact'+self.test_case+'.pdf')




    def store_model_weights(self):
      self.model_weights_store.append([a.numpy() for a in self.model.trainable_weights])

"""## Numerical tests

"""

"""### Confidence interval for the loss"""

run_ci=True#set to True to run the following, note that it is LONG
if(run_ci):
  nr_runs=1000
  nbiter=2000
  loss_array=np.zeros((nbiter,nr_runs))
  for ii in range(nr_runs):
    print('run number ',ii)
    c_pinn_learning = PhysicsInformedNeuralNetwork(random_seed=ii,
        weights_update_type=['lyapunov_lambda','causal','cst_lambda'][0],learning_rate=0.01)#random_seed=1234

    c_pinn_learning.train(nbiter)
    loss_array[:,ii]=np.array(c_pinn_learning.loss_list)
  np.savez('loss_ci.npz',loss_array)


if(run_ci):
  print('test case: ',c_pinn_learning.test_case)
  plt.figure('lr loss CI',figsize=(8,4))
  plt.clf()
  for ii in range(nr_runs):
    plt.semilogy(range(nbiter), loss_array[:,ii],linewidth=0.1)
  plt.semilogy(range(nbiter), np.quantile(loss_array,0.025,axis=1),linewidth=4,label='lower IC')
  plt.semilogy(range(nbiter), np.quantile(loss_array,0.5,axis=1),linewidth=4,label='median')
  plt.semilogy(range(nbiter), np.quantile(loss_array,0.975,axis=1),linewidth=4,label='upper IC')
  plt.ylabel("loss")
  plt.xlabel("iteration")
  plt.legend()
  plt.tight_layout()
  plt.savefig('confidence_intervals_losses_lorenz_lyapunov.pdf')
