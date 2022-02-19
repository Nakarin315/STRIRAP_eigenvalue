import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.integrate import odeint

###############################################################################
## Ref: Beterov, I. I., et al. "Application of adiabatic passage in Rydberg atomic ensembles for quantum information processing." Journal of Physics B: Atomic, Molecular and Optical Physics 53.18 (2020): 182001.
###############################################################################


###############################################################################
## Define odeint for a complex number
def odeintz(func, z0, t, **kwargs):
    """An odeint-like function for complex valued differential equations."""

    # Disallow Jacobian-related arguments.
    _unsupported_odeint_args = ['Dfun', 'col_deriv', 'ml', 'mu']
    bad_args = [arg for arg in kwargs if arg in _unsupported_odeint_args]
    if len(bad_args) > 0:
        raise ValueError("The odeint argument %r is not supported by "
                         "odeintz." % (bad_args[0],))

    # Make sure z0 is a numpy array of type np.complex128.
    z0 = np.array(z0, dtype=np.complex128, ndmin=1)

    def realfunc(x, t, *args):
        z = x.view(np.complex128)
        dzdt = func(z, t, *args)
        # func might return a python list, so convert its return
        # value to an array with type np.complex128, and then return
        # a np.float64 view of that array.
        return np.asarray(dzdt, dtype=np.complex128).view(np.float64)

    result = odeint(realfunc, z0.view(np.float64), t, **kwargs)

    if kwargs.get('full_output', False):
        z = result[0].view(np.complex128)
        infodict = result[1]
        return z, infodict
    else:
        z = result.view(np.complex128)
        return z
###############################################################################





## Define pulsed Rabi frequencies
hbar=1.0545718e-34*0+1;

tf =10;
t=np.linspace(-6, 6, 1000)*1e-6;

Omega_S0=2*np.pi*10e6;
Omega_P0=2*np.pi*10e6;
t1=-1e-6;
t2=1e-6;
w=1e-6;
delta=10*2*np.pi*1e6;

# ###############################################################################
# def H(t): ### Hamiltonian of one atom 

#     Omega_S =Omega_S0*np.exp(-1*(t-t1)**2/(2*w**2));
#     Omega_P =Omega_P0*np.exp(-1*(t-t2)**2/(2*w**2));
#     A=(hbar/2)*np.array([[0,Omega_P,0],
#                 [Omega_P,2*delta,Omega_S],
#                 [0,Omega_S,0]]);
#     return A

###############################################################################
def H(t): ### Hamiltonian of wo atom 

    Omega_S =Omega_S0*np.exp(-1*(t-t1)**2/(2*w**2));
    Omega_P =Omega_P0*np.exp(-1*(t-t2)**2/(2*w**2));
    A=(hbar/2)*np.array([[0,Omega_P,0,Omega_P,0,0,0,0],
                [Omega_P,2*delta,Omega_S,0,Omega_P,0,0,0],
                [0,Omega_S,0,0,0,Omega_P,0,0],
                [Omega_P,0,0,2*delta,Omega_P,0,Omega_S,0],
                [0,Omega_P,0,Omega_P,4*delta,Omega_S,0,Omega_S],
                [0,0,Omega_P,0,Omega_S,2*delta,0,0],
                [0,0,0,Omega_S,0,0,0,Omega_P],
                [0,0,0,0,Omega_S,0,Omega_P,2*delta]]);
    return A

# ###############################################################################
# def dydt(y0, t): ### Hamiltonian of one atom 
#     Omega_S =Omega_S0*np.exp(-1*(t-t1)**2/(2*w**2));
#     Omega_P =Omega_P0*np.exp(-1*(t-t2)**2/(2*w**2));
#     A=(hbar/2)*np.array([[0,Omega_P,0,Omega_P,0,0,0,0],
#                 [Omega_P,2*delta,Omega_S,0,Omega_P,0,0,0],
#                 [0,Omega_S,0,0,0,Omega_P,0,0],
#                 [Omega_P,0,0,2*delta,Omega_P,0,Omega_S,0],
#                 [0,Omega_P,0,Omega_P,4*delta,Omega_S,0,Omega_S],
#                 [0,0,Omega_P,0,Omega_S,2*delta,0,0],
#                 [0,0,0,Omega_S,0,0,0,Omega_P],
#                 [0,0,0,0,Omega_S,0,Omega_P,2*delta]]);
#     return np.dot(A, y0)
# ###############################################################################

###############################################################################
def dydt(y0, t): ### Hamiltonian of two atom 
    Omega_S =Omega_S0*np.exp(-1*(t-t1)**2/(2*w**2));
    Omega_P =Omega_P0*np.exp(-1*(t-t2)**2/(2*w**2));
    A=-1j*(hbar/2)*np.array([[0,Omega_P,0,Omega_P,0,0,0,0],
                [Omega_P,2*delta,Omega_S,0,Omega_P,0,0,0],
                [0,Omega_S,0,0,0,Omega_P,0,0],
                [Omega_P,0,0,2*delta,Omega_P,0,Omega_S,0],
                [0,Omega_P,0,Omega_P,4*delta,Omega_S,0,Omega_S],
                [0,0,Omega_P,0,Omega_S,2*delta,0,0],
                [0,0,0,Omega_S,0,0,0,Omega_P],
                [0,0,0,0,Omega_S,0,Omega_P,2*delta]]);
    return np.dot(A, y0)
###############################################################################
def eigenH(A): # Return eigen values
    w, v = LA.eig(A)
    return sorted(w) 
Omega_S_t =Omega_S0*np.exp(-1*(t-t1)**2/(2*w**2));
Omega_P_t =Omega_P0*np.exp(-1*(t-t2)**2/(2*w**2));

eigen = eigenH(H(t[0]))

for i in range(1,len(t)):
    z = eigenH(H(t[i]))
    eigen = np.vstack((eigen,z))


# # Assume that all of population accumulated in state |1>
# y0 = np.array([1,0,0])

# Assume that all of population accumulated in state |1>
y0 = np.array([1,0,0,0,0,0,0,0])

population= odeintz(dydt, y0, t, rtol=1e-12,atol=1e-10)
y1 = population[:,0]
y2 = population[:,1]
y3 = population[:,2]
y4 = population[:,3]
y5 = population[:,4]
y6 = population[:,5]
y7 = population[:,6]
y8 = population[:,7]

y_sum_0 = (y3+y7)/np.sqrt(2)

y_sum = np.real(np.conjugate(y8)*y8) # Population in state |re>
y1 = np.real(np.conjugate(y1)*y1) # Population in state |gg>
y2 = np.real(np.conjugate(y2)*y2) # Population in state |ge>
y3 = np.real(np.conjugate(y3)*y3) # Population in state |gr>
y4 = np.real(np.conjugate(y4)*y4) # Population in state |eg>
y5 = np.real(np.conjugate(y5)*y5) # Population in state |ee>
y6 = np.real(np.conjugate(y6)*y6) # Population in state |er>
y7 = np.real(np.conjugate(y7)*y7) # Population in state |rg>
y8 = np.real(np.conjugate(y8)*y8) # Population in state |re>
y_sum = np.real(np.conjugate(y_sum_0)*y_sum_0) # Population in state |re>



###############################################################################
## Plot pulsed Rabi frequencies
fig, (ax1,ax2,ax3) = plt.subplots(3)
ax1.plot(t,Omega_S_t,'--', label='S');
ax1.plot(t,Omega_P_t, label='P');
ax1.legend(loc=2, prop={'size': 15})
ax1.set(ylabel='Rabi frequency [arb. units]')
###############################################################################
## Eigen values
ax2.plot(t,np.real(eigen[:,0]),'b')
ax2.plot(t,np.real(eigen[:,1]),'r')
ax2.plot(t,np.real(eigen[:,2]),'g')
ax2.plot(t,np.real(eigen[:,3]),'b')
ax2.plot(t,np.real(eigen[:,4]),'r')
ax2.plot(t,np.real(eigen[:,5]),'g')
ax2.plot(t,np.real(eigen[:,6]),'b')
ax2.plot(t,np.real(eigen[:,7]),'r')
ax2.set(xlabel='Time [arb. units]', ylabel='Eigen values')



# ###############################################################################
# ## Plot population
ax3.plot(t,y1,'b', label='gg')
ax3.plot(t,y5,'r--', label='ee')
ax3.plot(t,y_sum,'g', label='gr+rg')
ax3.legend(loc=2, prop={'size': 15})
ax3.set(xlabel='Time [arb. units]', ylabel='Population')
