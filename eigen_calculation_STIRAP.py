import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA


###############################################################################
## Ref: Beterov, I. I., et al. "Application of adiabatic passage in Rydberg atomic ensembles for quantum information processing." Journal of Physics B: Atomic, Molecular and Optical Physics 53.18 (2020): 182001.
###############################################################################

## Define pulsed Rabi frequencies
hbar=1.0545718e-34*0+1;

tf =10;
t=np.linspace(-6, 6, 500)*1e-6;

Omega_S0=2*np.pi*10e6;
Omega_P0=2*np.pi*10e6;
t1=-1e-6;
t2=1e-6;
w=1e-6;
delta=0*2*np.pi*1e6;

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


###############################################################################
## Plot pulsed Rabi frequencies
fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(t,Omega_S_t,'--', label='S');
ax1.plot(t,Omega_P_t, label='P');
ax1.legend(loc=2, prop={'size': 15})
ax1.set(ylabel='Rabi frequency [arb. units]')
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
plt.savefig('my_fig.png', dpi=400)
