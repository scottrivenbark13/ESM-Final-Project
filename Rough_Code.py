# rough idea of the G equation
# correct variable values later
a = .81e-3;
b = .3e-6;
E0 = 1;
k = 1;
dQ = 1;
E = E0+(k*dQ);
h = []
h_prime = []
G = 0;
if h + h_prime - E <= 1500:
    G = a*(h+h_prime-E)-b(h+h_prime-E)^2
else:
    G = .56;


# Setup (Imports and Variables)
# NOTE: All units of distance are in METERS
# 1 degree latiture ~= 111,000 m
# Since all h values are in meters, it is better to use meters instead of degrees when solving\
# all relevant constant/intital values are stored in the values dictionary


import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.sparse import diags


def initialize(numberPoints, dt):

    values = {}

    values['n'] = numberPoints
    values['dt'] = dt # years

    values['alpha'] = 5
    values['beta'] = 2
    values['A'] = 5.77 * 10 ** -4
    values['lat_max'] = 74
    values['lat_min'] = 30

    values['a'] = 0.81 * 10 ** -3
    values['b'] = 0.30 * 18 ** -6
    values['r'] = 0.3
    values['k'] = 17
    values['nu'] = 100
    values['v'] = 100,000 * 1000 # km2 / yr, converted to meters 

    values['dx'] = (74-30) / numberPoints
    values['dx_m'] = values['dx'] * 111,000


    values['h'] = np.zeros(numberPoints)
    values["h_prime"] = np.zeros(numberPoints)
    values['lat'] = np.linspace(30, 74, numberPoints)
    values['E0'] = 550 + 111 * (values['lat']-70) # from paper, this math might be wrong, will probably have to change
    values["h_prime0"] = np.interp(values['lat'], np.array[30, 40, 66, 70, 74], np.array[400, 400, 200, 850, -500]) #generating topography

    return values

v = initialize(80, 50) # Test values from paper, 80 points and 50 year timesteps

def bedrockM (size): #Creating the matricies that will be used to solve equation 3
    arr_size = size + 1
    cD = v['v'] * v['dt'] / v['dx_m'] ** 2
    ML_Data = np.array([[-0.5*cD]*arr_size,[1+cD]*arr_size,[-0.5*cD]*arr_size])
    MLdiag = np.array([-1,0,1])
    ML = sp.sparse.spdiags(ML_Data, MLdiag, arr_size, arr_size).toarray()

    MR_Data = np.array([[0.5*cD]*arr_size,[1-cD]*arr_size,[0.5*cD]*arr_size])
    MRdiag = np.array([-1,0,1])
    MR = sp.sparse.spdiags(MR_Data, MRdiag, arr_size, arr_size).toarray()

    # Applying Dirichlet conditions to arrays
    ML[0, 0] = 1
    MR[0, 0] = 1
    ML[arr_size-1, arr_size-1] = 1
    MR[arr_size-1, arr_size-1] = 1

    return ML, MR

# Mass balance function, solving for G, likely not going to work first time, dQ is a mess to calculate so
# I wrote a placeholder function

def massBal(h, h_prime, timestep):
    dQ = 50 * np.sin(2*np.pi * timestep / 100000)
    elevation = h + h_prime
    G = np.zeros(elevation)
    E = E0 + v['k'] * dQ
    for i in range(elevation):
        if elevation[i] - E[i] <= 1500:
            #quad
            G[i] = v['a'] * (elevation[i] - E[i]) - v['b'] * (elevation[i] - E[i]) ** 2
        elif elevation[i] - E[i] > 1500:
            G[i] = 0.56
    
    return G

# Function for Ice density matricies, this is not a one and done call. 