'''
rough idea of the G equation
 correct variable values later
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

'''

# Setup (Imports and Variables)
# NOTE: All units of distance are in METERS
# 1 degree latiture ~= 111,000 m
# Since all h values are in meters, it is better to use meters instead of degrees when solving\
# all relevant constant/intital values are stored in the values dictionary
# code DOES NOT RUN RIGHT NOW, need to fix intial values as well as the main loop, ice density function also needs a rework


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
    values['b'] = 0.30 * 10 ** -6
    values['r'] = 0.3
    values['k'] = 17
    values['nu'] = 100
    values['v'] = 100000 * 1000 # km2 / yr, converted to meters 

    values['dx'] = (74-30) / (numberPoints - 1)
    values['dx_m'] = values['dx'] * 111000


    values['h'] = np.zeros(numberPoints)
    values["h_prime"] = np.zeros(numberPoints)
    values['lat'] = np.linspace(30, 74, numberPoints)
    values['E0'] = 550 + 111 * (values['lat']-70) # from paper, this math might be wrong, will probably have to change
    values["h_prime0"] = np.interp(values['lat'], np.array([30, 40, 66, 70, 74]), np.array([400, 400, 200, 850, -500])) #generating topography

    return values

v = initialize(80, 50) # Test values from paper, 80 points and 50 year timesteps

def bedrockM (size): #Creating the matricies that will be used to solve equation 3, DOES NOT NEED TO BE CALLED MORE THAN ONCE
    arr_size = size
    cD = v['v'] * v['dt'] / v['dx_m'] ** 2
    ML_Data = np.array([[-0.5*cD]*arr_size,[1+cD]*arr_size,[-0.5*cD]*arr_size])
    MLdiag = np.array([-1,0,1])
    ML = sp.sparse.spdiags(ML_Data, MLdiag, arr_size, arr_size).toarray()

    MR_Data = np.array([[0.5*cD]*arr_size,[1-cD]*arr_size,[0.5*cD]*arr_size])
    MRdiag = np.array([-1,0,1])
    MR = sp.sparse.spdiags(MR_Data, MRdiag, arr_size, arr_size).toarray()

    # Applying Dirichlet conditions to arrays
    # not sure why the entire row needs to be zero, had to look it up to get this function working, LOOK INTO THIS
    ML[0, :] = 0
    ML[0, 0] = 1
    MR[0, :] = 0
    MR[0, 0] = 1

    ML[arr_size-1, :] = 0
    ML[arr_size-1, arr_size-1] = 1
    MR[arr_size-1, :] = 0
    MR[arr_size-1, arr_size-1] = 1


    return ML, MR

# Mass balance function, solving for G, likely not going to work first time, dQ is a mess to calculate so
# I wrote a placeholder function for now, will have more time when I'm not dying from exams to fix this.

def massBal(h, h_prime, timestep):
    dQ = 50 * np.sin(2*np.pi * timestep / 100000)
    elevation = h + h_prime
    G = np.zeros(len(elevation))
    E = v['E0'] + v['k'] * dQ
    for i in range(len(elevation)):
        if elevation[i] - E[i] <= 1500:
            #quadratic function
            G[i] = v['a'] * (elevation[i] - E[i]) - v['b'] * (elevation[i] - E[i]) ** 2
        elif elevation[i] - E[i] > 1500:
            #constant
            G[i] = 0.56
    
    return G

'''
first iteration of ice density function
# Function for Ice density matricies, this is not a one and done call. 
def iceDensity(size):
    arr_size = size + 1
    cD = (v['A']*v['dt'])/(v['dx']**2)
    # Create the diagonal vectors
    a = -cD/2 * np.ones(arr_size)
    b = (1 + cD) * np.ones(arr_size)
    c = -cD/2 * np.ones(arr_size)
    b[0] = 1
    c[0] = 0
    a[-1] = 0
    b[-1] = 1
    left = diags([a, b, c], [-1, 0, 1], shape=(space_step, space_step)) # create the left hand matrix
    right = np.zeros(arr_size,1) #create the rhs matrix 
    return left,right
'''

def iceDensity(h, h_prime):
    noZeroMath = 1e-10 # prevents h and the gradient from being zero, cause that would cause a lot of problems 
    dx = v['dx_m']
    n = len(h)
    elevation = h + h_prime
    gradient = np.zeros(len(elevation))
    for i in range(1, n - 1):
        gradient[i] = (elevation[i + 1] - elevation[i-1]) / (2 * dx)
    
    # boundary conditions for gradient array
    gradient[0] = (elevation[1] - elevation[0]) / dx
    gradient[-1] = (elevation[-1] - elevation[-2]) / dx
    
    diffuse = v['A'] * (h + noZeroMath) ** v['alpha'] * (np.abs(gradient) + noZeroMath) ** v['beta']

    # Different way of creating the matricies than the bedrock due to the fact that we have to change them each timestep
    # we have to recalculate our "cD" each time
    lower = np.zeros(n - 1)
    mid = np.ones(n)
    upper = np.zeros(n - 1)
    
    for i in range(1, n-1):
        D_left = 0.5 * (diffuse[i-1] + diffuse[i])
        D_right = 0.5 * (diffuse[i] + diffuse[i+1])
        
        r_left = 0.5 * v['dt'] * D_left / dx ** 2
        r_right = 0.5 * v['dt'] * D_right / dx ** 2
        
        lower[i-1] = -r_left
        mid[i] = 1 + r_left + r_right
        upper[i] = -r_right

    # creating ML
    mid[0] = 1
    upper[0] = 0
    lower_full = np.zeros(n)
    lower_full[1:] = lower

    mid_full = mid

    upper_full = np.zeros(n)
    upper_full[:-1] = upper

    ML_Data = np.vstack([lower_full, mid_full, upper_full])
    MLdiag = np.array([-1,0,1])
    ML = sp.sparse.diags(ML_Data, MLdiag, shape=(n,n)).toarray()

    # had to look this up, idk how it works to be honest need to figure that out
    lower_MR = -lower
    mid_MR = 2 - mid
    upper_MR = -upper
    

    lower_MR_full = np.zeros(n)
    lower_MR_full[1:] = lower_MR

    mid_MR_full = mid_MR

    upper_MR_full = np.zeros(n)
    upper_MR_full[:-1] = upper_MR
    MR_data = [lower_MR, mid_MR, upper_MR]
    MRdiag = np.array([-1,0,1])
    MR = sp.sparse.diags(MR_data, MRdiag, shape=(n,n)).toarray()
    
    MR[0, :] = 0
    MR[0, 0] = 1
    
    return ML, MR



def solveIceEquation(h, h_prime, time):
    ML_Ice, MR_Ice = iceDensity(h, h_prime) # fix this function
    G = massBal(h, h_prime, time) # change timestep if needed idk if this is right
    RHS = MR_Ice @ h + v['dt'] * G
    # need to apply boundary conditions 
    RHS[0] = 0
    h_updated = np.linalg.inv(ML_Ice) @ RHS  # this math might be wrong, check later
    return h_updated


def solveBedrock(h, h_prime):
    ML_Bedrock, MR_Bedrock = bedrockM(v['n'])
    temp_var = h_prime - v['h_prime0'] + v['r'] * h # makes matrix math easier than it would be otherwise
    temp_var[0] = 0
    temp_var[-1] = 0
    temp_new = np.linalg.inv(ML_Bedrock) @ MR_Bedrock @ temp_var
    # do we need to apply boundary conditions here? maybe later
    final_hPrime = temp_new + v['h_prime0'] - v['r'] * h
    final_hPrime[0] = v['h_prime0'][0]
    final_hPrime[-1] = v['h_prime0'][-1]
    return final_hPrime


# Main loop, likely needs work
"""
for i in range(700000):
    # WORK ON MAIN LOOP HERE

    temp = 1
 # order of functions for solving
final_hprime = solveBedrock(h,h_prime)
G = massBal(h,h_prime,timestep)
h_updated = solveIceEquation(h,final_hprime)
"""

# setting up the results arrays
n_years = 700000 /v['dt']


save_interval = 50

numSave = int(n_years // save_interval) + 1

results_h = np.zeros((numSave, v['n']))
results_h_prime = np.zeros((numSave, v['n']))
results_time = np.zeros(numSave)
results_volume = np.zeros(numSave)

current_time = 0.0
save_index = 0

# main loop, saving all the info from running the functions
for step in range(numSave):
    time = step * v['dt']
    
    h_old = v['h'].copy()
    h_prime_old = v['h_prime'].copy()
    
    h_new = solveIceEquation(h_old, h_prime_old, current_time)
    
    h_avg = 0.5 * (h_old + h_new)
    
    h_prime_new = solveBedrock(h_avg, h_prime_old)
    
    v['h'] = h_new
    v['h_prime'] = h_prime_new
    
    if step % save_interval == 0:
        results_h[save_index, :] = v['h']
        results_h_prime[save_index, :] = v['h_prime']
        results_time[save_index] = time
        
        results_volume[save_index] = np.trapz(v['h'], dx=v['dx_m']) * 3000 / 1e9  # from paper, gets total volume

        save_index += 1



# Seems like the functions work! maybe the numbers are wrong and thats why the plot isn't correct at all

plt.figure(figsize=(12, 5))
plt.plot(results_time, results_volume, 'b-', linewidth=2)
plt.show()