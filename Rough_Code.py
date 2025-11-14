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
if h + h_prime - E <= 1500 
    G = a*(h+h_prime-E)-b(h+h_prime-E)^2
else 
    G = .56;


