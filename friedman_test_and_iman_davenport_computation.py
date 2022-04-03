# Calculating the Friedman Stat

# Get the average rank of each ensemble for regression problems
kNNE = 7.70
DTE = 6.90
RF = 2.80
SVRE = 10.60
NNE = 7.40
kNNhte = 6.30
DThte = 3.80
SVRhte = 9.80
NNhte = 6.30
HTEsm = 2.80
HTEdf = 1.60

#Number of Datasets
N = 10

# Number of ensembles
K = 11

X_F = ((12*N)/(K*(K+1)))*(((((kNNE**2)+(DTE**2)+(RF**2)+(SVRE**2)+(NNE**2)+(kNNhte**2)+(DThte**2)+
                         (SVRhte**2)+(NNhte**2)+(HTEsm**2)+(HTEdf**2))*4) - (K*((K+1)**2)))/4)

print('Friedman Test Statistic, X$^2$F =', X_F)
print("\n")

F_F = ((N-1)*X_F)/((N*(K-1))-X_F)

print('Iman Daveport test statistic, F_F =', F_F)








"""
# Get the average rank of each ensemble for classification problems
NBE = 12.35
kNNE = 8.90
DTE = 10.10
RF = 7.05
SVME = 6.15
NNE = 6.60
NBhte = 11.00
kNNhte = 8.50
DThte = 7.25
SVMhte = 3.90
NNhte = 4.20
HTEsm = 3.40
HTEdf = 1.60

#Number of Datasets
N = 10

# Number of ensembles
K = 13

X_F = ((12*N)/(K*(K+1)))*(((((NBE**2)+(kNNE**2)+(DTE**2)+(RF**2)+(SVME**2)+(NNE**2)+(NBhte**2)+(kNNhte**2)+(DThte**2)+
                         (SVMhte**2)+(NNhte**2)+(HTEsm**2)+(HTEdf**2))*4) - (K*((K+1)**2)))/4)

print('Friedman Test Statistic, X$^2$F =', X_F)
print("\n")

F_F = ((N-1)*X_F)/((N*(K-1))-X_F)

print('Iman Daveport test statistic, F_F =', F_F)

"""

