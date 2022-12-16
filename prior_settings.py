# optimize kernel
MUA_y = np.empty( (m,p2) )
for i in range(m):
    mean_y = 0
    for j in range(q):
        mean_y = mean_y + X[i,j] * MUA_beta[:,j]
    
    MUA_y[i,:] = mean_y
    
MUA_S = (Y - MUA_y).T @ ( np.eye(m) - np.outer( np.repeat(1,m), np.repeat(1,m))/m ) @ (Y - MUA_y)/m

def est_Psi(theta, Ecov, d):
    rho = np.exp(theta[0])
    sigmas = np.exp(theta[1])
    A = sigmas * ( 1 + np.sqrt(5)*d/rho + 5 * (d**2)/(3* (rho**2)) ) * np.exp(- np.sqrt(5)*d/rho)
    return np.mean( (A-Ecov) ** 2)

est_theta = scipy.optimize.minimize( est_Psi, x0 = [1,1], args = (MUA_S, dist))

rho = np.exp( est_theta.x[0] )
sigmas = np.exp( est_theta.x[1] )
Prior_Psi = sigmas * ( 1 + np.sqrt(5)*dist/rho + 5 * (dist**2)/(3* (rho**2)) ) * np.exp(- np.sqrt(5)*dist/rho)

# prior settings
# MCMC settings
maxiter = 2000

prior_pia = 1
prior_pib = 1
prior_ea = 1
prior_eb = 1
delta = 5

d = 0.2

mu0 = np.empty( shape = (p2,q) )
sigma0 = np.empty( shape = (p2,q) )

mu0[:,:] = 0
sigma0[:,:] = 1

# initial value
tem_z = np.empty( shape = (m,p2))
tem_beta = np.empty( shape = (p2, q))
tem_theta = np.empty( shape = (p2,q))
tem_tau = np.empty( shape = (p2,q))
tem_gamma = np.empty( shape = (q))
tem_pi = np.empty( shape = (q))
tem_Sigma = np.empty( shape = (p2,p2))
tem_sigmae = np.empty( shape = (1))
tem_invSigma = np.empty( shape = (p2,p2))

tem_beta[:,:] = MUA_beta
tem_theta[:,:] = 1
tem_tau[:,:] = 1
tem_gamma[:] = 1
tem_pi[:] = 0.5
tem_Sigma[:,:] = Prior_Psi
tem_sigmae[:] = 1

Ip = np.eye(p2)
pad = 0

tem_invSigma[:,:] = torch.linalg.inv( torch.tensor( tem_Sigma + pad*Ip)).numpy()