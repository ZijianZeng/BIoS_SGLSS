# generate data
np.random.seed(123)

m = 100; p = 30;
p2 = p**2
s = np.linspace(1,p,p)/(p+1)
S = np.array(np.meshgrid(s, s)).reshape(2, p2).T
dist = scipy.spatial.distance.cdist(S, S)

# 12 continuous 3 binary and 1 intercept
q = 16
beta = np.empty( (p2,q) )

def myMatern( d, sigma, theta, nu):
    K = scipy.special.kv( nu, np.sqrt(2*nu)*d/theta)
    np.fill_diagonal(K,1)
    Matern = (sigma / ( (2 ** (nu-1)) * scipy.special.gamma(nu)) *
         ((np.sqrt(2*nu) * d / theta) ** nu) * K)
    np.fill_diagonal(Matern,sigma)
    return Matern

s2e = 1; s2b = 1

Sigma_beta = np.empty( (p2,p2) )
Sigma_eta = np.empty( (p2,p2) )

Sigma_beta[:,:] = myMatern( dist, s2b, 1/4, 5/2)
Sigma_eta[:,:] = myMatern( dist, s2e, 1/4, 5/2)

LSigma_beta = np.linalg.cholesky( Sigma_beta )

for j in range(q):
    beta[:,j] = LSigma_beta @ np.random.randn(p2)

    if np.max(beta[:,j]) > np.abs( np.min(beta[:,j]) ):
        beta[:,j] = (beta[:,j] + np.max(beta[:,j]))/(2*np.abs(np.max(beta[:,j])))
    else:
        beta[:,j] = (beta[:,j] + np.min(beta[:,j]))/(2*np.abs(np.min(beta[:,j])))

tau = np.zeros( (p2,q) )
tau[:,[0,1]] = 1
tau[np.random.choice(range(p2), int(p2*0.9), replace = False), 2] = 1
tau[np.random.choice(range(p2), int(p2*0.8), replace = False), 3] = 1
tau[np.random.choice(range(p2), int(p2*0.7), replace = False), 4] = 1
tau[np.random.choice(range(p2), int(p2*0.6), replace = False), 5] = 1

tau[:,6] = 1
tau[np.random.choice(range(p2), int(p2*0.9), replace = False), 7] = 1
tau[np.random.choice(range(p2), int(p2*0.8), replace = False), 8] = 1
beta = beta * tau

# generate data
Y = np.empty( (m,p2) )
X = np.empty( (m,q) )
eta = np.empty( (m,p2) )
u = np.empty( (m,p2) )
Z = np.empty( (m,p2) )
mean = np.empty( (m,p2) )
# continuous covariates
X[:,1:6] = np.random.randn(m,5)
X[:,6:9] = np.random.binomial(1, 0.5, (m,3))
X[:,9:16] = np.random.randn(m,7)

# interception
X[:,0] = 1

sigma2 = 1
sigma = np.sqrt(sigma2)
LSigma_eta = np.linalg.cholesky( Sigma_eta )
for i in range(m):
    eta[i,:] = LSigma_eta @ np.random.randn(p2)
    u[i,:] = np.random.randn(p2) * np.sqrt(sigma)
    mean[i,:] = 0
    for j in range(q):
        mean[i,:] = mean[i,:] + beta[:,j] * X[i,j]
    
    Z[i,:] = mean[i,:] + eta[i,]
    Y[i,:] = Z[i,:] + u[i,:]
    
MUA_beta = np.empty( (p2,q))
MUA_p2 = np.empty( (p2,q) )
for i in range(p2):
    mod = sm.OLS(Y[:,i], X)
    res = mod.fit()
    MUA_beta[i,:] = res.params
    MUA_p2[i,:] = res.pvalues

pd.DataFrame(mean).to_csv('mean.csv')
pd.DataFrame(Z).to_csv('Z.csv')
pd.DataFrame(MUA_p2).to_csv('MUA_pvalue.csv')
pd.DataFrame(X).to_csv('X.csv')
pd.DataFrame(Y).to_csv('Y.csv')
pd.DataFrame(S).to_csv('S.csv')
pd.DataFrame(tau).to_csv('tau.csv')
pd.DataFrame(beta).to_csv('beta.csv')
pd.DataFrame(Sigma_eta).to_csv('Sigma_eta.csv')
pd.DataFrame(Sigma_beta).to_csv('Sigma_beta.csv')

