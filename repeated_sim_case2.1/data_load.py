m = 100; p = 30;
p2 = p**2
s = np.linspace(1,p,p)/(p+1)
S = np.array(np.meshgrid(s, s)).reshape(2, p2).T
dist = scipy.spatial.distance.cdist(S, S)
q = 16

mean = pd.read_csv(  path + '/samples/cases_' + str(sim) + '/mean.csv' )
Z = pd.read_csv( path + '/samples/cases_' + str(sim) + '/Z.csv' )
MUA_p2 = pd.read_csv( path + '/samples/cases_' + str(sim) +  '/MUA_pvalue.csv')
X = pd.read_csv( path + '/samples/cases_' + str(sim) +  '/X.csv')
Y = pd.read_csv( path + '/samples/cases_' + str(sim) +  '/Y.csv')
S = pd.read_csv( path + '/samples/cases_' + str(sim) +  '/S.csv')
tau = pd.read_csv( path + '/samples/cases_' + str(sim) +  '/tau.csv')
beta = pd.read_csv( path + '/samples/cases_' + str(sim) +  '/beta.csv')
Sigma_eta = pd.read_csv( path + '/samples/cases_' + str(sim) +  '/Sigma_eta.csv')
Sigma_beta = pd.read_csv( path + '/samples/cases_' + str(sim) +  '/Sigma_beta.csv')


mean = mean.values[:,1:(p2+1)]
Z = Z.values[:,1:(p2+1)]
MUA_p2 = MUA_p2.values[:,1:(q+1)]
X = X.values[:,1:(q+1)]
Y = Y.values[:,1:(p2+1)]
S = S.values[:,1:3]
tau = tau.values[:,1:(q+1)]
beta = beta.values[:,1:(q+1)]
Sigma_eta = Sigma_eta.values[:,1:(p2+1)]
Sigma_beta = Sigma_beta.values[:,1:(p2+1)]


MUA_beta = np.empty( (p2,q))
MUA_p2 = np.empty( (p2,q) )
for i in range(p2):
    mod = sm.OLS(Y[:,i], X)
    res = mod.fit()
    MUA_beta[i,:] = res.params
    MUA_p2[i,:] = res.pvalues

pd.DataFrame(mean).to_csv( newpath + '/mean.csv' )
pd.DataFrame(Z).to_csv( newpath + '/Z.csv' )
pd.DataFrame(MUA_p2).to_csv( newpath + '/MUA_pvalue.csv')
pd.DataFrame(X).to_csv( newpath + '/X.csv')
pd.DataFrame(Y).to_csv( newpath + '/Y.csv')
pd.DataFrame(S).to_csv( newpath + '/S.csv')
pd.DataFrame(tau).to_csv( newpath + '/tau.csv')
pd.DataFrame(beta).to_csv( newpath + '/beta.csv')
pd.DataFrame(Sigma_eta).to_csv( newpath + '/Sigma_eta.csv')
pd.DataFrame(Sigma_beta).to_csv( newpath + '/Sigma_beta.csv')

np.save( 'current_' + str(sim), 1)

if sim != 0:
    os.remove('current_' + str( sim-1 ) + '.npy' )
    
