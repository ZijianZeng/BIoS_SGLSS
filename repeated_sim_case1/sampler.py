# used for function
import numpy as np
import scipy.linalg
import scipy.stats
import scipy.special
import torch
import os
from tqdm import tqdm
import statsmodels.api as sm
import pandas as pd

def myZrnd( sigmae, invSigma, Y, X, beta, Ip):
    invV = torch.tensor( 1/sigmae * Ip  + invSigma )
    m_zi =  torch.tensor( 1/sigmae * Y) + torch.matmul( torch.tensor(invSigma), 
                                                       torch.matmul( torch.tensor(X), torch.tensor(beta.T)).T ).T
    mu_zi = torch.linalg.solve( invV, m_zi.T ).T
    m , p = Y.shape
    stn = torch.tensor( np.random.randn( p,m ) )
    
    z = mu_zi + torch.linalg.solve( torch.linalg.cholesky(invV).T, stn).T
    return z.numpy()

def myIWrnd(df, G, p, pad, Ip):
    stn = torch.tensor( np.random.randn(p,df))
    sqK = torch.linalg.solve( torch.linalg.cholesky(torch.tensor( G )).T, stn)
    K = torch.matmul( sqK, sqK.T) # this is Sigma^{-1} 
    iK = torch.linalg.inv( K + torch.tensor( pad*Ip ) ) # this is Sigma
    
    return K.numpy(), iK.numpy()
    
def myZXB( z, X, beta):
    tem_ZXB = torch.tensor( tem_z) - torch.matmul( torch.tensor(X), torch.tensor( beta).T )
    sum_ZXB = torch.matmul( tem_ZXB.T, tem_ZXB)
    return sum_ZXB.numpy()

torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Get the name of the current GPU
# print(torch.cuda.get_device_name(torch.cuda.current_device()))
# Is PyTorch using a GPU?
# print(torch.cuda.is_available())

np.random.seed(123)
path = os.getcwd()

for sim in range(50):
    newpath = path + '/samples/cases_' + str(sim)
    if not os.path.exists(newpath):
        os.makedirs(newpath)
        os.makedirs(newpath + '/samples/beta_store')
        os.makedirs(newpath + '/samples/gamma_store')
        os.makedirs(newpath + '/samples/pi_store')
        os.makedirs(newpath + '/samples/Sigma_store')
        os.makedirs(newpath + '/samples/sigmae_store')
        os.makedirs(newpath + '/samples/tau_store')
        os.makedirs(newpath + '/samples/theta_store')
        os.makedirs(newpath + '/samples/z_store')
        

    exec(open('data_load.py').read())
    exec(open('prior_settings.py').read())

    # sampler

    for iter in tqdm(range(maxiter)):

        # step 1: sample z
        tem_z[:,:] = myZrnd( tem_sigmae, tem_invSigma, Y, X, tem_beta, Ip)

        # print( iter, 'iter: Step 1 done')

        # step 2: Sample sigmae
        post_ea = prior_ea + m*p2/2
        post_eb = prior_eb + 1/2 * np.sum( (Y - tem_z) ** 2)
        tem_sigmae = scipy.stats.invgamma.rvs( a = post_ea, scale = post_eb, size = 1)

        # print( iter, 'iter: Step 2 done')

        # step 3: Blocked Sample
        # common factor
        xx = (X ** 2).sum(axis = 0)

        for j in range(q):
            sumxz = (X[:,j] * tem_z.T).T.sum(axis = 0)
            sumxxb =   (X[:,j][:,np.newaxis]*((np.delete( X, j, axis = 1)[:, np.newaxis] * np.delete( tem_beta, j, axis = 1)).sum(axis = 2) )).sum(axis = 0)
            xz = sumxz - sumxxb

            # theta and tau
            tildev = 1/(xx[j]/np.diag(tem_Sigma) + 1/sigma0[:,j])
            tildem = xz/np.diag(tem_Sigma) + mu0[:,j]/sigma0[:,j]

            log_theta = ( np.log(1-tem_pi[j]) - ( np.log(tem_pi[j]) - 1/2*np.log(sigma0[:,j]) - 
                                               1/2 * (mu0[:,j]**2)/sigma0[:,j] + 1/2 *  np.log(tildev) +
                                               1/2 * (tildem ** 2) * tildev))
            tem_theta[:,j] = 1/(1+np.exp( log_theta))
            tem_tau[:,j] = np.random.binomial(n = 1, p = tem_theta[:,j], size = p2) 

            # pi and gamma
            post_pia = prior_pia + sum(tem_tau[:,j])
            post_pib = prior_pib + p2 - sum(tem_tau[:,j])
            tem_pi[j] = np.random.beta( a = post_pia, b = post_pib, size = 1 )

            # hard-thresholding
            tem_gamma[j] = 1 * (tem_pi[j] >= d)
    
            if j == 0:
                tem_tau[:,j] = 1
                tem_pi[j] = 1
                tem_gamma[j] = 1

            # beta
            if( tem_gamma[j] > 0 ):
                tem_beta[:,j] = np.random.normal( loc = tildev*tildem, scale = np.sqrt(tildev), size = p2)
                tem_beta[:,j] = tem_beta[:,j] * tem_tau[:,j]
            else: 
                tem_beta[:,j] = 0

        # prevent all gamma == 0
        if (np.sum( tem_gamma ) == 0):
            j = np.random.choice(q)
            # set all ones for the covariate
            tem_gamma[j] = 1
            tem_theta[:,j] = 1
            tem_tau[:,j] = 1

            # sample beta
            sumxz = (X[:,j] * tem_z.T).T.sum(axis = 0)
            sumxxb =   (X[:,j][:,np.newaxis]*((np.delete( X, j, axis = 1)[:, np.newaxis] * np.delete( tem_beta, j, axis = 1)).sum(axis = 2) )).sum(axis = 0)
            xz = sumxz - sumxxb

            tildev = 1/(xx[j]/np.diag(tem_Sigma) + 1/sigma0[:,j])
            tildem = xz/np.diag(tem_Sigma) + mu0[:,j]/sigma0[:,j]

            tem_beta[:,j] = np.random.normal( loc = tildev*tildem, scale = np.sqrt(tildev), size = p2)
        # print( iter, 'iter: Step 3 done')

        # step 4: Sample Sigma
        nu = (m + delta) + p2 - 1
        sum_ZXB = myZXB( tem_z, X, tem_beta)
        Psi = sum_ZXB + Prior_Psi + pad * Ip

        tem_invSigma[:,:], tem_Sigma[:,:] = myIWrnd(nu, Psi, p2, pad, Ip)

        # print( iter, 'iter: Step 4 done')

        # data saving
        np.save( newpath + '/samples/beta_store/tem_beta_'+str(iter)+'.npy',tem_beta)
        np.save( newpath + '/samples/theta_store/tem_theta_'+str(iter)+'.npy',tem_theta)
        np.save( newpath + '/samples/tau_store/tem_tau_'+str(iter)+'.npy',tem_tau)
        np.save( newpath + '/samples/pi_store/tem_pi_'+str(iter)+'.npy',tem_pi)
        np.save( newpath + '/samples/gamma_store/tem_gamma_'+str(iter)+'.npy',tem_gamma)
        np.save( newpath + '/samples/z_store/tem_z_'+str(iter)+'.npy',tem_z)
        np.save( newpath + '/samples/Sigma_store/tem_Sigma_'+str(iter)+'.npy',tem_Sigma)
        np.save( newpath + '/samples/sigmae_store/tem_sigmae_'+str(iter)+'.npy',tem_sigmae)

        # print( iter, 'iter: Data saved')
    print('the end of sim_' + str(sim))
