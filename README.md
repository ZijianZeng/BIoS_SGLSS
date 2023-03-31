## Bayesian Image-on-Scalar Regression with a Spatial Global-Local Spike-and-Slab Prior
### Reference  
Zeng, Z., Li, M. and Vannucci, M. (2022+). Bayesian Image-on-Scalar Regression with a Spatial Global-Local Spike-and-Slab Prior. Bayesian Analysis, accepted. [arXiv: 2209.08234](https://arxiv.org/abs/2209.08234)

### Use
Run file `sampler.py'

## Code annotation 

#### python codes (`.py` files):

- The main codes are in the `.py` files

  - `sampler.py`  
    The main function, when running, it  
    - loads module (with version for the main modules):  
      numpy: 1.21.2  
      scipy: 1.6.1  
      torch: 1.9.0 + cu102  
      statsmodels: 0.11.0  
      pandas: 1.3.3  

    - runs `data_gen.py` to generate data   
    - runs `prior_settings.py` to load prior settings  
    - defines a few functions for Gibbs sampler  
    - does the sampling and saves results under path: `./samplers/*`  
      **The folders needs to be manually created.**  
  - `data_gen.py`  
    The data generating function, which will be called when running `sampler.py`  
    When running, it:  
    - sets random seeds with the setting,  
      `m` represents the number of images observed  
      `p` represents image-size, $p-by-p$ image  
      `p2` represents the number of pixels  
      `q` represents the number of covariates  
      `S` represents the grid on image  
    - generates parameters from Gaussian Process  
      `s2e` represents the $\sigma^2$ used for the covariance surface  
      `s2b` represents the $\sigma^2$ used for coefficient images  
    - generates data from the settings  
    - runs MUA on each pixels and saves both generated data and statistics maps.  
  - `prior_settings.py`  
    The prior setting function, which will be called when running `sampler.py`  
    When running, it:  
    - obtained a optimized kernel for the prior of Inverse-Wishart process  
    - set priors and initial values for MCMC algorithms  
    - add an pad in case of the real data provide invertible variance for IW kernel / variance  


  - Note: some warnings may be there due to we manually set the intercept is always selected for all local points, `sampler.py` line 90-93, where $\pi_0 = 1$ is enforced for intercept, leading to `log_theta` on line 76 calculated $log(1-\pi_0)$ and report warning. Meanwhile, this won't cause trouble since we will set $\tau_0(s) = 1$ later. 

#### Example results (`result_plot.ipynb`):

- The jupyter notebook we used to check results for single simulation. The provided one is for the 'good separation plot' we show in the appendix. When doing the main revision, we re-run everything in the 50 repeated simulations, and no longer use this notebook.
  We attached it considering the recorded plots may offer some intuition about how we report the results

#### Reproducible simulation settings (`repeated_sim_case*`)

- Codes we used to generate dataset and run the models. `data_gen.py` is used to generate the 50 data sets;  
  `sampler.py` will do what we described above repeatedly for all the set generated.
- It takes around 6 mins on our cluster for one single data set and around 800GB to store all results for the 50 data sets, mainly because of the MCMC chains for IWP sample, which is 2000-by-900-by-900 dimension. If you'd like to try it, please use a cluster or something.

