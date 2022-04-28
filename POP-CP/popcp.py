import numpy as np
import pandas as pd
from scipy import interpolate
import scipy.stats
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV

def choice_prob_1D(x1, x2):
    """
    Calculates CP for a given 1-D sample.

    Parameters:
      x1 (ndarray): negative trials
      x2 (ndarray): positive trials

    Returns:
      CP given by Mann-Whitney U statistic (scalar)
    """
    
    assert x1.ndim == 1, 'x1 should be an array'
    assert x2.ndim == 1, 'x2 should be an array'

    N1 = len(x1)
    N2 = len(x2)
    U, pval = scipy.stats.mannwhitneyu(x2, x1, alternative='two-sided')
    CP = U / N1 / N2
    
    return CP

def gaussian_1D_true_CP(mu1, sigma1, mu2, sigma2):
    """
    Calculates true CP for two 1-D Gaussian distributions.
    
    Parameters:
        mu1 (scalar): mean of distribution 1
        sigma1 (scalar): standard deviation of distribution 1
        mu2 (scalar): mean of distribution 2
        sigma2 (scalar):  standard deviation of distribution 2
        
    Returns:
        True CP of distributions (scalar)
    """
    
    t = abs(mu1 - mu2)/pow(pow(sigma1, 2) + pow(sigma2, 2), 0.5)
    CP = scipy.stats.norm.cdf(t)
    
    return CP

def best_linear_map(mu1, mu2, cov1, cov2):
    """
    Calculates best linear map for two N-D Gaussian distributions (N > 1).
    
    Parameters:
        mu1 (ndarray): mean of distribution 1
        mu2 (ndarray): mean of distribution 2
        cov1 (ndarray): covariance of distribution 2
        cov2 (ndarray): covariance of distribution 2
        
    Returns:
        N-D vector representing a linear map (ndarray)
    """
    
    if (mu1 == mu2).all():
        return np.zeros(len(mu1))
    
    inv_cov = np.linalg.inv(cov1 + cov2)
    vec = np.transpose(np.matmul(inv_cov, mu1 - mu2))
    
    return vec

def projected_gaussian(mu, cov, vec):
    """
    Projects N-D Gaussian distribution onto a normalized vector.
    
    Parameters:
        mu (ndarray): mean of distribution
        cov (ndarray): covariance of distribution
        vec (ndarray): vector to be projected onto
        
    Returns:
        mean of projected distribution (scalar), variance of projected distribution (scalar)
    """
    
    unit_vector = vec/np.linalg.norm(vec)
    mu_projected = mu @ unit_vector
    var_projected = np.matmul(np.matmul(np.transpose(unit_vector), cov), unit_vector)
    
    return mu_projected, var_projected

def gaussian_ND_true_CP(mu1, mu2, cov1, cov2):
    """
    Calculates true CP for two N-D Gaussian distributions (N > 1).
    
    Parameters:
        mu1 (ndarray): mean of distribution 1
        mu2 (ndarray): mean of distribution 2
        cov (ndarray): covariance of distributions
        
    Returns:
        True CP of distributions (scalar)
    """
    
    if (mu1 == mu2).all():
        return 0.5
    
    boundary_normal = best_linear_map(mu1, mu2, cov1, cov2)
    z1_mean, z1_var = projected_gaussian(mu1, cov1, boundary_normal)
    z1_sigma = z1_var ** 0.5
    z2_mean, z2_var = projected_gaussian(mu2, cov2, boundary_normal)
    z2_sigma = z2_var ** 0.5
    
    return gaussian_1D_true_CP(z1_mean, z1_sigma, z2_mean, z2_sigma)

def naive_estimator(x1, x2):
    """
    Estimates CP for a sample based on the naive method.
    
    Parameters:
      x1 (ndarray): negative trials
      x2 (ndarray): positive trials

    Returns:
      Estimated CP using naive method (scalar)
    """
    
    N1 = len(x1)
    N2 = len(x2)
    
    X = np.concatenate((x1, x2))
    y = np.concatenate((np.zeros(N1), np.ones(N2))).astype('bool')

    model = LDA().fit(X, y)
    z1 = model.predict_log_proba(X[:N1, :])[:, 1]
    z2 = model.predict_log_proba(X[N1:, :])[:, 1]
    CP = choice_prob_1D(z1, z2)
    
    return CP

def halfnhalf_estimator(x1, x2):
    """
    Estimates CP for a sample based on the halfnhalf method.
    
    Parameters:
      x1 (ndarray): negative trials
      x2 (ndarray): positive trials

    Returns:
      Estimated CP using halfnhalf method (scalar)
    """
    
    N1 = len(x1)
    N2 = len(x2)
    np.random.shuffle(x1)
    np.random.shuffle(x2)
    
    # train/test sets are stratified
    X_train = np.concatenate((x1[:N1//2], x2[:N2//2]))
    X_test = np.concatenate((x1[N1//2:], x2[N2//2:]))

    y = np.concatenate((np.zeros(N1//2), np.ones(N2//2))).astype('bool')

    model = LDA().fit(X_train, y)
    
    z1 = model.predict_log_proba(X_test[:(N1-N1//2), :])[:, 1]
    z2 = model.predict_log_proba(X_test[(N1-N1//2):, :])[:, 1]
    CP = choice_prob_1D(z1, z2)
    
    return CP

def unregularized_pooled_estimator(x1, x2, k=5):
    """
    Estimates CP for a sample based on the unregularized pooled method.
    
    Parameters:
      x1 (ndarray): negative trials
      x2 (ndarray): positive trials
      
      k (int): number of XV folds

    Returns:
      Estimated CP using unregularized pooled method (scalar)
    """
    
    kfold = StratifiedKFold(k, shuffle=True)
    
    N1 = len(x1)
    N2 = len(x2)
    X = np.concatenate((x1, x2))
    y = np.concatenate((np.zeros(N1), np.ones(N2))).astype('bool')

    Z1 = np.empty((0))
    Z2 = np.empty((0))

    for train, test in kfold.split(X, y):

        test_z1 = (y[test] == True).nonzero()
        test_z2 = (y[test] == False).nonzero()

        model = LDA().fit(X[train, :], y[train])

        z1 = model.predict_log_proba(np.squeeze(X[test][test_z1, :]))[:, 1] #probability that point belongs to "choice 2 group"
        z2 = model.predict_log_proba(np.squeeze(X[test][test_z2, :]))[:, 1]
        Z1 = np.concatenate((Z1, z1))
        Z2 = np.concatenate((Z2, z2))
    pooled_CP = choice_prob_1D(Z2, Z1)
    CP = pooled_CP
    
    return CP

def regularized_pooled_estimator(x1, x2, k=5, Cs=np.logspace(-15, 4, 20)):
    """
    Estimates CP for a sample based on the regularized pooled method.
    
    Parameters:
      x1 (ndarray): negative trials
      x2 (ndarray): positive trials
      
      k (int): number of XV folds
      Cs (list or ndarray): possibilites for inverse of regularization strength

    Returns:
      Estimated CP using regularized pooled method (scalar)
    """
    
    kfold = StratifiedKFold(k, shuffle=True)
    
    N1 = len(x1)
    N2 = len(x2)
    X = np.concatenate((x1, x2))
    y = np.concatenate((np.zeros(N1), np.ones(N2))).astype('bool')

    Z1 = np.empty((0))
    Z2 = np.empty((0))

    for train, test in kfold.split(X, y):

        test_z1 = (y[test] == True).nonzero()
        test_z2 = (y[test] == False).nonzero()
        model = LogisticRegressionCV(Cs = Cs, cv=2, max_iter=10000).fit(X[train, :], y[train])

        z1 = model.predict_proba(np.squeeze(X[test][test_z1, :]))[:, 1]
        z2 = model.predict_proba(np.squeeze(X[test][test_z2, :]))[:, 1]
        Z1 = np.concatenate((Z1, z1))
        Z2 = np.concatenate((Z2, z2))
    CP = choice_prob_1D(Z2, Z1)
    
    return CP

def estimate_CP(x1, x2, method, k=5, Cs=np.logspace(-15, 4, 20)):
    """
    Estimates CP for a sample based on the chosen method.
    
    Parameters:
        x1 (ndarray): negative trials
        x2 (ndarray): positive trials
        method (str): method name from [naive, halfnhalf, unregularized_pooled, regularized_pooled]
        
        k (int): number of XV folds
        Cs (list or ndarray): possibilites for inverse of regularization strength
        
    Returns:
        Estimated CP (scalar)

    """
    
    if method=='naive':
        return naive_estimator(x1, x2)
    elif method=='halfnhalf':
        return halfnhalf_estimator(x1, x2)
    elif method=='unregularized_pooled':
        return unregularized_pooled_estimator(x1, x2, k)
    elif method=='regularized_pooled':
        return regularized_pooled_estimator(x1, x2, k, Cs)
    else:
        raise ValueError('invalid estimation method')

def monte_carlo_CP(mu1, mu2, cov1, cov2, method, N1, N2, M, k=5, Cs=np.logspace(-15, 4, 20)):
    """
    Monte carlo simulation for samples drawn from Gaussian distributions.
    
    Parameters:
        mu1 (ndarray): mean of distribution 1
        mu2 (ndarray): mean of distribution 2
        cov1 (ndarray): covariance of distribution 1
        cov2 (ndarray): covariance of distribution 2
        method (str): method name from [naive, halfnhalf, unregularized_pooled, regularized_pooled]
        N1 (int): number of trials drawn from distribution 1 for each sample
        N2 (int): number of trials drawn from distribution 2 for each sample
        M (int): number of samples drawn
        
        k (int): number of XV folds
        Cs (list or ndarray): possibilites for inverse of regularization strength
        
    Returns:
        Average of estimated CPs (scalar), variance in distribution of estimated CPs (scalar)
    """
    
    CPs = np.empty((M))
        
    for kMC in range(M):
        x1 = np.random.multivariate_normal(mu1, cov1, size=N1)
        x2 = np.random.multivariate_normal(mu2, cov2, size=N2)
        CPs[kMC] = estimate_CP(x1, x2, method, k, Cs)
        
    return np.mean(CPs), np.var(CPs)

def binary_search_CP(e_CP, method, N1, N2, dim, cov1, cov2, low=0, high=3, M=50, error=0.005, k=5, Cs = np.logspace(-15, 4, 20)):
    """
    Searches for the true CP and distance between means that produces a certain estimated CP with underlying Gaussian distributions assumption.
    
    Parameters:
        e_CP (scalar): estimated CP from sample
        method (str): method name from [naive, halfnhalf, unregularized_pooled, regularized_pooled]
        N1 (int): number of negative trials in sample
        N2 (int): number of positive trials in sample
        dim (int): dimension of sample (number of neurons)
        cov1 (ndarray): covariance of distribution 1
        cov2: covariance of distribution 2
        low (scalar): lower limit for distance between distribution means
        high (scalar): upper limit for distance between distribution means
        
        M (int): number of monte carlo samples to get estimated CP based on a certain true CP
        error (scalar): margin of error
        k (int): number of XV folds
        Cs (list or ndarray): possibilites for inverse of regularization strength
        
    Returns:
        Corresponding true CP (scalar), distance between distribution means for that true CP (scalar)
        
    Note: 'distance' is distance in each dimension, not Euclidean distance. E.g. 'distance' between [0, 0] and [1, 1] is 1.
    """
    
    if e_CP < 0.5:
        return 0.5, 0
    
    dist = (high + low)/2
    mu1 = np.zeros(dim)
    mu2 = mu1 + dist
    
    true_CP = gaussian_ND_true_CP(mu1, mu2, cov1, cov2)
    estimate, var = monte_carlo_CP(mu1, mu2, cov1, cov2, method, N1, N2, M, k, Cs)
    
    if abs(estimate - e_CP) < error:
        return true_CP, dist
    elif (estimate < e_CP):
        return binary_search_CP(e_CP, method, N1, N2, dim, cov1, cov2, dist, high, M, error, k, Cs)
    else:
        return binary_search_CP(e_CP, method, N1, N2, dim, cov1, cov2, low, dist, M, error, k, Cs)
    
def delta_method(points, center, variance, smoothness):
    """
    Based on an estimated CP, a corresponding true CP, and the variance in the empirical distribution for the estimated CP 
    calculate the variance in true CP.
    Steps:
        1. binary search for true CP corresponding to estimated CP
        2. run simulations using this true CP to get find variance in distribution of estimated CPs
        3. run simulations with 4 surrounding true CPs and get the mean estimated CPs
        4. use points from steps 2 and 3 to calculate slope of true CP vs estimated CP at the original estimated CP
        5. use delta method to get variance in true CP
    
    Parameters:
        points (ndarray): column 1 - estimated CPs, column 2 - true CPs (each point is stored in one row),
                        must be sorted by estimated CP
        center: index of central point that contains original estimated CP
        variance (scalar): variance in distribution of estimated CPs at middle point
        smoothness (scalar): smoothness of spline interpolation
        
    Returns:
        Variance in true CP (scalar)
    """
    
    points = points[points[:,0].argsort()]
    
    f = interpolate.splrep(points[:,0], points[:,1], k=3, s=smoothness)
    fprime = interpolate.splder(f)
    deriv = interpolate.splev(points[center,0], fprime)
    
    real_var = variance * (deriv**2)
    return real_var

def estimate_CI(e_CP, method, N1, N2, dim, cov1, cov2, surrounding_points, smoothness,
               low=0, high=3, bs_M=50, error=0.005, final_M=2000, k=5, Cs=np.logspace(-15,4,20)):
    """
    Given an estimated CP calculated from a sample, calculates a 95% confidence interval for the true CP.
    
    Parameters:
        e_CP (scalar): estimated CP from sample
        method (str): method name from [naive, halfnhalf, unregularized_pooled, regularized_pooled]
        N1 (int): number of negative trials in sample
        N2 (int): number of positive trials in sample
        dim (int): dimension of sample (number of neurons)
        cov1 (ndarray): covariance of Gaussian distribution 1 used for binary search 
        cov2 (ndarray): covariance of Gaussian distribution 2 used for binary search
        surrounding_points (int): number of surrounding points *on either side*
        smoothness (scalar): smoothess of spline interpolation for delta method
        
        low (scalar): binary search initial lower limit for distance between distribution means
        high (scalar): binary search initial upper limit for distance between distribution means
        bs_M: binary search number of monte carlo samples
        error (scalar): binary search margin of error
        final_M (int): number of monte carlo samples to get surrounding points and variance of estimated CP
        k (int): number of XV folds
        Cs (list or ndarray): possibilites for inverse of regularization strength
    
    Returns:
        List containing estimated CP, bias corrected CP, confidence interval lower bound, confidence interval upper bound
    """
    
    true_CP, dist = binary_search_CP(e_CP, method, N1, N2, dim, cov1, cov2, low, high, bs_M, error, k, Cs)
    
    mu1 = np.zeros(dim)
    distance_change_normalization = np.sqrt(2/dim)
    
    points = []
    
    for i in range(surrounding_points):
        mu2_above = mu1 + dist + 0.02*(i + 1)*distance_change_normalization
        mu2_below = mu1 + dist - 0.02*(i + 1)*distance_change_normalization
        true_CP_upper = gaussian_ND_true_CP(mu1, mu2_above, cov1, cov2)
        true_CP_lower = gaussian_ND_true_CP(mu1, mu2_below, cov1, cov2)
        estimated_CP_upper = monte_carlo_CP(mu1, mu2_above, cov1, cov2, method, N1, N2, final_M, k=5, Cs=np.logspace(-15, 4, 20))[0]
        estimated_CP_lower = monte_carlo_CP(mu1, mu2_below, cov1, cov2, method, N1, N2, final_M, k=5, Cs=np.logspace(-15, 4, 20))[0]
        
        points.append([estimated_CP_upper, true_CP_upper])
        points.append([estimated_CP_lower, true_CP_lower])
    
    variance_center = monte_carlo_CP(mu1, mu1 + dist, cov1, cov2, method, N1, N2, final_M, k=5, Cs=np.logspace(-15, 4, 20))[1]
    
    points.append([e_CP, true_CP])
    points = np.array(points)
    points = points[points[:,0].argsort()]
    
    true_CP_variance = delta_method(points, surrounding_points, variance_center, smoothness)
    true_CP_SD = true_CP_variance**(1/2)
    
    CI_upper = true_CP + 2*true_CP_SD
    CI_lower = true_CP - 2*true_CP_SD

    return [e_CP, true_CP, CI_lower, CI_upper]