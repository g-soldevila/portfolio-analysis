import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import minimize
from numpy.linalg import inv

# ---------------------------------------------------------------------------------
# Return Analysis and general statistics
# ---------------------------------------------------------------------------------

def terminal_wealth(s):
    '''
    Computes the terminal wealth of a sequence of return, which is, in other words, 
    the final compounded return. 
    The input s is expected to be either a pd.DataFrame or a pd.Series
    '''
    if not isinstance(s, (pd.DataFrame, pd.Series)):
        raise ValueError("Expected either a pd.DataFrame or pd.Series")
    return (1 + s).prod()

def compound(s):
    '''
    Single compound rule for a pd.Dataframe or pd.Series of returns. 
    The method returns a single number - using prod(). 
    See also the TERMINAL_WEALTH method.
    '''
    if not isinstance(s, (pd.DataFrame, pd.Series)):
        raise ValueError("Expected either a pd.DataFrame or pd.Series")
    return (1 + s).prod() - 1
    # Note that this is equivalent to (but slower than)
    # return np.expm1( np.logp1(s).sum() )
    
def compound_returns(s, start=100):
    '''
    Compound a pd.Dataframe or pd.Series of returns from an initial default value equal to 100.
    In the former case, the method compounds the returns for every column (Series) by using pd.aggregate. 
    The method returns a pd.Dataframe or pd.Series - using cumprod(). 
    See also the COMPOUND method.
    '''
    if isinstance(s, pd.DataFrame):
        return s.aggregate( compound_returns, start=start )
    elif isinstance(s, pd.Series):
        return start * (1 + s).cumprod()
    else:
        raise TypeError("Expected pd.DataFrame or pd.Series")
        
def compute_returns(s):
    '''
    Computes the returns (percentage change) of a Dataframe of Series. 
    In the former case, it computes the returns for every column (Series) by using pd.aggregate
    '''
    if isinstance(s, pd.DataFrame):
        return s.aggregate( compute_returns )
    elif isinstance(s, pd.Series):
        return s / s.shift(1) - 1
    else:
        raise TypeError("Expected pd.DataFrame or pd.Series")
        
def compute_logreturns(s):
    '''
    Computes the log-returns of a Dataframe of Series. 
    In the former case, it computes the returns for every column (Series) by using pd.aggregate
    '''
    if isinstance(s, pd.DataFrame):
        return s.aggregate( compute_logreturns )
    elif isinstance(s, pd.Series):
        return np.log( s / s.shift(1) )
    else:
        raise TypeError("Expected pd.DataFrame or pd.Series")
    
def drawdown(rets: pd.Series, start=1000):
    '''
    Compute the drawdowns of an input pd.Series of returns. 
    The method returns a dataframe containing: 
    1. the associated wealth index (for an hypothetical starting investment of $1000) 
    2. all previous peaks 
    3. the drawdowns
    '''
    wealth_index   = compound_returns(rets, start=start)
    previous_peaks = wealth_index.cummax()
    drawdowns      = (wealth_index - previous_peaks ) / previous_peaks
    df = pd.DataFrame({"Wealth": wealth_index, "Peaks": previous_peaks, "Drawdown": drawdowns} )
    return df

def skewness(s):
    '''
    Computes the Skewness of the input Series or Dataframe.
    There is also the function scipy.stats.skew().
    '''
    return ( ((s - s.mean()) / s.std(ddof=0))**3 ).mean()

def kurtosis(s):
    '''
    Computes the Kurtosis of the input Series or Dataframe.
    There is also the function scipy.stats.kurtosis() which, however, 
    computes the "Excess Kurtosis", i.e., Kurtosis minus 3
    '''
    return ( ((s - s.mean()) / s.std(ddof=0))**4 ).mean()

def exkurtosis(s):
    '''
    Returns the Excess Kurtosis, i.e., Kurtosis minus 3
    '''
    return kurtosis(s) - 3

def is_normal(s, level=0.01):
    '''
    Jarque-Bera test to see if a series (of returns) is normally distributed.
    Returns True or False according to whether the p-value is larger 
    than the default level=0.01.
    '''
    statistic, pvalue = scipy.stats.jarque_bera( s )
    return pvalue > level

def semivolatility(s):
    '''
    Returns the semivolatility of a series, i.e., the volatility of
    negative returns
    '''
    return s[s<0].std(ddof=0) 

def var_historic(s, level=0.05):
    '''
    Returns the (1-level)% VaR using historical method. 
    By default it computes the 95% VaR, i.e., alpha=0.95 which gives level 1-alpha=0.05.
    The method takes in input either a DataFrame or a Series and, in the former 
    case, it computes the VaR for every column (Series) by using pd.aggregate
    '''
    if isinstance(s, pd.DataFrame):
        return s.aggregate( var_historic, level=level )
    elif isinstance(s, pd.Series):
        return - np.percentile(s, level*100)
    else:
        raise TypeError("Expected pd.DataFrame or pd.Series")
        
def var_gaussian(s, level=0.05, cf=False):
    '''
    Returns the (1-level)% VaR using the parametric Gaussian method. 
    By default it computes the 95% VaR, i.e., alpha=0.95 which gives level 1-alpha=0.05.
    The variable "cf" stands for Cornish-Fisher. If True, the method computes the 
    modified VaR using the Cornish-Fisher expansion of quantiles.
    The method takes in input either a DataFrame or a Series and, in the former 
    case, it computes the VaR for every column (Series).
    '''
    # alpha-quantile of Gaussian distribution 
    za = scipy.stats.norm.ppf(level,0,1) 
    if cf:
        S = skewness(s)
        K = kurtosis(s)
        za = za + (za**2 - 1)*S/6 + (za**3 - 3*za)*(K-3)/24 - (2*za**3 - 5*za)*(S**2)/36    
    return -( s.mean() + za * s.std(ddof=0) )

def cvar_historic(s, level=0.05):
    '''
    Computes the (1-level)% Conditional VaR (based on historical method).
    By default it computes the 95% CVaR, i.e., alpha=0.95 which gives level 1-alpha=0.05.
    The method takes in input either a DataFrame or a Series and, in the former 
    case, it computes the VaR for every column (Series).
    '''
    if isinstance(s, pd.DataFrame):
        return s.aggregate( cvar_historic, level=level )
    elif isinstance(s, pd.Series):
        # find the returns which are less than (the historic) VaR
        mask = s < -var_historic(s, level=level)
        # and of them, take the mean 
        return -s[mask].mean()
    else:
        raise TypeError("Expected pd.DataFrame or pd.Series")

def annualize_rets(s, periods_per_year):
    '''
    Computes the return per year, or, annualized return.
    The variable periods_per_year can be, e.g., 12, 52, 252, in 
    case of monthly, weekly, and daily data.
    The method takes in input either a DataFrame or a Series and, in the former 
    case, it computes the annualized return for every column (Series) by using pd.aggregate
    '''
    if isinstance(s, pd.DataFrame):
        return s.aggregate( annualize_rets, periods_per_year=periods_per_year )
    elif isinstance(s, pd.Series):
        growth = (1 + s).prod()
        n_period_growth = s.shape[0]
        return growth**(periods_per_year/n_period_growth) - 1

def annualize_vol(s, periods_per_year):
    '''
    Computes the volatility per year, or, annualized volatility.
    The variable periods_per_year can be, e.g., 12, 52, 252, in 
    case of monthly, weekly, and daily data.
    The method takes in input either a DataFrame, a Series, a list or a single number. 
    In the former case, it computes the annualized volatility of every column 
    (Series) by using pd.aggregate. In the latter case, s is a volatility 
    computed beforehand, hence only annulization is done
    '''
    if isinstance(s, pd.DataFrame):
        return s.aggregate(annualize_vol, periods_per_year=periods_per_year )
    elif isinstance(s, pd.Series):
        return s.std() * (periods_per_year)**(0.5)
    elif isinstance(s, list):
        return np.std(s) * (periods_per_year)**(0.5)
    elif isinstance(s, (int,float)):
        return s * (periods_per_year)**(0.5)

def sharpe_ratio(s, risk_free_rate, periods_per_year, v=None):
    '''
    Computes the annualized sharpe ratio. 
    The variable periods_per_year can be, e.g., 12, 52, 252, in case of yearly, weekly, and daily data.
    The variable risk_free_rate is the annual one.
    The method takes in input either a DataFrame, a Series or a single number. 
    In the former case, it computes the annualized sharpe ratio of every column (Series) by using pd.aggregate. 
    In the latter case, s is the (allready annualized) return and v is the (already annualized) volatility 
    computed beforehand, for example, in case of a portfolio.
    '''
    if isinstance(s, pd.DataFrame):
        return s.aggregate( sharpe_ratio, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year, v=None)
    elif isinstance(s, pd.Series):
        # convert the annual risk free rate to the period assuming that:
        # RFR_year = (1+RFR_period)^{periods_per_year} - 1. Hence:
        rf_to_period = (1 + risk_free_rate)**(1/periods_per_year) - 1        
        excess_return = s - rf_to_period
        # now, annualize the excess return
        ann_ex_rets = annualize_rets(excess_return, periods_per_year)
        # compute annualized volatility
        ann_vol = annualize_vol(s, periods_per_year)
        return ann_ex_rets / ann_vol
    elif isinstance(s, (int,float)) and v is not None:
        # Portfolio case: s is supposed to be the single (already annnualized) 
        # return of the portfolio and v to be the single (already annualized) volatility. 
        return (s - risk_free_rate) / v


# ---------------------------------------------------------------------------------
# Modern Portfolio Theory 
# ---------------------------------------------------------------------------------
def portfolio_return(weights, vec_returns):
    '''
    Computes the return of a portfolio. 
    It takes in input a row vector of weights (list of np.array) 
    and a column vector (or pd.Series) of returns
    '''
    return np.dot(weights, vec_returns)
    
def portfolio_volatility(weights, cov_rets):
    '''
    Computes the volatility of a portfolio. 
    It takes in input a vector of weights (np.array or pd.Series) 
    and the covariance matrix of the portfolio asset returns
    '''
    return ( np.dot(weights.T, np.dot(cov_rets, weights)) )**(0.5) 

def efficient_frontier(n_portfolios, rets, covmat, periods_per_year, risk_free_rate=0.0, 
                       iplot=False, hsr=False, cml=False, mvp=False, ewp=False):
    '''
    Returns (and plots) the efficient frontiers for a portfolio of rets.shape[1] assets. 
    The method returns a dataframe containing the volatilities, returns, sharpe ratios and weights 
    of the portfolios as well as a plot of the efficient frontier in case iplot=True. 
    Other inputs are:
        hsr: if true the method plots the highest return portfolio,
        cml: if True the method plots the capital market line;
        mvp: if True the method plots the minimum volatility portfolio;
        ewp: if True the method plots the equally weigthed portfolio. 
    The variable periods_per_year can be, e.g., 12, 52, 252, in case of monthly, weekly, and daily data.
    '''   
    
    def append_row_df(df,vol,ret,spr,weights):
        temp_df = list(df.values)
        temp_df.append( [vol, ret, spr,] + [w for w in weights] )
        return pd.DataFrame(temp_df)
        
    ann_rets = annualize_rets(rets, periods_per_year)
    
    # generates optimal weights of porfolios lying of the efficient frontiers
    weights = optimal_weights(n_portfolios, ann_rets, covmat, periods_per_year) 
    # in alternative, if only the portfolio consists of only two assets, the weights can be: 
    #weights = [np.array([w,1-w]) for w in np.linspace(0,1,n_portfolios)]

    # portfolio returns
    portfolio_ret = [portfolio_return(w, ann_rets) for w in weights]
    
    # portfolio volatility
    vols          = [portfolio_volatility(w, covmat) for w in weights] 
    portfolio_vol = [annualize_vol(v, periods_per_year) for v in vols]
    
    # portfolio sharpe ratio
    portfolio_spr = [sharpe_ratio(r, risk_free_rate, periods_per_year, v=v) for r,v in zip(portfolio_ret,portfolio_vol)]
    
    df = pd.DataFrame({"volatility": portfolio_vol,
                       "return": portfolio_ret,
                       "sharpe ratio": portfolio_spr})
    df = pd.concat([df, pd.DataFrame(weights)],axis=1)
    
    if iplot:
        ax = df.plot.line(x="volatility", y="return", style="--", color="coral", grid=True, label="Efficient frontier", figsize=(8,4))
        if hsr or cml:
            w   = maximize_shape_ratio(ann_rets, covmat, risk_free_rate, periods_per_year)
            ret = portfolio_return(w, ann_rets)
            vol = annualize_vol( portfolio_volatility(w,covmat), periods_per_year)
            spr = sharpe_ratio(ret, risk_free_rate, periods_per_year, v=vol)
            df  = append_row_df(df,vol,ret,spr,w)
            if cml:
                # Draw the CML: the endpoints of the CML are [0,risk_free_rate] and [port_vol,port_ret]
                ax.plot([0, vol], [risk_free_rate, ret], color="g", linestyle="-.", label="CML")
                ax.set_xlim(left=0)
                ax.legend()
            if hsr:
                # Plot the highest sharpe ratio portfolio
                ax.scatter([vol], [ret], marker="o", color="g", label="MSR portfolio")
                ax.legend()
        if mvp:
            # Plot the global minimum portfolio:
            w   = minimize_volatility(ann_rets, covmat)
            ret = portfolio_return(w, ann_rets)
            vol = annualize_vol( portfolio_volatility(w,covmat), periods_per_year)
            spr = sharpe_ratio(ret, risk_free_rate, periods_per_year, v=vol)
            df  = append_row_df(df,vol,ret,spr,w)
            ax.scatter([vol], [ret], color="midnightblue", marker="o", label="GMV portfolio")
            ax.legend()  
        if ewp:
            # Plot the equally weighted portfolio:
            w   = np.repeat(1/ann_rets.shape[0], ann_rets.shape[0])
            ret = portfolio_return(w, ann_rets)
            vol = annualize_vol( portfolio_volatility(w,covmat), periods_per_year)
            spr = sharpe_ratio(ret, risk_free_rate, periods_per_year, v=vol)
            df  = append_row_df(df,vol,ret,spr,w)
            ax.scatter([vol], [ret], color="goldenrod", marker="o", label="EW portfolio")
            ax.legend()
        return df, ax
    else: 
        return df
    
def summary_stats(s, risk_free_rate=0.03, periods_per_year=12, var_level=0.05):
    '''
    Returns a dataframe containing annualized returns, annualized volatility, sharpe ratio, 
    skewness, kurtosis, historic VaR, Cornish-Fisher VaR, and Max Drawdown
    '''
    if isinstance(s, pd.Series):
        stats = {
            "Ann. return"  : annualize_rets(s, periods_per_year=periods_per_year),
            "Ann. vol"     : annualize_vol(s, periods_per_year=periods_per_year),
            "Sharpe ratio" : sharpe_ratio(s, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year),
            "Skewness"     : skewness(s),
            "Kurtosis"     : kurtosis(s),
            "Historic CVar": cvar_historic(s, level=var_level),
            "C-F Var"      : var_gaussian(s, level=var_level, cf=True),
            "Max drawdown" : drawdown(s)["Drawdown"].min()
        }
        return pd.DataFrame(stats, index=["0"])
    
    elif isinstance(s, pd.DataFrame):        
        stats = {
            "Ann. return"  : s.aggregate( annualize_rets, periods_per_year=periods_per_year ),
            "Ann. vol"     : s.aggregate( annualize_vol,  periods_per_year=periods_per_year ),
            "Sharpe ratio" : s.aggregate( sharpe_ratio, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year ),
            "Skewness"     : s.aggregate( skewness ),
            "Kurtosis"     : s.aggregate( kurtosis ),
            "Historic CVar": s.aggregate( cvar_historic, level=var_level ),
            "C-F Var"      : s.aggregate( var_gaussian, level=var_level, cf=True ),
            "Max Drawdown" : s.aggregate( lambda r: drawdown(r)["Drawdown"].min() )
        } 
        return pd.DataFrame(stats)
     
def optimal_weights(n_points, rets, covmatrix, periods_per_year):
    '''
    Returns a set of n_points optimal weights corresponding to portfolios (of the efficient frontier) 
    with minimum volatility constructed by fixing n_points target returns. 
    The weights are obtained by solving the minimization problem for the volatility. 
    '''
    target_rets = np.linspace(rets.min(), rets.max(), n_points)    
    weights = [minimize_volatility(rets, covmatrix, target) for target in target_rets]
    return weights

def minimize_volatility(rets, covmatrix, target_return=None):
    '''
    Returns the optimal weights of the minimum volatility portfolio on the effient frontier. 
    If target_return is not None, then the weights correspond to the minimum volatility portfolio 
    having a fixed target return. 
    The method uses the scipy minimize optimizer which solves the minimization problem 
    for the volatility of the portfolio
    '''
    n_assets = rets.shape[0]    
    # initial guess weights
    init_guess = np.repeat(1/n_assets, n_assets)
    weights_constraint = {
        "type": "eq",
        "fun": lambda w: 1.0 - np.sum(w)  
    }
    if target_return is not None:
        return_constraint = {
            "type": "eq",
            "args": (rets,),
            "fun": lambda w, r: target_return - portfolio_return(w, r)
        }
        constr = (return_constraint, weights_constraint)
    else:
        constr = weights_constraint
        
    result = minimize(portfolio_volatility, 
                      init_guess,
                      args = (covmatrix,),
                      method = "SLSQP",
                      options = {"disp": False},
                      constraints = constr,
                      bounds = ((0.0,1.0),)*n_assets ) # bounds of each individual weight, i.e., w between 0 and 1
    return result.x

def minimize_volatility_2(rets, covmatrix, target_return=None, weights_norm_const=True, weights_bound_const=True):
    '''
    Returns the optimal weights of the minimum volatility portfolio.
    If target_return is not None, then the weights correspond to the minimum volatility portfolio 
    having a fixed target return (such portfolio will be on the efficient frontier).
    The variables weights_norm_const and weights_bound_const impose two more conditions, the firt one on 
    weight that sum to 1, and the latter on the weights which have to be between zero and 1
    The method uses the scipy minimize optimizer which solves the minimization problem 
    for the volatility of the portfolio
    '''
    n_assets = rets.shape[0]    
    
    # initial guess weights
    init_guess = np.repeat(1/n_assets, n_assets)
    
    if weights_bound_const:
        # bounds of the weights (between 0 and 1)
        bounds = ((0.0,1.0),)*n_assets
    else:
        bounds = None
    
    constraints = []
    if weights_norm_const:
        weights_constraint = {
            "type": "eq",
            "fun": lambda w: 1.0 - np.sum(w)  
        }
        constraints.append( weights_constraint )    
    if target_return is not None:
        return_constraint = {
            "type": "eq",
            "args": (rets,),
            "fun": lambda w, r: target_return - portfolio_return(w, r)
        }
        constraints.append( return_constraint )
    
    result = minimize(portfolio_volatility, 
                      init_guess,
                      args = (covmatrix,),
                      method = "SLSQP",
                      options = {"disp": False},
                      constraints = tuple(constraints),
                      bounds = bounds)
    return result.x

def maximize_shape_ratio(rets, covmatrix, risk_free_rate, periods_per_year, target_volatility=None):
    '''
    Returns the optimal weights of the highest sharpe ratio portfolio on the effient frontier. 
    If target_volatility is not None, then the weights correspond to the highest sharpe ratio portfolio 
    having a fixed target volatility. 
    The method uses the scipy minimize optimizer which solves the maximization of the sharpe ratio which 
    is equivalent to minimize the negative sharpe ratio.
    '''
    n_assets   = rets.shape[0] 
    init_guess = np.repeat(1/n_assets, n_assets)
    weights_constraint = {
        "type": "eq",
        "fun": lambda w: 1.0 - np.sum(w)  
    }
    if target_volatility is not None:
        volatility_constraint = {
            "type": "eq",
            "args": (covmatrix, periods_per_year),
            "fun": lambda w, cov, p: target_volatility - annualize_vol(portfolio_volatility(w, cov), p)
        }
        constr = (volatility_constraint, weights_constraint)
    else:
        constr = weights_constraint
        
    def neg_portfolio_sharpe_ratio(weights, rets, covmatrix, risk_free_rate, periods_per_year):
        '''
        Computes the negative annualized sharpe ratio for minimization problem of optimal portfolios.
        The variable periods_per_year can be, e.g., 12, 52, 252, in case of yearly, weekly, and daily data.
        The variable risk_free_rate is the annual one.
        '''
        # annualized portfolio returns
        portfolio_ret = portfolio_return(weights, rets)        
        # annualized portfolio volatility
        portfolio_vol = annualize_vol(portfolio_volatility(weights, covmatrix), periods_per_year)
        return - sharpe_ratio(portfolio_ret, risk_free_rate, periods_per_year, v=portfolio_vol)    
        #i.e., simply returns  -(portfolio_ret - risk_free_rate)/portfolio_vol
        
    result = minimize(neg_portfolio_sharpe_ratio,
                      init_guess,
                      args = (rets, covmatrix, risk_free_rate, periods_per_year),
                      method = "SLSQP",
                      options = {"disp": False},
                      constraints = constr,
                      bounds = ((0.0,1.0),)*n_assets)
    return result.x

def weigths_max_sharpe_ratio(covmat, mu_exc, scale=True):
    '''
    Optimal (Tangent/Max Sharpe Ratio) portfolio weights using the Markowitz Optimization Procedure:
    - mu_exc is the vector of Excess expected Returns (has to be a column vector as a pd.Series)
    - covmat is the covariance N x N matrix as a pd.DataFrame
    Look at pag. 188 eq. (5.2.28) of "The econometrics of financial markets", by Campbell, Lo, Mackinlay.
    '''
    w = inverse_df(covmat).dot(mu_exc)
    if scale:
        # normalize weigths
        w = w/sum(w) 
    return w

# ---------------------------------------------------------------------------------
# Factor and Style analysis 
# ---------------------------------------------------------------------------------

def linear_regression(dep_var, exp_vars, alpha=True):
    '''
    Runs a linear regression to decompose the dependent variable into the explanatory variables 
    using statsmodels OLS method. 
    It returns the object of type statsmodel's RegressionResults on which we can call on it:
    - .summary() to print a full summary
    - .params for the coefficients
    - .tvalues and .pvalues for the significance levels
    - .rsquared_adj and .rsquared for quality of fit
    Note that exp.vars can be both a pd.DataFrame a np.array.
    '''
    if alpha:
        # the OLS methods assume a bias equal to 0, hence a specific variable for the bias has to be given
        if isinstance(exp_vars,pd.DataFrame):
            exp_vars = exp_vars.copy()
            exp_vars["Alpha"] = 1
        else:
            exp_vars = np.concatenate( (exp_vars, np.ones((exp_vars.shape[0],1))), axis=1 )
    return sm.OLS(dep_var, exp_vars).fit()

def capm_betas(ri, rm):
    '''
    Returns the CAPM factor exposures beta for each asset in the ri pd.DataFrame, 
    where rm is the pd.DataFrame (or pd.Series) of the market return (not excess return).
    The betas are defined as:
      beta_i = Cov(r_i, rm) / Var(rm)
    with r_i being the ith column (i.e., asset) of DataFrame ri.
    '''
    market_var = ( rm.std()**2 )[0]
    betas = []
    for name in ri.columns:
        cov_im = pd.concat( [ri[name],rm], axis=1).cov().iloc[0,1]
        betas.append( cov_im / market_var )
    return pd.Series(betas, index=ri.columns)

def tracking_error(r_a, r_b):
    '''
    Returns the tracking error between two return series. 
    This method is used in Sharpe Analysis minimization problem.
    See STYLE_ANALYSIS method.
    '''
    return ( ((r_a - r_b)**2).sum() )**(0.5)

def style_analysis_tracking_error(weights, ref_r, bb_r):
    '''
    Sharpe style analysis objective function.
    Returns the tracking error between the reference returns
    and a portfolio of building block returns held with given weights. 
    '''
    return tracking_error(ref_r, (weights*bb_r).sum(axis=1))

def style_analysis(dep_var, exp_vars):
    '''
    Sharpe style analysis optimization problem.
    Returns the optimal weights that minimizes the tracking error between a portfolio 
    of the explanatory (return) variables and the dependent (return) variable.
    '''
    # dep_var is expected to be a pd.Series
    if isinstance(dep_var,pd.DataFrame):
        dep_var = dep_var[dep_var.columns[0]]
    
    n = exp_vars.shape[1]
    init_guess = np.repeat(1/n, n)
    weights_const = {
        'type': 'eq',
        'fun': lambda weights: 1 - np.sum(weights)
    }
    solution = minimize(style_analysis_tracking_error, 
                        init_guess,
                        method='SLSQP',
                        options={'disp': False},
                        args=(dep_var, exp_vars),
                        constraints=(weights_const,),
                        bounds=((0.0, 1.0),)*n)
    weights = pd.Series(solution.x, index=exp_vars.columns)
    return weights

def sample_cov(r, **kwargs):
    '''
    Returns the sample covariance of the supplied series of returns (a pd.DataFrame) 
    '''
    if not isinstance(r,pd.DataFrame):
        raise ValueError("Expected r to be a pd.DataFrame of returns series")
    return r.cov()

def cc_cov(r, **kwargs):
    '''
    Estimates a covariance matrix by using the Elton/Gruber Constant Correlation model
    '''
    # correlation coefficents  
    rhos = r.corr()
    n = rhos.shape[0]
    # compute the mean correlation: since the matrix rhos is a symmetric with diagonals all 1, 
    # the mean correlation can be computed by:
    mean_rho = (rhos.values.sum() - n) / (n**2-n) 
    # create the constant correlation matrix containing 1 on the diagonal and the mean correlation outside
    ccor = np.full_like(rhos, mean_rho)
    np.fill_diagonal(ccor, 1.)
    # create the new covariance matrix by multiplying mean_rho*std_i*std_i 
    # the product of the stds is done via np.outer
    ccov = ccor * np.outer(r.std(), r.std())
    return pd.DataFrame(ccov, index=r.columns, columns=r.columns)

def factor_model_cov(r, factors=None, risk_free=None, **kwargs):    
    # Get a slice of the factor data with the same date range as r    
    r, factors, risk_free = adjust_date_indexes([r, factors, risk_free])
    n_assets = r.shape[1]
    m = np.zeros(shape=(n_assets,n_assets))
    
    factors_cov = factors.cov()
         
    for i in range(n_assets):
        for j in range(n_assets):
            i_betas = linear_regression(r.iloc[:,i] - risk_free, factors, alpha=False).params
            j_betas = linear_regression(r.iloc[:,j] - risk_free, factors, alpha=False).params
            m[i][j] = i_betas.to_numpy() @ factors_cov @ np.array([j_betas]).T
                        
    return m

def shrinkage_cov(r, delta=0.5, **kwargs):
    '''
    Statistical shrinkage: it returns a covariance matrix estimator that shrinks between 
    the constant correlation and standard sample covariance estimators 
    '''
    samp_cov  = sample_cov(r, **kwargs)
    const_cov = cc_cov(r, **kwargs)
    return delta*const_cov + (1-delta)*samp_cov

# ---------------------------------------------------------------------------------
# Back-test weigthing schemes
# ---------------------------------------------------------------------------------

def weight_ew(r, cap_ws=None, max_cw_mult=None, microcap_thr=None, **kwargs):
    """
    Returns the weights of the Equally-Weighted (EW) portfolio based on the asset returns "r" as a DataFrame. 
    If the set of cap_ws is given, the modified scheme is computed, i.e., 
    microcaps are removed and a capweight tether applied.
    """
    ew = pd.Series(1/len(r.columns), index=r.columns)
    if cap_ws is not None:
        cw = cap_ws.loc[r.index[0]] # starting cap weight
        if microcap_thr is not None and microcap_thr > 0.0:
            # exclude microcaps according to the threshold    
            ew[ cw < microcap_thr ] = 0
            ew = ew / ew.sum()
        if max_cw_mult is not None and max_cw_mult > 0:
            # limit weight up to a multiple of capweight
            ew = np.minimum(ew, cw*max_cw_mult)
            ew = ew / ew.sum()
    return ew

def weight_cw(r, cap_ws, **kwargs):
    '''
    Returns the weights of the Cap-Weigthed (CW) portfolio based on the time series of capweights
    '''
    return cap_ws.loc[r.index[0]]
    
def weight_rp(r, cov_estimator=sample_cov, **kwargs):
    '''
    Produces the weights of the risk parity portfolio given a covariance matrix of the returns.
    The default coavariance estimator is the sample covariance matrix.
    '''
    est_cov = cov_estimator(r, **kwargs)
    return risk_parity_weigths(est_cov)  

def backtest_weight_scheme(r, window=36, weight_scheme=weight_ew, **kwargs):
    '''
    Backtests a given weighting scheme. Here:
    - r: asset returns to use to build the portfolio
    - window: the rolling window used
    - weight_scheme: the weighting scheme to use, it must the name of a 
    method that takes "r", and a variable number of keyword-value arguments
    '''
    n_periods = r.shape[0]
    windows = [ (start, start+window) for start in range(0,n_periods-window) ]
    weights = [ weight_scheme( r.iloc[win[0]:win[1]], **kwargs) for win in windows ]
    weights = pd.DataFrame(weights, index=r.iloc[window:].index, columns=r.columns)    
    returns = (weights * r).sum(axis=1,  min_count=1) #mincount is to generate NAs if all inputs are NAs
    return returns, weights

def annualize_vol_ewa(r, decay=0.95, periods_per_year=12):
    '''
    Computes the annualized exponentially weighted average volatility of a 
    series of returns given a decay (smoothing) factor in input. 
    '''
    N = r.shape[0]
    times = np.arange(0,N,1)
    # compute the square error returns
    sq_errs = pd.DataFrame( ( r - r.mean() )**2 )
    # exponential weights
    weights = [ decay**(N-t) for t in times ] / np.sum(decay**(N-times))
    weights = pd.DataFrame(weights, index=r.index)
    # EWA
    vol_ewa = (weights * sq_errs).sum()**(0.5)
    # Annualize the computed volatility
    ann_vol_ewa = vol_ewa[0] * np.sqrt(periods_per_year)
    return ann_vol_ewa

def weight_minvar(r, cov_estimator=sample_cov, periods_per_year=12, **kwargs):
    '''
    Produces the weights of the Minimum Volatility Portfolio given a covariance matrix of the returns 
    '''
    est_cov = cov_estimator(r, **kwargs)
    ann_ret = annualize_rets(r, periods_per_year=12)
    return minimize_volatility(ann_ret, est_cov)

def weight_maxsharpe(r, cov_estimator=sample_cov, periods_per_year=12, risk_free_rate=0.03, **kwargs):
    '''
    Produces the weights of the Maximum Sharpe Ratio Portfolio given a covariance matrix of the returns 
    '''
    est_cov = cov_estimator(r, **kwargs)
    ann_ret = annualize_rets(r, periods_per_year=12)
    return maximize_shape_ratio(ann_ret, est_cov, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year)

# ---------------------------------------------------------------------------------
# Black-Litterman model
# ---------------------------------------------------------------------------------

def implied_returns(covmat, weigths, delta=2.5):
    '''
    Computes the implied expected returns \Pi by reverse engineering the weights according to 
    the Black-Litterman model:
       \Pi = \delta \Sigma weigths
    Here, the inputs are:
    - delta, the risk aversion coefficient
    - covmat: variance-covariance matrix (N x N) as pd.DataFrame (\Sigma)
    - weigths: portfolio weights (N x 1) as pd.Series 
    The output is the \Pi returns pd.Series (N x 1) 
    '''
    imp_rets = delta * covmat.dot(weigths).squeeze() # to get a series from a 1-column dataframe
    imp_rets.name = 'Implied Returns'
    return imp_rets

def omega_uncertain_prior(covmat, tau, P):
    '''
    Returns the He-Litterman simplified Omega matrix in case the investor does not explicitly 
    quantify the uncertainty on the views. This matrix is going to be:
       \Omega := diag( P(\tau\Sigma)P^T ) 
    Inputs:
    - covmat: N x N covariance Matrix as pd.DataFrame (\Sigma)
    - tau: a scalar denoting the uncertainty of the CAPM prior
    - P: the Projection K x N matrix as a pd.DataFrame.
    The output is a P x P matrix as a pd.DataFrame, representing the Prior Uncertainties.
    '''
    he_lit_omega = P.dot(tau * covmat).dot(P.T)
    # Make a diag matrix from the diag elements of Omega
    return pd.DataFrame( np.diag(np.diag(he_lit_omega.values)), index=P.index, columns=P.index )

def black_litterman(w_prior, Sigma_prior, P, Q, Omega=None, delta=2.5, tau=0.02):
    '''
    Black-Litterman model.
    Computes the posterior expected returns and covariaces based on the original Black-Litterman model 
    using the Master formulas, where:
    - w_prior is the N x 1 pd.Series of prior weights
    - Sigma_prior is the N x N covariance matrix as a pd.DataFrame
    - P is the projection K x N matrix of weights portfolio views, a pd.DataFrame
    - Q is the K x 1 pd.Series of views
    - Omega is the K x K matrix as a pd.DataFrame (or None) representing the uncertainty of the views. 
      In particualar, if Omega=None, we assume that it is proportional to variance of the prior (see OMEGA_UNCERTAIN_PRIOR).
    - delta is the risk aversion (scalar)
    - tau represents the uncertainty of the CAPM prior (scalar)
    '''
    if Omega is None:
        Omega = omega_uncertain_prior(Sigma_prior, tau, P)
    # Force w.prior and Q to be column vectors
    #w_prior = as_colvec(w_prior)
    #Q = as_colvec(Q)
    
    # number of assets
    N = w_prior.shape[0]

    # number of views
    K = Q.shape[0]
    
    # First, reverse-engineer the weights to get \Pi = \delta\Sigma\w^{prior}
    Pi = implied_returns(Sigma_prior,  w_prior, delta)
        
    # Black-Litterman posterior estimate (Master Formulas), using the versions that do not require Omega to be inverted
    invmat   = inv( P.dot(tau * Sigma_prior).dot(P.T) + Omega )
    mu_bl    = Pi + (tau * Sigma_prior).dot(P.T).dot(invmat.dot(Q - P.dot(Pi).values))
    sigma_bl = Sigma_prior + (tau * Sigma_prior) - (tau * Sigma_prior).dot(P.T).dot(invmat).dot(P).dot(tau * Sigma_prior)
    return (mu_bl, sigma_bl)


# ---------------------------------------------------------------------------------
# Helpers for working with DataFrames
# ---------------------------------------------------------------------------------

# Derived fields
# --------------

def get_market_caps(portfolio_returns, porfolio_info, normalized=True):
    portfolio_market_caps = pd.DataFrame()
    portfolio_market_caps.index = portfolio_returns.index

    for ticker in portfolio_returns.columns:

        ticker_market_cap = porfolio_info["marketCap"][ticker]
        ticker_returns = portfolio_returns[ticker]

        marketCaps = [ticker_market_cap]
        previous_cap = ticker_market_cap

        reversed_returns = ticker_returns[::-1][:-1]
        for r in reversed_returns:
            c = previous_cap/(1 + r)
            marketCaps.append(c)
            previous_cap = c

        portfolio_market_caps[ticker] = marketCaps[::-1]
        
    if normalized:
        return normalize_market_caps(portfolio_market_caps)
        
    return portfolio_market_caps

def normalize_market_caps(market_caps):
    return market_caps.div(market_caps.sum(axis=1), axis=0)

# Working with dates
# ------------------

def resample_returns(rets, period="M"):
    rets = rets.resample(period).apply(compound).to_period(period)
    return rets

def resample_market_caps(market_caps, period="M"):
    market_caps = market_caps.resample(period).last().to_period(period)
    return market_caps

def get_max_compatible_date_range(df_array):
    '''
    Gets max compatible date range from 2 dataframes
    '''
    mins = []
    maxs = []
    
    for df in df_array:
        mins.append(df.index.min())
        maxs.append(df.index.max())
    
    return [max(mins), min(maxs)]

def adjust_date_indexes(df_array):    
    max_date_range = get_max_compatible_date_range(df_array)
    return [df[max_date_range[0]:max_date_range[1]] for df in df_array]

# Working with matrices
# ---------------------

def as_colvec(x):
    '''
    In order to consistently use column vectors, this method takes either a np.array or a np.column 
    matrix (i.e., a column vector) and returns the input data as a column vector.
    '''
    if (x.ndim == 2):
        return x
    else:
        return np.expand_dims(x, axis=1)
    
def inverse_df(d):
    '''
    Inverse of a pd.DataFrame (i.e., inverse of dataframe.values)
    '''
    return pd.DataFrame( inv(d.values), index=d.columns, columns=d.index)

def insert_first_row_df(df, row):
    '''
    The method inserts a row at the beginning of a given dataframe and shift by one existing rows.
    The input row has to be either a single element (in case of 1-column dataframe) or 
    a dictionary in case of multi-column dataframe.
    
    Example:
        df  = pd.DataFrame([1, 2, 3])
        row = 0.5
        df  = insert_first_row_df(df, row)
    Example:    
        df  = pd.DataFrame([[1,2],[34,12]], columns=["A","B"])
        row = {"A":-34, "B":443}
        df  = insert_first_row_df(df, row)        
    '''
    df.loc[-1] = row
    df.index = df.index + 1
    return df.sort_index()