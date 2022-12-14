import numpy as np
from scipy import optimize

# local
import blocks
class Fonttype:
    HEADER = '\033[1m' + '\033[94m'
    END = '\033[0m'

    
def find_household_consumption_ss(model):
    """ find household behavior in steady state given... """
    
    par = model.par
    ss = model.ss

    result = optimize.root_scalar(household_ss,
        bracket=[0.0001,1000],method='brentq',args=(par,ss,))
    
    household_ss(result.root,par,ss)

    return result
    
def household_ss(Bq,par,ss):
    """ household behavior in steady state """

    ss.Bq = Bq
    # a. income 
    ss.inc_a[:] = (1-ss.tau)*ss.W*ss.L_a+(1-par.Lambda)*ss.Bq/par.A
    
    # b. HtM 
    ss.C_HtM_a[:] = ss.inc_a/ss.P_C
    ss.B_HtM_a[:] = np.zeros(par.A)
    
    # a. find consumption using final savings and Euler
    for i in range(par.A):

        a = par.A-1-i
        if i == 0:
            RHS = par.mu_B*(ss.Bq/ss.P_C)**(-par.sigma)
        else:
            RHS = par.beta*(1+par.r_hh)*ss.C_R_a[a+1]**(-par.sigma)

        ss.C_R_a[a] = RHS**(-1/par.sigma)

    # b. find implied savings
    for a in range(par.A):

        if a == 0:
            B_R_a_lag = 0.0
        else: 
            B_R_a_lag = ss.B_R_a[a-1]
        
        ss.B_R_a[a] = (1+par.r_hh)/(1+ss.pi_hh)*B_R_a_lag + ss.inc_a[a] - ss.P_C*ss.C_R_a[a] 

    # c. aggregate
    ss.C_a = par.Lambda*ss.C_HtM_a+(1-par.Lambda)*ss.C_R_a
    ss.B_a = par.Lambda*ss.B_HtM_a+(1-par.Lambda)*ss.B_R_a
    
    ss.inc = np.sum(ss.inc_a)
    ss.C = np.sum(ss.C_a)
    ss.C_HtM = np.sum(ss.C_HtM_a)
    ss.C_R = np.sum(ss.C_R_a)
    ss.B = np.sum(ss.B_a)

    return ss.Bq-ss.B_a[-1]

def households_search_ss(par, ss):
    """ find labor supply in steady state """
    for a in range(par.A):
        if a == 0:
            ss.S_a[a] = 1.0
            ss.L_ubar_a[a] = 0.0
            
        elif a >= par.A_R:
            ss.S_a[a] = 0.0
            ss.L_ubar_a[a] = 0.0 
            
        else:
            ss.S_a[a] = (1-ss.L_a[a-1]) + par.delta_L_a[a]*ss.L_a[a-1]
            ss.L_ubar_a[a] = (1-par.delta_L_a[a])*ss.L_a[a-1]

        ss.L_a[a] = ss.L_ubar_a[a] + ss.m_s*ss.S_a[a]
        
    ss.S = np.sum(ss.S_a)
    ss.L_ubar = np.sum(ss.L_ubar_a)
    ss.L = np.sum(ss.L_a)
    
    ss.delta_L = (ss.L-ss.L_ubar)/ss.L
    ss.curlyM = ss.delta_L*ss.L
    ss.v = (ss.m_s**(1/par.sigma_m)*ss.S**(1/par.sigma_m)/(1-ss.m_s**(1/par.sigma_m)))**par.sigma_m
    ss.m_v = ss.curlyM/ss.v

def find_ss(par,ss,do_print=True):

    # a. price noramlizations
    ss.P_Y = 1.0
    ss.P_F = 1.0
    ss.P_M_C = 1.0
    ss.P_M_G = 1.0
    ss.P_M_I = 1.0
    ss.P_M_X = 1.0
    
    # b. fixed variables
    ss.pi_hh = par.pi_hh_ss  
    ss.m_s = par.m_s_ss   
    ss.B_G = par.B_G_ss
    ss.G = par.G_ss 
    
    # c. pricing in repacking firms
    ss.P_C = blocks.CES_P(ss.P_M_C,ss.P_Y,par.mu_M_C,par.sigma_C)
    ss.P_G = blocks.CES_P(ss.P_M_G,ss.P_Y,par.mu_M_G,par.sigma_G)
    ss.P_I = blocks.CES_P(ss.P_M_I,ss.P_Y,par.mu_M_I,par.sigma_I)
    ss.P_X = blocks.CES_P(ss.P_M_X,ss.P_Y,par.mu_M_X,par.sigma_X)

    # d. labor supply and search and matching
    for a in range(par.A):
        
        if a == 0:
            ss.S_a[a] = 1.0
            ss.L_ubar_a[a] = 0.0
        elif a >= par.A_R:
            ss.S_a[a] = 0.0
            ss.L_ubar_a[a] = 0.0            
        else:
            ss.S_a[a] = (1-ss.L_a[a-1]) + par.delta_L_a[a]*ss.L_a[a-1]
            ss.L_ubar_a[a] = (1-par.delta_L_a[a])*ss.L_a[a-1]

        ss.L_a[a] = ss.L_ubar_a[a] + ss.m_s*ss.S_a[a]

    ss.S = np.sum(ss.S_a)
    ss.L_ubar = np.sum(ss.L_ubar_a)
    ss.L = np.sum(ss.L_a)

    ss.delta_L = (ss.L-ss.L_ubar)/ss.L
    ss.curlyM = ss.delta_L*ss.L
    ss.v = (ss.m_s**(1/par.sigma_m)*ss.S**(1/par.sigma_m)/(1-ss.m_s**(1/par.sigma_m)))**par.sigma_m
    ss.m_v = ss.curlyM/ss.v

    if do_print:
        print(Fonttype.HEADER + 'Labor supply and search and matching:' + Fonttype.END)
        print(f'{ss.S = :.2f}' ',  ' f'{ss.L = :.2f}' ',  ' f'{ss.delta_L = :.2f}' ',  ' f'{ss.v = :.2f}' ',  ' f'{ss.m_v = :.2f}')

    # e. capital agency FOC
    ss.r_K = (par.r_firm + par.delta_K)*ss.P_I

    if do_print: 
        print(Fonttype.HEADER + 'Capital agency FOC:' + Fonttype.END)
        print(f'{ss.r_K = :.2f}')

    # f. production firm pricing
    ss.r_ell = ((1-par.mu_K*(ss.r_K)**(1-par.sigma_Y))/(1-par.mu_K))**(1/(1-par.sigma_Y))

    if do_print: 
        print(Fonttype.HEADER + 'Production firm pricing:' + Fonttype.END)
        print(f'{ss.r_ell = :.2f}')

    # g. labor agency
    ss.ell = ss.L - par.kappa_L*ss.v
    ss.W = ss.r_ell*(1-par.kappa_L/ss.m_v+(1-ss.delta_L)/(1+par.r_firm)*par.kappa_L/ss.m_v)

    if do_print: 
        print(Fonttype.HEADER + 'Labor agency:' + Fonttype.END)
        print(f'{ss.ell = :.2f}' ',  ' f'{ss.W = :.2f}')

    # h. government
    ss.tau = (par.r_b*ss.B_G+ss.P_G*ss.G)/(ss.W*ss.L)
    if do_print: 
        print(Fonttype.HEADER + 'Government:' + Fonttype.END)
        print(f'{ss.B_G = :.2f}' ',  ' f'{ss.G = :.2f}' ',  ' f'{ss.tau = :.2f}')

    # i. household behavior
    ss.real_W = ss.W/ss.P_C
    ss.real_r_hh = (1+par.r_firm)/(1+ss.pi_hh)-1
    
    if do_print: 
        print(Fonttype.HEADER + 'Households:' + Fonttype.END)
        print(f'solving for household behavior:',end='')

    result = optimize.root_scalar(household_ss,bracket=[0.001,1000],method='brentq',args=(par,ss,))
    if do_print: print(f' {result.converged = }')
    
    household_ss(result.root,par,ss)

    if do_print: 
        print(f'{ss.C = :.2f}' ',  ' f'{ss.B = :.2f}')
    
    # j. production firm FOCs
    ss.K = par.mu_K/(1-par.mu_K)*(ss.r_ell/ss.r_K)**par.sigma_Y*ss.ell

    if do_print: 
        print(Fonttype.HEADER + 'Production firm FOCs:' + Fonttype.END)
        print(f'{ss.K = :.2f}')

    # k. capital accumulation equation
    ss.iota = ss.I = par.delta_K*ss.K

    if do_print: 
        print(Fonttype.HEADER + 'Capital accumulation equation:' + Fonttype.END)
        print(f'{ss.I = :.2f}')

    # l. output in production firm
    ss.Y = blocks.CES_Y(ss.K,ss.ell,par.mu_K,par.sigma_Y)

    if do_print: 
        print(Fonttype.HEADER + 'Output in production firm:' + Fonttype.END)
        print(f'{ss.Y = :.2f}')

    # m. CES demand in packing firms
    ss.C_M = blocks.CES_demand(par.mu_M_C,ss.P_M_C,ss.P_C,ss.C,par.sigma_C)
    ss.C_Y = blocks.CES_demand(1-par.mu_M_C,ss.P_Y,ss.P_C,ss.C,par.sigma_C)

    ss.G_M = blocks.CES_demand(par.mu_M_G,ss.P_M_G,ss.P_G,ss.G,par.sigma_G)
    ss.G_Y = blocks.CES_demand(1-par.mu_M_G,ss.P_M_G,ss.P_G,ss.G,par.sigma_G)

    ss.I_M = blocks.CES_demand(par.mu_M_I,ss.P_M_I,ss.P_I,ss.I,par.sigma_I)
    ss.I_Y = blocks.CES_demand(1-par.mu_M_I,ss.P_Y,ss.P_I,ss.I,par.sigma_I)

    # n. market clearing
    ss.X_Y = ss.Y - (ss.C_Y + ss.G_Y + ss.I_Y) 
    ss.chi = ss.X_Y/(1-par.mu_M_X)
    ss.X = ss.X_Y/(1-par.mu_M_X)
    ss.X_M = blocks.CES_demand(par.mu_M_X,ss.P_M_X,ss.P_X,ss.X,par.sigma_X)
    
    ss.M = ss.C_M + ss.G_M + ss.I_M + ss.X_M

    if do_print: 
        print(Fonttype.HEADER + 'Market clearing:' + Fonttype.END)
        print(f'{ss.C_Y = :.2f}' ',  ' f'{ss.G_Y = :.2f}' ',  ' f'{ss.I_Y = :.2f}' ',  ' f'{ss.X_Y = :.2f}')
        print('[ ' f'{ss.C_M = :.2f}' ',  ' f'{ss.G_M = :.2f}' ',  ' f'{ss.I_M = :.2f}' ',  ' f'{ss.X_M = :.2f}' ' ] = ' f'{ss.M = :.2f}')
        print(f'{ss.X = :.2f}')

    # n. bargaining
    ss.W_ast = ss.W
    ss.MPL = ((1-par.mu_K)*ss.Y/ss.ell)**(1/par.sigma_Y)
    ss.W_bar = ss.P_Y*ss.MPL
    par.phi = (ss.W_ast-par.W_U)/(ss.W_bar-par.W_U)
    if do_print: 
        print(Fonttype.HEADER + 'Bargaining:' + Fonttype.END)
        print(f'{par.phi = :.3f}')