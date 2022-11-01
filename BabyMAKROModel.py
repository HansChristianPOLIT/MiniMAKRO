from multiprocessing.sharedctypes import Value
import time
import numpy as np

from EconModel import EconModelClass, jit
from consav import elapsed

import matplotlib.pyplot as plt   
plt.style.use('seaborn-whitegrid')
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# local
import blocks
import steady_state
from broyden_solver import broyden_solver

class BabyMAKROModelClass(EconModelClass):    

    # This is the BabyMAKROModelClass
    # It builds on the EconModelClass -> read the documentation

    # in .settings() you must specify some variable lists
    # in .setup() you choose parameters
    # in .allocate() all variables are automatically allocated

    def settings(self):
        """ fundamental settings """

        # a. namespaces
        self.namespaces = ['par','ss','ini','sol']

        # b. blocks
        self.blocks = [
            'household_search',
            'labor_agency',
            'production_firm',
            'bargaining',
            'repacking_firms_prices',
            'foreign_economy',
            'capital_agency',
            'government',
            'households_consumption',
            'repacking_firms_components',
            'goods_market_clearing',
        ]
        
        # c. variable lists
        
        # exogenous variables
        self.exo = [
            'chi',
            'P_F',
            'P_M_C',
            'P_M_G',
            'P_M_I',
            'P_M_X',
            'G',
        ]
        
        # unknowns
        self.unknowns = [
            'Bq',   #Inheritance flow
            'K',   #Capital
            'L',   #Labour supply
            'r_K',  #Rental price of capital
            'W',   #wage
        ]

        # targets
        self.targets = [
            'bargaining_cond',
            'Bq_match',
            'FOC_capital_agency',
            'FOC_K_ell',
            'mkt_clearing',
        ]

        # all non-household variables
        self.varlist = [
            'B',    #End-of-period saving
            'B_G',   #government debt
            'bargaining_cond',
            'Bq_match',
            'Bq',
            'C_HtM', #aggregated HtM consumption
            'C_M',   #Imported consumption components
            'C_Y',   #Output consumption good
            'C',   #Aggregate consumption
            'C_R', #aggregated Ricardian HH consumption
            'chi',
            'curlyM',
            'delta_L',
            'ell',
            'FOC_C',
            'FOC_capital_agency',
            'FOC_K_ell',
            'G',
            'G_M',
            'G_Y',
            'I_M',
            'I_Y',
            'I',
            'iota',
            'inc',
            'K',
            'L_ubar',
            'L',
            'm_s',
            'm_v',
            'M',
            'mkt_clearing',
            'MPL',
            'P_C',
            'P_G',
            'P_F',
            'P_I',
            'P_M_C',
            'P_M_G',
            'P_M_I',
            'P_M_X',
            'P_X',
            'P_Y',
            'pi_hh',
            'r_ell',
            'real_r_hh',
            'real_W',
            'r_K',
            'S',
            'tau',
            'v',
            'W_ast',
            'W',
            'W_bar',
            'X_M',
            'X_Y',
            'X',
            'Y',
        ]

        # all household variables
        self.varlist_hh = [
            'B_a',
            'B_HtM_a',
            'B_R_a',
            'C_a',
            'C_HtM_a',
            'C_R_a',
            'inc_a',
            'L_a',
            'L_ubar_a',
            'S_a',
        ]

    def setup(self):
        """ set baseline parameters """

        par = self.par

        par.T = 500 # number of time-periods
        
        # a. households
        par.A = 80 # life-span
        par.A_R = 60 # work-life-span
        par.beta = 0.95 # discount factor
        par.sigma = 1.25 # CRRA coefficient
        par.sigma_m = 0.98 #CRRA coefficient from matching function
        par.mu_B = 2.5 # weight on bequest motive
        par.r_hh = 0.04 # nominal return rate                               - Note: Meget afgørende for resultaterne (ved 0.08 er der ingen løsnings)
        par.delta_L_a = 0.02*np.ones(par.A_R) # separation probabilities    - Note: Umiddelbart mindre sensitiv efter seneste ændre i labor agency
        par.W_U = 0.8 # outside option in bargaining                       - Note: Hvorfor er outside-option meget lavere end w_ss? Burde de et eller andet sted ikke ligge tæt på hinanden?
        par.Lambda = 0.2 # Share of hands-to-mouth households

        # b. production firm
        par.r_firm = 0.04 # internal rate of return
        par.delta_K = 0.10 # depreciation rate
        par.mu_K = 1/3 # weigth on capital
        par.sigma_Y = 1.01 # substitution

        # c. labor agency
        par.kappa_L = 0.05 # cost of vacancies in labor units

        # d. capital agency
        par.Psi_0 = 4.0 # adjustment costs

        # e. government
        par.r_b = 0.03 # rate of return on government debt
        par.lambda_B = 0.5 # rigidity in taxes
        par.delta_B = 5 # number of adjustment years
        par.epsilon_B = 0.2 #   

        # e. repacking
        par.mu_M_C = 0.30 # weight on imports in C
        par.sigma_C = 1.5 # substitution
        par.mu_M_G = 0.10 # weight on imports in G
        par.sigma_G = 1.5 # substitution
        par.mu_M_I = 0.35 # weight on imports in I
        par.sigma_I = 1.5 # substitution
        par.mu_M_X = 0.40 # weight on imports in X
        par.sigma_X = 1.5 # substitution
        
        # f. foreign
        par.sigma_F = 5.0 # substitution in export demand
        par.lambda_X = 0.8 # rigidity in export demand

        # g. matching
        par.sigma_m = 1.5 # curvature

        # h. bargaining
        par.gamma_w = 0.80 # wage persistence
        par.phi = np.nan # bargaining power of firms (determined when finding steady state)

        # j. Steady State
        par.pi_hh_ss = 0.0  #Set inflation in steady state to 0
        par.m_s_ss = 0.50   #Set the job finding rate in steady state to 0.5
        par.B_G_ss = 100.0
        par.G_ss = 61.82
        par.W_ss = 1.0 # wage
        
    def allocate(self):
        """ allocate model """

        par = self.par
        ini = self.ini
        ss = self.ss
        sol = self.sol

        # a. non-household variables
        for varname in self.varlist:
            setattr(ini,varname,np.nan)
            setattr(ss,varname,np.nan)
            setattr(sol,varname,np.zeros(par.T))

        for varname in self.exo: assert varname in self.varlist, varname

        # b. household variables
        for varname in self.varlist_hh:
            setattr(ini,varname,np.zeros(par.A))
            setattr(ss,varname,np.zeros(par.A))
            setattr(sol,varname,np.zeros((par.A,par.T)))            

        for varname in self.unknowns: assert varname in self.varlist+self.varlist_hh, varname
        for varname in self.targets: assert varname in self.varlist+self.varlist_hh, varname

    ################
    # steady state #
    ################
    
    def find_ss(self,do_print=False):
        """ find steady state """

        steady_state.find_ss(self.par,self.ss,do_print=do_print)

    #################
    # set functions #
    #################

    # functions for setting and getting variables
    
    def set_ss(self,varlist):
        """ set variables in varlist to steady state """

        par = self.par
        sol = self.sol
        ss = self.ss

        for varname in varlist:

            ssvalue = ss.__dict__[varname]

            if varname in self.varlist:
                sol.__dict__[varname] = np.repeat(ssvalue,par.T)
            elif varname in self.varlist_hh:
                sol.__dict__[varname] = np.zeros((par.A,par.T))
                for t in range(par.T):
                    sol.__dict__[varname][:,t] = ssvalue
            else:
                raise ValueError(f'unknown variable name, {varname}')

    def set_exo_ss(self):
        """ set exogenous variables to steady state """

        self.set_ss(self.exo)

    def set_unknowns_ss(self):
        """ set unknowns to steady state """

        self.set_ss(self.unknowns)

    def set_unknowns(self,x):
        """ set unknowns """

        sol = self.sol

        i = 0
        for unknown in self.unknowns:
            n = sol.__dict__[unknown].size
            sol.__dict__[unknown].ravel()[:] = x[i:i+n]
            i += n
    
    def get_errors(self,do_print=False):
        """ get errors in target equations """

        sol = self.sol

        errors = np.array([])
        for target in self.targets:

            errors_ = sol.__dict__[target]
            errors = np.hstack([errors,errors_.ravel()])

            if do_print: print(f'{target:20s}: abs. max = {np.abs(errors_).max():8.2e}')

        return errors

    ############
    # evaluate #
    ############

    def evaluate_block(self,block,py=False):

        with jit(self) as model: # use jit for faster evaluation

            if not hasattr(blocks,block): raise ValueError(f'{block} is not a valid block')
            func = getattr(blocks,block)

            if py: # python version for debugging
                func.py_func(model.par,model.ini,model.ss,model.sol)
            else:
                func(model.par,model.ini,model.ss,model.sol)
    
    def evaluate_blocks(self,ini=None,do_print=False,py=False):
        """ evaluate all blocks """

        # a. initial conditions
        if ini is None: # initial conditions are from steady state
            for varname in self.varlist: self.ini.__dict__[varname] = self.ss.__dict__[varname] 
            for varname in self.varlist_hh: self.ini.__dict__[varname] = self.ss.__dict__[varname].copy() 
        else: # initial conditions are user determined
            for varname in self.varlist: self.ini.__dict__[varname] = ini.__dict__[varname] 
            for varname in self.varlist_hh: self.ini.__dict__[varname] = ini.__dict__[varname].copy() 

        # b. evaluate
        with jit(self) as model: # use jit for faster evaluation

            for block in self.blocks:
                
                if not hasattr(blocks,block): raise ValueError(f'{block} is not a valid block')
                func = getattr(blocks,block)

                if py: # python version for debugging
                    func.py_func(model.par,model.ini,model.ss,model.sol)
                else:
                    func(model.par,model.ini,model.ss,model.sol)

                if do_print: print(f'{block} evaluated')
    
    ########
    # IRFs #
    ########
    
    def calc_jac(self,do_print=False,dx=1e-4):
        """ calculate Jacobian arround steady state """

        t0 = time.time()

        sol = self.sol

        # a. baseline
        self.set_exo_ss()
        self.set_unknowns_ss()
        self.evaluate_blocks()

        base = self.get_errors()

        x_ss = np.array([])
        for unknown in self.unknowns:
            x_ss = np.hstack([x_ss,sol.__dict__[unknown].ravel()])

        # b. allocate
        jac = self.jac = np.zeros((x_ss.size,x_ss.size))

        # c. calculate
        for i in range(x_ss.size):
            
            x = x_ss.copy()
            x[i] += dx

            self.set_unknowns(x)
            self.evaluate_blocks()
            alt = self.get_errors()
            jac[:,i] = (alt-base)/dx

        if do_print: print(f'Jacobian calculated in {elapsed(t0)} secs')

    def find_IRF(self,ini=None):
        """ find IRF """

        sol = self.sol

        # a. set initial guess
        self.set_unknowns_ss()

        x0 = np.array([])
        for unknown in self.unknowns:
            x0 = np.hstack([x0,sol.__dict__[unknown].ravel()])

        # b. objective
        def obj(x):
            
            # i. set unknowns from x
            self.set_unknowns(x)

            # ii. evaluate
            self.evaluate_blocks(ini=ini)

            # iii. get and return errors
            return self.get_errors()

        # c. solver
        broyden_solver(obj,x0,self.jac,tol=1e-10,maxiter=100,do_print=True,model=self)

    ###########
    # figures #
    ###########

    def get_key(self,varname):
        """ fetch name associated with variable """
        variabel_name = {"Export ($X$)": "X", "GDP ($Y$)":"Y", "Consumption ($C$)":"C","Imports ($M$)":"M", "Government Spending ($G$)":"G", "Investments ($I$)":"I", "Capital ($K$)":"K", "Age Specific Searchers ($S_a$)":"S_a", "Aggregated Searchers ($S$)":"S","Age Specific Income ($inc_a$)":"inc_a", "Age Specific Consumption ($C_a$)":"C_a", "Bequest ($B_q$)":"Bq", "Tax Rate ($\tau$)": "tau", "Nominal Wage ($W$)": "W", "Real Wage ($w$)": "real_W", "Rental Price of Capital ($r_K$)":"r_K", "Labor Supply ($L$)": "L", "Public Debt ($B_G$)": "B_G", "End-of-Period Savings ($B$)": "B", "Income ($inc$)": "inc", "Vacancies ($v$)":"v", "Vancancy Filling Rate ($m_v$)":"m_v", "Job Finding Rate ($m_s$)": "m_s"}
        
        for key, value in variabel_name.items():
            if value == varname:
                 return key
    
    def plot_IRF(self,varlist=[],ncol=3,T_IRF=50,abs=[],Y_share=[]):
        """ plot IRFs """
        
        ss = self.ss
        sol = self.sol

        nrow = len(varlist)//ncol
        if len(varlist) > nrow*ncol: nrow+=1

        fig = plt.figure(figsize=(ncol*6,nrow*6/1.5))
        for i,varname in enumerate(varlist):

            ax = fig.add_subplot(nrow,ncol,1+i)

            path = sol.__dict__[varname]
            ssvalue = ss.__dict__[varname]

            if varname in abs:
                ax.axhline(ssvalue,color='black')
                ax.plot(path[:T_IRF],'-o',markersize=3)
            elif varname in Y_share:
                ax.plot(path[:T_IRF]/sol.Y[:T_IRF],'-o',markersize=3)
                ax.set_ylabel('share of Y', size = 14)         
            elif np.isclose(ssvalue,0.0):
                ax.plot(path[:T_IRF]-ssvalue,'-o',markersize=3)
                ax.axhline(y=0, color ='dimgrey')
                ax.set_ylabel('diff.to ss', size = 14)
            else:
                ax.plot((path[:T_IRF]/ssvalue-1)*100,'-o',markersize=3)
                ax.axhline(y=0, color ='dimgrey')
                ax.set_ylabel('$\%$ diff.to ss', size = 14)
            
            ax.set_title(f'{self.get_key(varname)} {varname}', size = 18)
            ax.set_xlabel('Years', size = 18)

        fig.tight_layout(pad=1.0)
        
    def plot_IRF_hh(self,varlist,t0_list=None,ncol=2):
        """ plot IRFs for household variables """

        par = self.par
        ss = self.ss
        sol = self.sol

        if t0_list is None: t0_list = [-par.A+1,0,par.A]

        nrow = len(varlist)//ncol
        if len(varlist) > nrow*ncol: nrow+=1

        fig = plt.figure(figsize=(ncol*6,nrow*6/1.5))

        for i,varname in enumerate(varlist):

            ax = fig.add_subplot(nrow,ncol,1+i)

            for t0 in t0_list:

                t_beg = np.fmax(t0,0)
                t_end = t0 + par.A-1

                y = np.zeros(t_end-t_beg)
                for j,t in enumerate(range(t_beg,t_end)):
                    a = t-t0
                    y[j] = sol.__dict__[varname][a,t]-ss.__dict__[varname][a]
                    
                ax.plot(np.arange(t_beg-t0,t_end-t0),y,label=f'$t_0$ = {t0}')
                ax.set_xlabel('Age', size = 18)
                ax.set_ylabel('diff to ss')
                ax.set_title(f'{varname}')

            if i == 0:
                ax.legend(frameon=True)

        fig.tight_layout(pad=1.0)