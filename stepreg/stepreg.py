from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt

def EarlyStopping(lr_results,patience,tol,verbosity):
    """
    Early Stopping for variable selection
    
    lr_results: pandas data frame
        including results from function Logitstep
        
    patience: int
        the number of rounds without improvement after which training will be early stopped
        
    tol: float
        minimum improvement in the set numbers of rounds
        
    verbosity: int 
        verbosity level: If <= 1, no progress messages are printed, else if >1 all progress messages are printed.        
        
    """
    
    # exclude NaN
    lr_results_ex_nan=lr_results[~lr_results["AUC"].isnull()]

    i=0
    pointer=1
    rounds=0
    while rounds <= patience:
        step=lr_results_ex_nan["step"][i]
        if verbosity > 1:
            print("Number of variables is "+str(step))
            print("rounds: ",rounds)

        if i+pointer == len(lr_results_ex_nan):
            step=lr_results_ex_nan["step"][i]
            if verbosity > 1:
                print("Break, maximal number of variables reached.")
            break
        else:
            if lr_results_ex_nan.iloc[i+pointer]["AUC"]-lr_results_ex_nan.iloc[i]["AUC"]> tol:
                rounds=0
                step=lr_results_ex_nan["step"][i+pointer]
                pointer=1
                i += 1
                if verbosity > 1:
                    print("Improvement > ",tol,"between ",patience, " rounds." )
            else:
                rounds = lr_results_ex_nan.iloc[i+pointer]["step"]-lr_results_ex_nan.iloc[i]["step"]
                pointer +=1
  
    return(step)

def Logitstep(y,X,maxsteps=30,p=0.05,force=["intercept"],plot=True,early_stopping=False,patience=10,tol=0.001,verbosity=1):
    """
    Logitstep fits a logistic regression model from statsmodel api in a forward stepwise fashion.
    By default an intercept is forced into the model. One can choose the significance level by p 
    and the number of maximal included variables by maxsteps.  
    
    Parameters
    ----------

    y : pandas series
        Endogenous response variable.
        
    X : pandas data frame
        Explanatory variables / features. An intercept is added by default.
              
    maxsteps: integer (1-n)
        Number of maximal included variables in model. 
        
    p : float64
        Accepted significant level to include variables in the selection process.
        
    force : list
        Variables to force in the model. Default is 'intercept'. If you do not specify 'intercept' in force
        no intercept is used.
        
    plot : boolean
        plot visualisation of forward stepwise selection 
          
    early_stopping: boolean
        early stopping for final variable selection 
        
    patience: int 
        the number of rounds without improvement after which training will be early stopped
    
    tol: float
        minimum improvement in the set numbers of rounds
        
    verbosity: int (0 or 1)
        verbosity level: If 0, no progress messages are printed, else if 1 progress messages for stepwise selection 
        are printed, else if 2 all messages are printed.
       
    Return
    ------
    
    1. pandas data frame including results
    
    2. statsmodels.discrete.discrete_model.BinaryResultsWrapper
    
        Attributes
        ----------
        Analogue to Logit Model in statsmodel.
        
    3. final variable list
        
     """
    
    warnings.simplefilter('ignore', ConvergenceWarning)

    # add intercept
    X=X.assign(intercept = 1)
    
    # initialize variable list
    varlist=list(X.columns)
    varlist.remove("intercept")

    # initialize variables choosen in stepwise selection
    steplist=force
    startpoint=len(steplist)
    pointer=startpoint
    
    # initialize AUC per step
    AUC_per_step=[float("nan")]*len(force)
    
    while len(varlist)>0 and (len(steplist)-startpoint) <= (maxsteps-1):
        
        # initializing next step
        results=[]
        logit_model_temp={}
        
        # calculate zvalues for every variable in varlist
        for j in range(len(varlist)):

            # update temporary steplist
            steplist_temp=steplist+[varlist[j]]

            # model estimation
            try:
                logit_model_temp[j]=sm.Logit(y,X[steplist_temp]).fit(method='newton',disp=0)

                # calculate results
                diag = np.diag(logit_model_temp[j].cov_params())
                if min(diag) > 0:
                    z_values = abs(logit_model_temp[j].params / np.sqrt(np.diag(logit_model_temp[j].cov_params())))
                    if logit_model_temp[j].pvalues[pointer] <= p:
                        results.append([pointer,varlist[j],logit_model_temp[j].pvalues[pointer],z_values[pointer],j]) 
                else:
                    if verbosity==2:
                        print("diagonal of covariance has negative values, the following variable will be ignored:",varlist[j])

            except np.linalg.LinAlgError as err:
                if 'Singular matrix' in str(err):
                    if verbosity==2:
                        print("Singular matrix, the following variable will be ignored:",varlist[j])
                else:
                    raise

        if len(results)>0:
            # choose variable with highest zvalue. If variables have the same zvalue the variable is chosen in alphabetical order.  
            results_df = pd.DataFrame(results,columns=["step","variable","pvalue","zvalue","modelindex"]).sort_values(["zvalue","variable"], axis=0, ascending=[False,True])

            # set final model
            logit_model = logit_model_temp[results_df.iloc[0,4]]       
            
            # update varlist and steplist
            steplist.append(results_df.iloc[0,1])
            varlist.remove(results_df.iloc[0,1])
            if verbosity == 1 or verbosity == 2:
                print("variable",results_df.iloc[0,1], "entered in step",pointer-startpoint+1)
            
            # calculate discriminatory power
            prob = logit_model.predict(X[steplist])
            AUC_per_step.append(roc_auc_score(y, prob))
            
        else:
            varlist=[]
            if verbosity == 1 or verbosity == 2:
                print("none of the remaining variables are significant")

        pointer += 1    

    if 'logit_model' in locals():
        
        # to DataFrame
        logit_model_as_html = logit_model.summary().tables[1].as_html()
        logit_results = pd.read_html(logit_model_as_html, header=0, index_col=0)[0]

        # prepare results
        logit_results["lower conf"] = logit_results.iloc[:,4]
        logit_results["upper conf"] = logit_results.iloc[:,5]
        logit_results["step"] = list(range(logit_results.shape[0]))
        logit_results["AUC"] = AUC_per_step       

        logit_results=logit_results[["step","coef","std err", "z","P>|z|","lower conf","upper conf","AUC"]]
              
        # Early Stopping  
        steplist_orig=steplist
        if early_stopping:
            best_step=EarlyStopping(logit_results,patience,tol,verbosity)
            steplist=steplist_orig[0:(startpoint+best_step)]
            
        # Output
        print("Final variable list:",steplist)
        print(logit_model.summary())      
        
        if plot:
            # plt.xkcd()
            AUC=logit_results.iloc[:,7]
            plt.rcParams['axes.facecolor'] = 'whitesmoke'
            fig,ax = plt.subplots()
            ax.plot(steplist_orig, AUC,zorder=5)
            ax.scatter(steplist_orig, AUC,zorder=10)
            ax.grid(linestyle="-", linewidth=1, color='white')
            plt.ylabel('AUC',fontsize=10)
            plt.xlabel('variable',fontsize=10)
            plt.xticks(steplist_orig,fontsize=10,rotation = 90)
            plt.yticks(fontsize=10)
            if early_stopping:
                plt.axvline(x=best_step,linestyle="--",color='coral')
            plt.title('forward stepwise selection',fontsize=16)
            plt.savefig("D:/Projekte/03_sonstiges/Python/AUC_per_step.pdf",bbox_inches='tight', dpi=1000)
        
        return(logit_results,logit_model,steplist)
        
    else:
        logit_model="no significant variable in the data"
        logit_results=logit_model
        print(logit_model)
        
        return(logit_results,logit_model,steplist)