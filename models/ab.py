import numpy as np
from dtw import *

import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV, LarsCV, Lasso, Ridge, BayesianRidge, LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA

from sklearn.pipeline import make_pipeline

from sklearn.metrics import mean_absolute_percentage_error

class AnalogBivariate():
    '''
    X_ref: Serie de referencia (temp)
    data: Serie donde se buscarán los más semejantes a X_ref (temp)
    self.Y: Time series to forecast (load)
    k: Number of neighbours to search
    tol: Window size tolerance for neighbour selection
    dist: distance measure, 'euclidian' or 'pearson' or 'dtw' / medida de distancia, 'euclidian' o 'pearson' o 'dtw' 
    '''
    def __init__ (self, data, X_ref, Y, k = 10, tol = 0.8, n_components = 3, dist = 'pearson', reg_model = 'OLSstep', verbose = False):
        self.data = data
        self.X_ref = X_ref
        self.Y = Y
        self.k = k
        self.tol = tol
        self.n_components = n_components
        self.dist = dist
        self.reg_model = reg_model
        self.verbose = verbose

        distances = []
        for i in range(0, len(self.data) - len(self.X_ref)): # No hay esa restricción :o
            if  dist == 'dtw':     ## dynamic time warping
                dist = dtw(self.X_ref, self.data[i, i + len(self.X_ref)]).distance
            elif dist == 'euclidian':
                dist = euclidean(self.X_ref, self.data[i : i + len(self.X_ref)])
            else:
                dist = np.corrcoef(self.X_ref, self.data[i : i + len(self.X_ref)])[1,0]
            distances.append((i, dist))

        ## We calculate the neighbourhood by distance from smallest to largest and the positions are saved.
        if dist == 'pearson':
            ## In the Pearson backwards case, we are interested in the indices with the highest correlation in Pearson backwards ordering.
            distances.sort(key=lambda tup: tup[1], reverse=True)
        else:
            distances.sort(key=lambda tup: tup[1], reverse=False)

        X = []
        X1 = []
        positions  = []

        ## We calculate the k nearest neighbors and save the positions.
        for pos, dis in distances:
            if len(positions) == 0:
                positions.append(pos)
                X.append(self.data[pos:pos + len(self.X_ref)])
                X1.append(self.data[pos + len(self.X_ref) : pos + 2 * len(self.X_ref)])
            else:
                for p in positions:
                    ## if we already had a position in the list that passed the tolerance, we no longer save it
                    if (abs(pos - p) < tol * len(self.X_ref)):
                        break
                    else:
                        ## save new neighbor
                        positions.append(pos)
                        X.append(self.data[pos:pos + len(self.X_ref)])
                        X1.append(self.data[pos + len(self.X_ref) : pos + 2 * len(self.X_ref)])
                if len(positions) >= k:
                    break
        if self.verbose == True:
            print('positions KNN:', positions) ## position of k nearest neighbors

        self.X  = np.array(X).T[:, 0:k]
        self.X1 = np.array(X1).T[:, 0:k]
        
    def fit(self):
        match self.reg_model:
            case 'RF':
                self.model = RandomForestRegressor(random_state=42)
            case 'OLSstep':
                model = sm.OLS(self.Y, self.X)
                results = model.fit()

                ## We sort the 'pi' values and the largest one is selected.
                i = 0
                pvalues = []
                for pi in results.pvalues:
                    pvalues.append((i, pi))
                    i = i + 1
                pvalues.sort(key=lambda tup: tup[1], reverse=True) ## We order by 'pi'
                (i, pi) = pvalues[0]  

                while pi > pi:
                    X   = sm.add_constant(self.X)
                    X_2 = sm.add_constant(self.X1)   
                    if self.verbose == True:
                        print('Retiramos regresor ---> X' + str(i))
                    X   = np.delete(arr=X,   obj=i+0, axis=1)
                    X_2 = np.delete(arr=X_2, obj=i+0, axis=1)   
                    model   = sm.OLS(Y, X)
                    self.results = model.fit()

                    ## We sort the 'pi' values and select the largest
                    i = 0
                    pvalues = []
                    for pi in results.pvalues:
                        pvalues.append((i,pi))
                        i = i + 1
                    pvalues.sort(key=lambda tup: tup[1], reverse=True) ## We order by 'pi'
                    (i, pi) = pvalues[0]
                    prediction_Y2 = results.predict(X_2)
                if len(prediction_Y2) == 0:      
                    if self.verbose == True:
                        print('>>> Warning, no variable was significant in the regression.')
                    model   = sm.OLS(Y, X)
                    results = model.fit()
                if self.verbose == True:
                    print(results.summary())
            case 'Boosting':
                self.model = GradientBoostingRegressor(random_state=42)
            case 'Bagging':
                self.model = BaggingRegressor(random_state=42,)
            case 'LinearReg':
                self.model = LinearRegression()
            case 'AdaBoost':
                self.model = AdaBoostRegressor(random_state=42)
            case 'BayesRidge':
                self.model = BayesianRidge(compute_score=True) # Revisar
            case 'LassoReg':
                self.model = Lasso(alpha=0.1)
            case 'RidgeReg':
                self.model = Ridge(alpha=0.1)
            case 'PLS':
                self.model = PLSRegression(n_components = self.n_components)
            case 'PCR':
                # https://scikit-learn.org/stable/auto_examples/cross_decomposition/plot_pcr_vs_pls.html
                self.model = make_pipeline(PCA(n_components = self.n_components), LinearRegression())
            case 'VotingEnsemble':
                # https://scikit-learn.org/stable/auto_examples/ensemble/plot_voting_regressor.html#sphx-glr-auto-examples-ensemble-plot-voting-regressor-py
                gb  = GradientBoostingRegressor(random_state=42)
                rf  = RandomForestRegressor(random_state=42)
                br  = BaggingRegressor(random_state=42)
                ab  = AdaBoostRegressor(random_state=42)
                gb.fit(self.X, self.Y)
                rf.fit(self.X, self.Y)
                br.fit(self.X, self.Y)
                ab.fit(self.X, self.Y)
                self.model = VotingRegressor([("gb",gb), ("rf",rf), ("br",br), ("ab",ab)])
            case 'VotingLinear':
                # https://scikit-learn.org/stable/auto_examples/ensemble/plot_voting_regressor.html#sphx-glr-auto-examples-ensemble-plot-voting-regressor-py
                pl = PLSRegression(n_components=self.n_components)
                lr = LinearRegression()
                ri = Ridge(alpha=0.1)
                la = Lasso(alpha=0.1)    
                pc = make_pipeline(PCA(n_components=self.n_components), LinearRegression())
                pl.fit(self.X, self.Y)
                lr.fit(self.X, self.Y)
                ri.fit(self.X, self.Y)
                la.fit(self.X, self.Y)
                pc.fit(self.X, self.Y)
                self.model = VotingRegressor([("lr",lr),("ri",ri),("la",la),("pc",pc)])
        results = self.model.fit(self.X, self.Y)
    
    def predict(self):
        self.Y_hat = self.model.predict(self.X1)
        return self.Y_hat
    
    def score(self):
        return mean_absolute_percentage_error(self.Y, self.Y_hat)