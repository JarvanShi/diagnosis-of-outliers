import numpy as np
from numpy.linalg import inv, cholesky
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2, kstest, probplot, ks_2samp
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

import os
os.environ['R_HOME'] = 'D:\R\R-3.3.3'
os.environ['R_USER'] = 'D:\\Anaconda\\Lib\\site-packages\\rpy2'
from rpy2 import robjects
from rpy2.robjects.packages import importr

plt.rc('figure', figsize=(10, 6))

# Montecarlo Simulation
class Montecarlo:
    """
    =====================
    Montecarlo Simulation
    =====================

    ================= =====================================================================================================
    Utility Functions
    ================= =====================================================================================================
    gene_samples      generate n-by-k samples through a model which error term obbey multivariate normalization distribution
    gene_mrn          generate n-by-k samples which obbey multivatiate normalization distribution
    ================= =====================================================================================================
    """

    __slots__ = 'mu', 'sigma', 'B', 'sample_n', 'vec_dimension', 'l'

    def __init__(self, mu, sigma, coef):
        self.mu = mu                                                #  means of k variables
        self.sigma = sigma                                          #  variance of k variable
        self.B = coef                                               #  model's coefficient
        self.sample_n = int(input("enter the number of samples:"))  #  number of samples
        self.vec_dimension = len(mu)                                #  dimension of vector
        self.l = np.ones(1)                                         #  the independent variable's value corresponding to the constant term

    # generate samples obbey multivariate normalization distribution
    def gene_samples(self):
        p =  (self.B.shape[1]-1)//self.vec_dimension                           # lags
        mrn = self.gene_mrn()
        x = np.zeros((self.sample_n, self.vec_dimension))                      # initial samples matrix
        x_patial = np.concatenate((self.l, x[-p:].ravel()), axis=0)            # concatenate constant term and independent variable

        for i in range(self.sample_n-p):
            new_x = np.dot(self.B, x_patial) + mrn[-(p+i+1)]                   # k-by-1 vector
            x[-(p+i+1)] = new_x                                                # update samples
            x_patial[1:] = x[-(p+i+1):-(i+1)].ravel()                          # get new independent variable

        columns = []
        for i in range(self.vec_dimension):
            columns.append('seq' + str(i+1))

        data = pd.DataFrame(x[:self.sample_n-p], columns=columns)
        date = pd.date_range('19260101', periods=self.sample_n-p, freq='M')
        date = pd.DataFrame(date, columns=['date'])
        samples = pd.concat((date, data), axis=1, ignore_index=False)
        return samples                                                         # pandas dataframe

    # generate multivariate normalization distribution
    def gene_mrn(self):
        #  1. by numpy.random.multivariate_normal
        mrn = np.random.multivariate_normal(self.mu, self.sigma, self.sample_n)

        #  2. by cholesky decomposition
        # R = cholesky(self.sigma)  # cholesky decomposition
        # mrn = np.dot(np.random.randn(self.sample_n, self.vec_dimension), R) + self.mu  # generate multivatiate normalization distribution
        return mrn  #  ndarray

# Score Test
class ScoreTest:
    """
    ==========
    Score Test
    ==========

    =================== =============================================================
    Utility functions
    =================== =============================================================
    dup_matrix          duplicaiton matrix, used in calculate score statistics
    gene_indep_variable generate the whole independence variable matrix
    mean_shift          calculate the score statistics by mean shift model
    variance_weight     calculate the score statistics by variance weight model
    plot                plot the samples' score and critical
    =================== =============================================================

    ======================= =======================================================
    Attribution description
    ======================= =======================================================
    samples                 pandas DataFrame, the first column must be date
    ======================= =======================================================
    """

    __slots__ = ('whole_samples', 'date', 'samples', 'whole_samples_n', 'sigma_inv', 'B', 'k', 'p', 'samples_n',
                 'l', 'z', 'dup_mat', 'I22', 'ms_critical', 'vw_critical', 'ms_scores', 'vw_scores')

    def __init__(self, samples, sigma, coefficient):
        self.whole_samples = samples
        self.date = samples.iloc[:, 0]                            # samples' date
        self.samples = samples.iloc[:, 1:]                        # samples, pandas type, date order is ascending
        self.whole_samples_n = len(samples)                       # number of whole samples
        self.sigma_inv = inv(sigma)                               # inverse of variance
        self.B = coefficient                                      # coefficient matrix
        [self.k, m] = self.B.shape                                # dimension of interesting parameters
        self.p = (m-1)//self.k                                    # lags
        self.samples_n = self.whole_samples_n - self.p            # number of real samples
        self.l = np.ones(1)                                       # the independent variable value corresponding to the constant term
        self.z = self.gene_indep_variable()
        self.dup_mat = self.dup_matrix()
        self.I22 = self.calc_I22()
        self.ms_critical = chi2.isf(0.05/self.samples_n, self.k)  # critirion of mean shift model
        self.vw_critical = chi2.isf(0.05/self.samples_n, 1)       # critirion of variance weight model
        self.ms_scores = []                                       # scores based on mean shift model
        self.vw_scores = []                                       # scores based on variance weight model

    # generate duplication matrix
    def dup_matrix(self):
        """
        Your machine must have installed R3.3 and matrixcalc library
        """
        maxtrixcalc = importr('matrixcalc')                # load matrixcalc library
        rscript = 'D.matrix( )'.replace(' ', str(self.k))  # generate R script
        rmatrix = robjects.r(rscript)                      # run R script
        dup_mat = np.array(rmatrix)                        # convert to ndarray
        return dup_mat                                     # ndarray

    # generate independencd variable matrix
    def gene_indep_variable(self):
        n, p, k = self.whole_samples_n, self.p, self.k
        sample = np.array(self.samples.values)             # in order to use numpy functions, extract values
        z = np.zeros((n-p, k*p+1))                         # initial independent variable matrix
        for i in range(n-p):
            temp = np.concatenate((self.l, np.array(list(reversed(sample[i:i+p]))).ravel()), axis=0)
            z[i] = temp
        return z                                           # ndarray

    def calc_I22(self):
        z = self.z
        k = self.k
        p = self.p
        n = self.whole_samples_n
        sigma_inv = self.sigma_inv
        dup_mat = self.dup_mat
        dup_transpose = dup_mat.T
        I22 = np.zeros((((1+2*p)*k*k+3*k)//2, ((1+2*p)*k*k+3*k)//2))                           # initial I22
        I221 = np.kron(np.dot(z.T, z), sigma_inv)                                               # calculate the upleft block of information matrix
        I222 = (n-p)/2 * np.dot(np.dot(dup_transpose, np.kron(sigma_inv, sigma_inv)), dup_mat)  # calculate the downright of information matrix
        I22[:(k*k*p+k), :(k*k*p+k)] = I221
        I22[(k*k*p+k):, (k*k*p+k):] = I222
        return I22      # ndarray

    # mean shift model
    def mean_shift(self):
        B = self.B
        k, p = self.k, self.p
        sample = np.array(self.samples.values)
        n = self.whole_samples_n
        critical = self.ms_critical
        sigma_inv = self.sigma_inv
        dup_mat = self.dup_mat
        dup_transpose = dup_mat.T   # transpose of duplication matrix

        z = self.z
        I22 = self.I22

        # calculate the score statistics
        score = []
        for i in range(n-p):
            # calculate the first-order patial derivative of gamma
            zi = z[i]
            Lr = np.dot((sample[i]-np.dot(B, zi)), sigma_inv)

            # calculate I12
            I12 = np.zeros((k, ((1+2*p)*k*k+3*k)//2))                          # initial I12
            I12[:, :(k*k*p+k)] = np.dot(sigma_inv, np.kron(zi, np.eye(k)))

            # calculate I21
            I21 = I12.T

            # calculate score statistics
            score.append(np.dot(np.dot(Lr.T, inv(sigma_inv - np.dot(np.dot(I12, inv(I22)), I21))), Lr))

        scores = pd.DataFrame(score, columns=['scores'])                                        # scores with nature index
        scores = pd.concat((self.whole_samples, scores), axis=1, ignore_index=False)            # scores with data
        self.ms_scores = scores

        outliers_scores = scores[scores.scores >= critical]                                     # outliers scores with corresponding index
        correct_samples = self.whole_samples.loc[list(scores[scores.scores < critical].index)]  # correct samples with correspongding index

        return self.ms_scores, outliers_scores, correct_samples

    # variance weight model
    def variance_weight(self):
        B = self.B
        k, p = self.k, self.p
        sample = np.array(self.samples.values)
        n = self.whole_samples_n
        critical = self.vw_critical
        sigma_inv = self.sigma_inv
        dup_mat = self.dup_mat
        dup_transpose = dup_mat.T

        z = self.z
        I22 = self.I22

        # calculate the score statistics
        score = []
        vec_sigma_inv = np.array(sigma_inv).ravel(order='F')   # semi-vector operator

        # calculate I12
        I12 = np.zeros(((1+2*p)*k*k+3*k)//2)                   # initial I12
        I12[k*k*p+k:] = np.dot(vec_sigma_inv, dup_mat)

        # calculate I21
        I21 = I12.T

        b = np.dot(np.dot(I12, inv(I22)), I21)

        for i in range(n-p):
            # calculate the first-order patial derivative of w
            zi = z[i]
            Lw_square = pow((k-np.dot(np.dot((sample[i]-np.dot(B, zi)), sigma_inv), sample[i]-np.dot(B, zi)))/2, 2)

            score.append(Lw_square/(k/2-b))   #  store scores

        scores = pd.DataFrame(score, columns=['scores'])                                       # scores with nature index
        scores = pd.concat((self.whole_samples, scores), axis=1, ignore_index=False)           # scores with data
        self.vw_scores = scores

        outliers_scores = scores[scores.scores >= critical]                                    # outliers scores with corresponding index
        correct_samples = self.whole_samples.loc[list(scores[scores.scores < critical].index)]  # correct samples with correspongding index

        return self.vw_scores, outliers_scores, correct_samples

    # plot of whole real samples' scores and critical
    def plot(self, model):
        if model == 'ms':
            self.ms_scores.iloc[:, -1].plot(kind='line')                                         # whole real samples
            plt.hlines(self.ms_critical, 0, self.samples_n, colors = "r", linestyles = "solid")  # horizontal line of critical
            plt.title('Score Test of Mean Shift Model')
            plt.legend(loc='best', prop={'size':12})                                             # font size
        elif model == 'vw':
            self.vw_scores.iloc[:, -1].plot(kind='line')                                                     # whole real samples
            plt.hlines(self.vw_critical, 0, self.samples_n, colors = "r", linestyles = "solid")  # horizontal line of critical
            plt.title('Score Test of Variance Weight Model')
            plt.legend(loc='best', prop={'size':12})                                             # font size

# VAR Model
class myVAR:
    __slots__ = 'samples', 'k', 'columns'
    def __init__(self, samples):
        self.samples = samples.iloc[:, 1:]
        self.k = samples.shape[1]-1
        self.columns = self.samples.columns

    def logdiff(self):
        self.samples = np.log(self.samples).diff().dropna()  # first-order difference

    # original samples graph
    def plot(self, residual):
        # 1. residuals graph
        plt.figure()
        for i in range(self.k):
            plt.subplot(self.k, 1, i+1)
            residual.iloc[:, i].plot(kind='line')
            plt.legend(loc='best', prop={'size':12})
            title = self.columns[i]
            plt.title('Residual Series of ' + title)
            plt.subplots_adjust(hspace=0.4)

        # 2. Q-Q graph
        for i in range(self.k):
            plt.figure()
            probplot(residual.iloc[:, i], plot=plt)
            title = self.columns[i]
            plt.title('Residual\'s Q-Q of ' + title)
            plt.subplots_adjust(hspace=0.4)

    def test(self, kind, residual=None):
        if kind == 'adfuller':
            # 1.adfuller test
            print("====================Adfuller Test====================")
            print('{0:<8}\t{1:<8}\t{2:<8}\t{3:<8}\t{4:<8}\t{5:<8}\t{6:<8}\t{7:<8}'.format('sample', 'statistic', 'pvalue',
                                                                                          'usedlag', 'nobs', '1%', '5%', '10%'))
            title = self.columns
            for i in range(self.k):
                adf = adfuller(self.samples.iloc[:, i])
                print('{0:<8}\t{1:<8.4f}\t{2:<8.4f}\t{3:<8}\t{4:<8}\t{5:<8.4f}\t{6:<8.4f}\t{7:<8.4f}'.format(title[i], adf[0], adf[1], adf[2],
                                                                                                             adf[3], adf[4]['1%'], adf[4]['5%'], adf[4]['10%']))
            print('\n')

        elif kind == 'residual':
        # 2.normality of residuals
            print("====================Normality of Residuals====================")
            print('{0:<8}\t{1:<8}\t{2:<8}'.format('sample', 'statistic', 'pvalue'))
            title = self.columns
            for i in range(self.k):
                ks = kstest(residual.iloc[:,0], 'norm')
                print('{0:<8}\t{1:<8.4f}\t{2:<8.4f}'.format(title[i], ks[0], ks[1]))
            print('\n')

    def estimate(self):
        self.test(kind='adfuller')

        model = VAR(self.samples)
        selected_orders = model.select_order().selected_orders              # best orders corresponding to information criterion
        lags = pd.Series(selected_orders).value_counts().index.tolist()[0]  # select the best order from all information criterion

        # output lags
        print("====================Select Order====================")
        print("{0:<5}\t{1:<5}\t{2:<5}\t{3:<5}\t{4:<10}".format('aic', 'bic', 'fpe', 'hqic', 'best-lags'))
        print("{0:<5}\t{1:<5}\t{2:<5}\t{3:<5}\t{4:<5}\n".format(selected_orders['aic'], selected_orders['bic'],
                                                              selected_orders['fpe'], selected_orders['hqic'], lags))

        result = model.fit(lags)                                            # model fit with the best order

        plt.figure()
        result.plot()                                                       # data visualization
        plt.figure()
        result.plot_acorr()                                                 # autocorrelation

        new_sigma = result.sigma_u_mle                                      # sigma of maximum likelihood estimation
        residual = result.resid                                             # residuals of the model
        self.test(kind='residual', residual=residual)                       # residuals test
        self.plot(residual)                                                 # residual and Q-Q graph

        print(result.summary())                                             # structured result

        shape = result.coefs.shape
        coefs = np.zeros((shape[1], shape[0]*shape[2]))                     # initial coefficient matrix without constant term
        for i in range(result.coefs.shape[0]):
            coefs[:, i*self.k:i*self.k+shape[2]] = result.coefs[i]
        new_B = np.concatenate((result.coefs_exog, coefs), axis=1)          # concatenate the constant term and coefficient term

        return new_sigma, new_B, residual

# Montecarlo Application
def montecarlo():
    mu = np.array([0, 0])
    sigma = np.array([[1, 0.6], [0.6, 1]])
    B = np.array([[0.0127, 0, 0.1369, 0.0829, -0.1874, 0, 0], [0.0061, 0, 0.0852, 0, 0, 0, -0.1054]])

    # generate samples
    M = Montecarlo(mu, sigma, B)
    samples = M.gene_samples()     # generate samples

    samples.iloc[[200, 400, 600, 800], 1:3] = samples.iloc[[200, 400, 600, 800], 1:3] + [2, 2]     # add noises

    # original model
    V1 = myVAR(samples)
    new_sigma, new_B, residual = V1.estimate()    # model estimation
    k = new_B.shape[0]
    p = (new_B.shape[1]-1)//k

    # find outliers by mean shift model
    S = ScoreTest(samples, new_sigma, new_B)
    print("=======================Mean Shift Model=======================\n")
    [ms_scores, ms_outliers_scores, ms_correct_samples] = S.mean_shift()  # score and outliers
    print('{0:^10}\t{1:^10}\t{2:^10}\t{3:^10}\t{4:^10}'.format('model', 'total', 'lags', 'outliers', 'correct'))
    print('{0:^10}\t{1:^10}\t{2:^10}\t{3:^10}\t{4:^10}'.format('mean shift', len(ms_scores), p, len(ms_outliers_scores), len(ms_correct_samples)))
    print('\n')

    plt.figure()
    S.plot('ms')  # outliers critical graph

    # find outliers by variance weight model
    print("=======================Variance Weight Model=======================\n")
    [vw_scores, vw_outliers_scores, vw_correct_samples] = S.variance_weight()  # 得分和异常点
    print('{0:^10}\t{1:^10}\t{2:^10}\t{3:^10}\t{4:^10}'.format('model', 'total', 'lags', 'outliers', 'correct'))
    print('{0:^10}\t{1:^10}\t{2:^10}\t{3:^10}\t{4:^10}'.format('variance', len(vw_scores), p, len(vw_outliers_scores), len(vw_correct_samples)))

    plt.figure()
    S.plot('vw')  # outliers critical graph

    # error correction model
    correct_samples = pd.concat([ms_correct_samples, vw_correct_samples], axis=0, ignore_index=False)
    correct_samples = correct_samples[correct_samples.index.duplicated()]

    print(len(correct_samples))
    print("=======================Corrected Model=======================")
    plt.figure()
    V2 = myVAR(correct_samples)
    new_sigma, new_B, residual = V2.estimate()

    return ms_outliers_scores, vw_outliers_scores, correct_samples

# Empirical Application
def empirical(samples):
    # original model
    V1 = myVAR(samples)
    new_sigma, new_B, residual = V1.estimate()    # model estimation
    k = new_B.shape[0]
    p = (new_B.shape[1]-1)//k

    # find outliers by mean shift model
    S = ScoreTest(samples, new_sigma, new_B)
    print("=======================Mean Shift Model=======================\n")
    [ms_scores, ms_outliers_scores, ms_correct_samples] = S.mean_shift()  # score and outliers
    print('{0:^10}\t{1:^10}\t{2:^10}\t{3:^10}\t{4:^10}'.format('model', 'total', 'lags', 'outliers', 'correct'))
    print('{0:^10}\t{1:^10}\t{2:^10}\t{3:^10}\t{4:^10}'.format('mean shift', len(ms_scores), p, len(ms_outliers_scores), len(ms_correct_samples)))
    print('\n')

    plt.figure()
    S.plot('ms')  # outliers critical graph

    # find outliers by variance weight model
    print("=======================Variance Weight Model=======================\n")
    [vw_scores, vw_outliers_scores, vw_correct_samples] = S.variance_weight()  # 得分和异常点
    print('{0:^10}\t{1:^10}\t{2:^10}\t{3:^10}\t{4:^10}'.format('model', 'total', 'lags', 'outliers', 'correct'))
    print('{0:^10}\t{1:^10}\t{2:^10}\t{3:^10}\t{4:^10}'.format('variance', len(vw_scores), p, len(vw_outliers_scores), len(vw_correct_samples)))
    print('\n')

    plt.figure()
    S.plot('vw')  # outliers critical graph

    # error correction model
    correct_samples = pd.concat([ms_correct_samples, vw_correct_samples], axis=0, ignore_index=False)
    correct_samples = correct_samples[correct_samples.index.duplicated()]

    print("=======================Corrected Model=======================")
    plt.figure()
    V2 = myVAR(correct_samples)
    new_sigma, new_B, residual = V2.estimate()

    return ms_outliers_scores, vw_outliers_scores, correct_samples
