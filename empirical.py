import numpy as np
from numpy.linalg import inv, cholesky
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2, kstest, probplot, ks_2samp
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.base.datetools import dates_from_str

import os
os.environ['R_HOME'] = 'D:\R\R-3.3.3'
os.environ['R_USER'] = 'D:\\Anaconda\\Lib\\site-packages\\rpy2'
from rpy2 import robjects
from rpy2.robjects.packages import importr

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
        I12[k*k*p+k:] = 1/2*np.dot(vec_sigma_inv, dup_mat)

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
            plt.title('Score statistical value of MSOM')
            plt.legend(('score', 'benchmark'),loc='best', prop={'size':6})                     # font size
        elif model == 'vw':
            self.vw_scores.iloc[:, -1].plot(kind='line')                                        # whole real samples
            plt.hlines(self.vw_critical, 0, self.samples_n, colors = "r", linestyles = "solid")  # horizontal line of critical
            plt.title('Score statistical value of CWM')
            plt.legend(('score', 'benchmark'), loc='best', prop={'size':6})                    # font size

class MyVAR:
    __slots__ = 'samples', 'k', 'columns'
    counts = 0
    def __init__(self, samples):
        self.samples = samples.iloc[:, 1:]
        self.k = samples.shape[1]-1
        self.columns = self.samples.columns
        MyVAR.counts += 1

    def logdiff(self):
        self.samples = np.log(self.samples).diff().dropna()  # first-order difference

    # standard residual figure
    def plot(self, residual):
        # 1. residuals graph
        plt.figure(dpi=300)
        for i in range(self.k):         
            plt.subplot(self.k, 1, i+1)
            residual.iloc[:, i].plot(kind='line')
            title = self.columns[i]
            plt.title('Residual of ' + title)
            plt.subplots_adjust(hspace=0.4)
        plt.savefig('figures/' + str(MyVAR.counts) + '_std_resi' + '.jpg')

        # 2. kernel density estimation
        plt.figure(dpi=300)
        for i in range(self.k):
            plt.subplot(self.k, 1, i+1)
            mu = residual.iloc[:, i].mean()
            sigma = residual.iloc[:, i].std()
            x = np.arange(-8, 8, 0.1)
            pdf = list(map(lambda x:np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi)), x))
            residual.iloc[:, i].plot(linestyle='--',kind='kde')
            plt.plot(x, pdf)
            plt.legend(('kde', 'normal'), loc='best', prop={'size':8})
            title = self.columns[i]
            plt.title('kde of ' + title)
            plt.subplots_adjust(hspace=0.4)
        plt.savefig('figures/' + str(MyVAR.counts) + '_kde' + '.jpg')
        
        # 3. Q-Q graph
        plt.figure(dpi=300)
        for i in range(self.k):
            plt.subplot(1, self.k, i+1)
            probplot(residual.iloc[:, i], plot=plt)
            title = self.columns[i]
            plt.title('Residual Q-Q of ' + title)
            plt.subplots_adjust(hspace=0.4)
        plt.savefig('figures/' + str(MyVAR.counts) + '_QQ' + '.jpg')

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
                ks = kstest(residual.iloc[:, i], 'norm')
                print('{0:<8}\t{1:<8.4f}\t{2:<8.4f}'.format(title[i], ks[0], ks[1]))
            print('\n')

    def estimate(self):
        self.test(kind='adfuller')

        model = VAR(self.samples)
        selected_orders = model.select_order().selected_orders              # best orders corresponding to information criterion
        lags = selected_orders['aic']
#         lags = pd.Series(selected_orders).value_counts().index.tolist()[0]  # select the best order from all information criterion

        # output lags
        print("====================Select Order====================")
        print("{0:<5}\t{1:<5}\t{2:<5}\t{3:<5}\t{4:<10}".format('aic', 'bic', 'fpe', 'hqic', 'best-lags'))
        print("{0:<5}\t{1:<5}\t{2:<5}\t{3:<5}\t{4:<5}\n".format(selected_orders['aic'], selected_orders['bic'],
                                                              selected_orders['fpe'], selected_orders['hqic'], lags))
        if MyVAR.counts==1:
            lag = lags
        else:
            lag = 1 # 这个包选择阶数可能有问题，用R、eviews试一下
        result = model.fit(lag)                                            # model fit with the best order

        new_sigma = result.sigma_u_mle                                      # sigma of maximum likelihood estimation
        residual = result.resid                                             # residuals of the model
        std_resi = (residual-residual.mean())/residual.std()                # 标准化残差
        self.test(kind='residual', residual=std_resi)                       # residuals test
        self.plot(std_resi)                                                 # Q-Q graph of standard residual

        print(result.summary())                                             # structured result

        shape = result.coefs.shape
        coefs = np.zeros((shape[1], shape[0]*shape[2]))                     # initial coefficient matrix without constant term
        for i in range(result.coefs.shape[0]):
            coefs[:, i*self.k:i*self.k+shape[2]] = result.coefs[i]
        new_B = np.concatenate((result.coefs_exog, coefs), axis=1)          # concatenate the constant term and coefficient term

        return new_sigma, new_B, residual

def empirical(samples):
    # original model
    V1 = MyVAR(samples)
    new_sigma, new_B, residual = V1.estimate()    # model estimation
    k = new_B.shape[0]
    p = (new_B.shape[1]-1)//k

    print(new_B)
    print(new_sigma)
    
    # find outliers by mean shift model
    S = ScoreTest(samples, new_sigma, new_B)
#     print("=======================Mean Shift Model=======================\n")
    [ms_scores, ms_outliers_scores, ms_correct_samples] = S.mean_shift()  # score and outliers
    print('{0:^10}\t{1:^10}\t{2:^10}\t{3:^10}\t{4:^10}'.format('model', 'total', 'lags', 'outliers', 'correct'))
    print('{0:^10}\t{1:^10}\t{2:^10}\t{3:^10}\t{4:^10}'.format('mean shift', len(ms_scores), p, len(ms_outliers_scores), len(ms_correct_samples)))
    print('\n')

    # find outliers by variance weight model
    print("=======================Variance Weight Model=======================\n")
    [vw_scores, vw_outliers_scores, vw_correct_samples] = S.variance_weight()  # 得分和异常点
    print('{0:^10}\t{1:^10}\t{2:^10}\t{3:^10}\t{4:^10}'.format('model', 'total', 'lags', 'outliers', 'correct'))
    print('{0:^10}\t{1:^10}\t{2:^10}\t{3:^10}\t{4:^10}'.format('variance', len(vw_scores), p, len(vw_outliers_scores), len(vw_correct_samples)))
    print('\n')

    plt.figure(dpi=300)
    plt.subplot(2, 1, 1)
    S.plot('ms')
    plt.subplot(2, 1, 2)
    S.plot('vw')  # outliers critical graph
    plt.subplots_adjust(hspace=0.4)
    plt.savefig('figures/score-benchmark.jpg')

    # error correction model
    correct_samples = pd.concat([ms_correct_samples, vw_correct_samples], axis=0, ignore_index=False)
    correct_samples = correct_samples[correct_samples.index.duplicated()]

    print("=======================Corrected Model=======================")
    plt.figure()
    V2 = MyVAR(correct_samples)
    new_sigma, new_B, residual = V2.estimate()
    
    return ms_outliers_scores, vw_outliers_scores, correct_samples

if __name__ == '__main__':
    samples = pd.read_csv('GSCI-CVX(08-19)w.csv')
    plt.figure(dpi=300)
    plt.subplot(211)
    samples['CVX'].plot(kind='line')
    plt.title('Weekly Log-return of CVX')
    plt.subplots_adjust(hspace=0.4)
    plt.subplot(212)
    samples['GSCI'].plot(kind='line')
    plt.title('Weekly Log-return of GSCI')
    plt.savefig('figures/CVX-GSCI.jpg')

    [ms_outliers_scores, vw_outliers_scores, correct_samples] = empirical(samples)
    correct_samples.to_csv('correct_samples.csv', sep=',')
