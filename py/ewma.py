
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Ewma(object):
    """
    In statistical quality control, the EWMA chart (or exponentially weighted moving average chart)
    is a type of control chart used to monitor either variables or attributes-type data using the monitored business
    or industrial process's entire history of output. While other control charts treat rational subgroups of samples
    individually, the EWMA chart tracks the exponentially-weighted moving average of all prior sample means.
    WIKIPEDIA: https://en.wikipedia.org/wiki/EWMA_chart
    """

    def __init__(self, alpha=0.3, coefficient=3):
        """
        :param alpha: Discount rate of ewma, usually in (0.2, 0.3).
        :param coefficient: Coefficient is the width of the control limits, usually in (2.7, 3.0).
        """
        self.alpha = alpha
        self.coefficient = coefficient

    def predict(self, X):
        """
        Predict the expontential fitting line, ucl and lcl
        :param X: the time series to detect of
        :param type X: pandas.Series
        :return: s,ucl,lcl
        
        s_ewma^2=alpha(2-alpha)*s^2
        
        UCL==x_bar+k*s_ewma
        LCL=x_bar−k*s_ewma
        
        """
        s = [X[0]]
        for i in range(1, len(X)):
            temp = self.alpha * X[i] + (1 - self.alpha) * s[-1]
            s.append(temp)
        s_avg = np.mean(s)
        sigma = np.sqrt(np.var(X))
        ucl = s_avg + self.coefficient * sigma * np.sqrt(self.alpha / (2 - self.alpha))
        lcl = s_avg - self.coefficient * sigma * np.sqrt(self.alpha / (2 - self.alpha))
#         if s[-1] > ucl or s[-1] < lcl:
#             return 0
        return s,ucl,lcl

    def ewma_control_chart(self,y):
        """ 
        Function to draw EWMA control chart,WIKIPEDIA: https://en.wikipedia.org/wiki/EWMA_chart
        """
        if len(y.columns)<1:
            print("No data plot for EWMA plot")
        elif len(y.columns)>1:
            fig,axis=plt.subplots(len(y.columns),1,figsize=(12,10))
            i=0
            for col in y.columns:
                s,ucl,lcl=self.predict(list(y[col]))
                axis[i].plot(y.index,s)
                axis[i].scatter(y.index,s)
                axis[i].axhline(ucl,color="red",linestyle='--',label="UCL")
                axis[i].text(y.index[-1]-1,ucl,f"UCL={ucl:.3f}",fontsize=12,color="red")
                axis[i].text(y.index[-1]-1,lcl,f"LCL={lcl:.3f}",fontsize=12,color="red")
                axis[i].axhline(lcl,color="red",linestyle='--',label="LCL")
                axis[i].set_title("EWMA Control Chart for {},alpha={}".format(col,self.alpha))
                i=i+1
        else:
            fig=plt.figure(figsize=(12,5))
            col=y.columns[0]
            s,ucl,lcl=self.predict(list(y[col]))
            plt.plot(y.index,s)
            plt.scatter(y.index,s)
            plt.axhline(ucl,color="red",linestyle='--',label="UCL")
            plt.text(y.index[-1]-1,ucl,f"UCL={ucl:.3f}",fontsize=12,color="red")
            plt.text(y.index[-1]-1,lcl,f"LCL={lcl:.3f}",fontsize=12,color="red")
            plt.axhline(lcl,color="red",linestyle='--',label="LCL")
            plt.title("EWMA Control Chart for {},alpha={}".format(col,self.alpha))

        return fig