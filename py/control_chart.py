

from hotelling.plots import control_chart, control_stats, univariate_control_chart
from py.ewma import Ewma
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def plot_control_chart(loc):
    '''Function to plot Hotelling T^2 chart and EWMA control chart
    '''
    out_dir='output/'
    chart_dir=os.path.join(out_dir, 'controlchart/')

    pred_path=out_dir+"prediction/"
    pred_csv="gap_flush_prediction.csv"
    gap_flush_pred_path=out_dir+"prediction/"+pred_csv
    if os.path.exists(gap_flush_pred_path) ==True:
        df_pred=pd.read_csv(gap_flush_pred_path)
        if df_pred.shape[0]>0:
            if loc=="all":
                data_pred_pivo=df_pred.pivot(index="run_no",columns="loc")[["Gap_pred(mm)","Flush_pred(mm)"]]
                data_pred_pivo.columns=[i[0]+"_"+i[1] for i in data_pred_pivo.columns]
                data_pred_pivo=data_pred_pivo.reset_index().sort_values(by="run_no").drop(['Flush_pred(mm)_A', 'Flush_pred(mm)_B'],axis=1)
                if data_pred_pivo.shape[0]>=3:
                    y=data_pred_pivo
                    y.set_index("run_no",inplace=True)
                    chart=control_chart(y, alpha=0.01, legend_right=True);
                    plt.xticks(range(len(y.index)),labels=y.index)
                    chart.figure.savefig(chart_dir+loc+"-hotelling-chart.png")
                else:
                    print("No T^2 control chart generated for loc{},since less than 3 samples".format(loc))

            if loc in ["A","B"]:
                df_pred_loc=df_pred[df_pred["loc"]==loc].sort_values(by="run_no")
                if df_pred_loc.shape[0]>=3:
                    y=df_pred_loc[["run_no","Gap_pred(mm)"]]
                    y.set_index("run_no",inplace=True)

                    # chart2=univariate_control_chart(y, legend_right=True);   
                    # # chart2.suptitle(loc)  
                    # chart2.savefig(chart_path+loc+"-Uni-chart.png")

                    ewma_object=Ewma()
                    chart=ewma_object.ewma_control_chart(y);   
                    # chart2.suptitle(loc)  
                    chart.savefig(chart_dir+loc+"-EWMA-chart.png")

                else:
                    print("No EWMA control chart generated for loc{},since less than 3 samples".format(loc))       

            if loc in ["C","D","E"]:
                df_pred_loc=df_pred[df_pred["loc"]==loc].sort_values(by="run_no")
                if df_pred_loc.shape[0]>=3:
                    y=df_pred_loc[["run_no","Gap_pred(mm)","Flush_pred(mm)"]]
                    y.set_index("run_no",inplace=True)

                    # chart2=univariate_control_chart(y, legend_right=True);   
                    # # chart2.suptitle(loc)  
                    # chart2.savefig(chart_path+loc+"-Uni-chart.png")

                    ewma_object=Ewma()
                    chart=ewma_object.ewma_control_chart(y);   
                    # chart2.suptitle(loc)  
                    chart.savefig(chart_dir+loc+"-EWMA-chart.png")

                else:
                    print("No EWMA control chart generated for loc{},since less than 3 samples".format(loc))       
    else:
        print("No control chart since no file found of {}".format(gap_flush_pred_path))
    
    return(chart)