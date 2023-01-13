
#!/usr/bin/env python
import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.patches as patches
import pickle


# measure_location="A"

def pred_gap_flush(df_pixels_new,pred_path,run_no,out_filename):
    cameras=["Left","Right","Middle Left","Middle"]
    ## Check if directory exist, create if not
    if os.path.exists(pred_path)==False:
        os.mkdir(pred_path)
    
    gap_flush_pred_path=pred_path+out_filename#"gap_flush_prediction.csv"
    if os.path.exists(gap_flush_pred_path) ==True:
        df_pred=pd.read_csv(gap_flush_pred_path)
    else:
        df_pred=pd.DataFrame()

    ## Load or create file
    if df_pred.shape[0]>0 and (int(run_no) in df_pred["run_no"].unique()):
        existing_loc=df_pred.loc[df_pred["run_no"]==int(run_no),"loc"].unique()
        if all( i in existing_loc for i in ["C","D","E"]):
            print("Warning: The prediction result of Top2 is avaliable for run= {}, it will be replaced.".format(run_no))
            # df_pred=df_pred[np.logical_xor(df_pred["run_no"]==int(run_no),df_pred["loc"]in ["C","D","E"])].copy(deep=True)
            df_pred=df_pred[~((df_pred["run_no"]==int(run_no))&(df_pred["loc"].isin(["C","D","E"])))].copy(deep=True)


    df_pred_new=pd.DataFrame()

    for measure_location in ["C","D","E"]: #["A","B","C","D","E","F"]:
        df_extract=df_pixels_new.loc[df_pixels_new["loc"]==measure_location,["pix_dist","run_no","camera"]]
        df_extract["run_no"]=pd.to_numeric(df_extract["run_no"])

        df_pivot=pd.DataFrame(df_extract.pivot(index="run_no",columns="camera",
                                               values="pix_dist").reset_index())
        df_pixels_new2=df_pivot.sort_values(by=["run_no"],ascending=True,axis=0)
        df_pixels_new2["loc"]=measure_location#[["Left","Middle","Right","Middle Left"]]


        ###load model
        m1=pickle.load(open("./model/Gap_Location_{}.pickle".format(measure_location),"rb"))
        m2=pickle.load(open("./model/Flush_Location_{}.pickle".format(measure_location),"rb"))


        if len(set(df_pixels_new.camera.unique()))==len(cameras):
            x=df_pixels_new2.loc[df_pixels_new2["run_no"]==int(run_no),["Right","Middle","Left","Middle Left"]]
            gap=m1.predict(x)
            flush=m2.predict(x)
        else:
            missed_camera=set(cameras)-set(set(df_pixels_new.camera.unique()))
            x=df_pixels_new2.loc[df_pixels_new2["run_no"]==int(run_no),list(set(df_pixels_new.camera.unique()))]
            for c in list(missed_camera):
                x[c]=0
            gap=float(0)
            flush=float(0)
            print("No pixel count from camera {},Gap and flush prediction will not run for loc=.".format(missed_camera,measure_location))

        ## Save prediction
        df_pred_new_loc=pd.DataFrame({"run_no":int(run_no),"loc":measure_location,"Left":x["Left"],"Middle":x["Middle"],
                     "Middle Left":x["Middle Left"],"Right":x["Right"],
                      "Gap_pred(mm)":np.round(gap,5),"Flush_pred(mm)":np.round(flush,5)})
        df_pred_new=pd.concat([df_pred_new,df_pred_new_loc]).sort_values(by=["run_no","loc"]).drop_duplicates()

    ## Export result
    df_pred=pd.concat([df_pred,df_pred_new]).sort_values(by=["run_no","loc"]).drop_duplicates()
    df_pred.to_csv(gap_flush_pred_path,index=False)
    print("Result is saved to "+gap_flush_pred_path)
    return df_pred_new

