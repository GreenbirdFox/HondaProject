
import numpy as np
import os
import sys

import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

def plot_result(img,bin_edg,df_pix,df_pred,ROI_location):
    ''' Default img= imgM
        df: eg df_pred_new. contains information of loc, y, x_1,x_2,gap,flush

    '''

    ROI_x_start,ROI_x_end,ROI_y_start,ROI_y_end=ROI_location
    
    if df_pix.shape[0]>0:
        fig, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3, figsize = (16,8), tight_layout = True)
        ax1.imshow(bin_edg);ax1.set_title("Edge Detection")
        for loc in list(df_pix["loc"].unique()):
            ax1.plot(df_pix.loc[df_pix["loc"]==loc,"x_1"],df_pix.loc[df_pix["loc"]==loc,"y_1"], '+')
            ax1.plot(df_pix.loc[df_pix["loc"]==loc,"x_2"], df_pix.loc[df_pix["loc"]==loc,"y_1"], '+')
            rect = patches.Rectangle((ROI_x_start, ROI_y_start), ROI_x_end-ROI_x_start, ROI_y_end-ROI_y_start, linewidth=2, edgecolor='b', facecolor='none')
            ax1.text(ROI_x_start+50,ROI_y_start-10,"ROI",fontsize=15)
            ax1.add_patch(rect)

        ax2.imshow(bin_edg[ROI_y_start:ROI_y_end,ROI_x_start:ROI_x_end]);ax2.set_title("ROI-Edge")
        for loc in list(df_pix["loc"].unique()):
            ax2.plot(np.max(df_pix.loc[df_pix["loc"]==loc,"x_1"]-ROI_x_start,0),df_pix.loc[df_pix["loc"]==loc,"y_1"]-ROI_y_start, '+')
            ax2.plot(np.max(df_pix.loc[df_pix["loc"]==loc,"x_2"]-ROI_x_start,0), df_pix.loc[df_pix["loc"]==loc,"y_1"]-ROI_y_start, '+')
            pixel_count=round(df_pix.loc[df_pix["loc"]==loc,"pix_dist"],5)
            ax2.text(df_pix.loc[df_pix["loc"]==loc,"x_2"]-ROI_x_start+50,df_pix.loc[df_pix["loc"]==loc,"y_1"]-ROI_y_start,
                     "pixel="+str(int(pixel_count)),bbox=dict(facecolor='red', alpha=0.4))

        ax3.imshow(img[ROI_y_start:ROI_y_end,ROI_x_start:ROI_x_end]);ax3.set_title("ROI")
        for loc in list(df_pred["loc"].unique()):
            ax3.plot(np.max(df_pix.loc[df_pix["loc"]==loc,"x_1"]-ROI_x_start,0),df_pix.loc[df_pix["loc"]==loc,"y_1"]-ROI_y_start, '+')
            ax3.plot(np.max(df_pix.loc[df_pix["loc"]==loc,"x_2"]-ROI_x_start,0), df_pix.loc[df_pix["loc"]==loc,"y_1"]-ROI_y_start, '+')
            gap=round(df_pred.loc[df_pred["loc"]==loc,'Gap_pred(mm)'].values[0],4)
            flush=round(df_pred.loc[df_pred["loc"]==loc,'Flush_pred(mm)'].values[0],4)
            ax3.text(df_pix.loc[df_pix["loc"]==loc,"x_2"]-ROI_x_start+50,df_pix.loc[df_pix["loc"]==loc,"y_1"]-ROI_y_start,
                     "Gap="+str(gap)+"mm",bbox=dict(facecolor='red', alpha=0.4))
            ax3.text(df_pix.loc[df_pix["loc"]==loc,"x_2"]-ROI_x_start+50,df_pix.loc[df_pix["loc"]==loc,"y_1"]-ROI_y_start+100,
                     "flush="+str(flush)+"mm",bbox=dict(facecolor='blue', alpha=0.4))
    else:
        fig, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3, figsize = (16,8), tight_layout = True)
        ax1.imshow(bin_edg);ax1.set_title("Edge Detection")
        rect = patches.Rectangle((ROI_x_start, ROI_y_start), ROI_x_end-ROI_x_start, ROI_y_end-ROI_y_start, linewidth=2, edgecolor='b', facecolor='none')
        ax1.text(ROI_x_start+50,ROI_y_start-10,"ROI",fontsize=15)
        ax1.add_patch(rect)

        pixel_count=0
        ax2.imshow(bin_edg[ROI_y_start:ROI_y_end,ROI_x_start:ROI_x_end]);ax2.set_title("ROI-Edge")
        ax2.text(ROI_x_end-ROI_x_start-10,ROI_y_end-ROI_y_start-10,
                 "Edge not detected",bbox=dict(facecolor='red', alpha=0.4))

        ax3.imshow(img[ROI_y_start:ROI_y_end,ROI_x_start:ROI_x_end]);ax3.set_title("ROI")
        gap=0; flush=0
        ax3.text(ROI_x_end-ROI_x_start-10,ROI_y_end-ROI_y_start-10,
                 "No gap &flush computed",bbox=dict(facecolor='red', alpha=0.4))

    # plt.show()
    return fig