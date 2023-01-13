
## Compare to imagePre_newV2.py, control charts are added. ROI measurement plots are improved.
#!/usr/bin/env python
import cv2
import numpy as np
import os
import sys

import glob
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.patches as patches
import py.pred_gap_flush as pred
import py.plot_result as pr

from hotelling.plots import control_chart, control_stats, univariate_control_chart
import warnings
from py.ewma_control_chart import ewma_control_chart

warnings.filterwarnings('ignore', '.*masked element*', )

### Need set
data_dir='220912_Photos and Groud Truth Results/'
out_dir='220912_analoutput/' ## Need set

run_no=sys.argv[1]
# run_no="15"


#### ---------Check and create directory to store outputs.--------------
fig_dir='fig/'
fig_path= os.path.join(out_dir, fig_dir)
if os.path.exists(fig_path) ==False:
    os.mkdir(fig_path)
    print("Directory '%s' created" %fig_path)

chart_dir='controlchart/'
chart_path= os.path.join(out_dir, chart_dir)
if os.path.exists(chart_path) ==False:
    os.mkdir(chart_path)
    print("Directory '%s' created" %chart_path)

csv_dir= 'pixelscount/'
csv_path= os.path.join(out_dir, csv_dir)

if os.path.exists(csv_path) ==False:
    os.mkdir(csv_path)
    print("Directory '%s' created" %csv_path)


#### ----------Parameters ---------------------------------------------

canny_thresh={"Right":[100,250],"Left":[100,200],"Middle Left":[100,200],"Middle":[50,150]}
shift_par={"Right":2.4,"Left":2.05,"Middle Left":2.4,"Middle":2.3}


#### ----------Function-----------------------------------------------
def fit_linear_lines(line1_pt,line2_pt,line3_pt,line4_pt):

    lines_pt=[line1_pt,line2_pt,line3_pt,line4_pt]
    lines_a=[]#store intercept of each line
    lines_b=[]#store slope of each line
    for line_pt in lines_pt:
        x=line_pt[:,:,0]
        y=line_pt[:,:,1]
        m = LinearRegression().fit(x, y)
        r_sq=m.score(x,y)
        # print(f"LR model R-square: {r_sq}")
        a,b=m.intercept_,m.coef_[0]
        lines_a.append(a)
        # print("intercept",a)
        lines_b.append(b)
    lines_a=[a[0] for a in lines_a]
    lines_b=[b[0] for b in lines_b]
    return(lines_a,lines_b)

def fun(out_dir,run_no,camera,df_pixels_new,plot_para):  ##button to trigger plot
    # df_pixes_camera=df_pixels_new.loc[df_pixels_new["camera"]==camera,].reset_index(drop=True)
    # [img,bin_ed,df_pred_new,[ROI_x_start,ROI_x_end,ROI_y_start,ROI_y_end]]=plot_para
    # plot_result(img,bin_edg,df_pixes_camera,df_pred_new,[ROI_x_start,ROI_x_end,ROI_y_start,ROI_y_end])
    # #     plt.show()
    messagebox.showinfo("Hello", "Left Button clicked") 

#### --------- Main -----------------------------------------------------

## Create data frame or csv to store result
pix_counts_file=csv_path+"allcameras_pixels_count_fromImageProcess.csv"
if os.path.exists(pix_counts_file) ==True:
    df_pixels_all=pd.read_csv(pix_counts_file)
else:
    df_pixels_all=pd.DataFrame()

if df_pixels_all.shape[0]>0 and (int(run_no) in df_pixels_all["run_no"].unique()):
    print("Warning: The result is avaliable for run= {}, it will be replaced.".format(run_no))
    df_pixels_all=df_pixels_all[df_pixels_all["run_no"]!=int(run_no)].copy(deep=True)
else:
    print("Pixel counting for run ={}, Check if the image have been created from cameras...".format(run_no))

## Set Image directory of 4 Cameras
cameras=["Left","Right","Middle Left","Middle"]
imgL_path=data_dir+'Run '+run_no+'/Run'+run_no+"-"+cameras[0]+".jpg"
imgR_path=data_dir+'Run '+run_no+'/Run'+run_no+"-"+cameras[1]+".jpg"
imgML_path=data_dir+'Run '+run_no+'/Run'+run_no+"-"+cameras[2]+".JPG"
imgM_path=data_dir+'Run '+run_no+'/Run'+run_no+"-"+cameras[3]+".JPG"

images = glob.glob(data_dir+'Run '+run_no+'/*')

valid_cameras=[]
images={}#save images
bin_edg_images={}#save bin_edge images
img_path_ls=[imgL_path,imgR_path,imgML_path,imgM_path]

for i in range(len(img_path_ls)) :
    if os.path.exists(img_path_ls[i]) ==False:
        print(" - Image Not Found for {} Camera, Image Processing will not run .".format(cameras[i]))
    else:
        img0=cv2.imread(imgL_path)
        img=cv2.imread(img_path_ls[i],0)
        images[cameras[i]]=img
        ## ---- [IP] Edge detection ----
        img_blur=cv2.GaussianBlur(img,(5,5),0)
        edges = cv2.Canny(image=img_blur, threshold1=canny_thresh[cameras[i]][0], threshold2=canny_thresh[cameras[i]][1]) # Canny Edge Detection
        bin_edg=edges[:,:]>0
        bin_edg=bin_edg.astype(float)  
        bin_edg_images[cameras[i]]=bin_edg
        valid_cameras.append(cameras[i])

##### Main Loop of image processing (IP)
figs={}# to save figs
ROI_loc={}
if valid_cameras:
    print("run={} pixel count......".format(run_no))
    for camera in valid_cameras:
        df_pixels=pd.DataFrame()
        print("---------{} Camera : Pixel Counts-------".format(camera))

        img=images[camera]

        ## ---- [IP] Edge detection ----
        bin_edg=bin_edg_images[camera]

        edge_y,edge_x=np.where(bin_edg==1)## find pixel location of edges (switch them)

        ##  ---- [IP] - Corner detection  ----

        nx,ny=6,4
        
        CHECKERBOARD = (nx,ny)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        gray=img
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny))
        print("Corner Detection works:",ret)

        ### ---- [IP] - Set ROI (Region of Interest)----

        # #### Set ROI automatically with relative position to checkboard
        if ret ==True:

            corners[:,:,0].min(),corners[:,:,0].max(),corners[:,:,1].min(),corners[:,:,1].max()

            corner_x_diff=np.diff(corners[:,:,0].flatten())[np.diff(corners[:,:,0].flatten())>np.quantile(abs(np.diff(corners[:,:,0].flatten())),0.25)]
            corner_y_diff=np.diff(corners[:,:,1].flatten())[np.diff(corners[:,:,1].flatten())>np.quantile(abs(np.diff(corners[:,:,1].flatten())),0.25)]
            corner_x_diff,corner_y_diff

            ROI_width=200
            shift_multiples=shift_par[camera]
            ROI_x_start,ROI_x_end=int(corners[:,:,0].max()+corner_x_diff.max()*shift_multiples),int(corners[:,:,0].max()+corner_x_diff.max()*shift_multiples+ROI_width)
            ROI_y_start,ROI_y_end=int(corners[:,:,1].min()-corner_y_diff.max()*shift_multiples),int(corners[:,:,1].max()+corner_y_diff.max()*shift_multiples)

        # #### Set ROI mannually
        else:
            ROI_x_start,ROI_x_end=2200,2500
            ROI_y_start,ROI_y_end=2800,3800

        ROI_loc[camera]=[ROI_x_start,ROI_x_end,ROI_y_start,ROI_y_end]

        ###  ---- [IP] -Fit lines to the corner points of checkerboard  ----

        print("Fit lines to corner points...")
        line1_pt,line2_pt,line3_pt,line4_pt=corners[:6],corners[6:12],corners[12:18],corners[18:]
        lines_a,lines_b=fit_linear_lines(line1_pt,line2_pt,line3_pt,line4_pt)

        lines_a.append(lines_a[3]+np.diff(lines_a).mean())## infer intercept for line 5
        lines_b.append(lines_b[3]+np.diff(lines_b).mean())## infer slop for line 5

        lines_a=[lines_a[0]-np.diff(lines_a).mean()]+lines_a## infer intercept for line 0
        lines_b=[lines_b[0]-np.diff(lines_b).mean()]+lines_b## infer slop for line 0

        ### Plot ROI area
        fig, ax = plt.subplots(1,figsize=(16,16))
        ax.imshow(img)
        x = np.linspace(corners[:,:,0].min()-100,corners[:,:,0].max()+500,10000)
        for a,b in zip(lines_a,lines_b):
            plt.plot(x,b*x+a, '-r', label='LINE')

        rect = patches.Rectangle((ROI_x_start, ROI_y_start), ROI_x_end-ROI_x_start, ROI_y_end-ROI_y_start, linewidth=2, edgecolor='b', facecolor='none')
        plt.text(ROI_x_start+50,ROI_y_start-10,"ROI",fontsize=15)
        ax.add_patch(rect)

        ### [OutPut] save ROI image

        # Check directory and create if not
        if os.path.exists(fig_path+camera+"/") ==False:
            os.mkdir(fig_path+camera+"/")
            print("Directory {} created".format(fig_path+camera+"/"))

        plt.savefig(fig_path+camera+"/"+"Run"+run_no+"-"+camera+"-ROI-Area.png")

        plt.close("all")

        # Filter the edge point in the ROI area
        ROI_edge_x=edge_x[np.where((edge_x>ROI_x_start)& (edge_x<ROI_x_end)&(edge_y>ROI_y_start)&(edge_y<ROI_y_end))]
        ROI_edge_y=edge_y[np.where((edge_x>ROI_x_start)& (edge_x<ROI_x_end)&(edge_y>ROI_y_start)&(edge_y<ROI_y_end))]
        if ROI_edge_x.size==0:
            print("--ERROR!!: No edges detected in ROI! No output created for this run......")
        else:
            print("Edges detected in ROI - Good :)")
            ROI_edge_x,ROI_edge_y


        ### ----[IP] Detect potential measurement location with fitting lines ----

            coor_2D=np.zeros((1, len(ROI_edge_x), 2), np.float32)

            min_dist1,min_dist2,min_dist3,min_dist4,min_dist5,min_dist6=100,100,100,100,100,100 # store the min_distance from lines to points
            for p in range(len(ROI_edge_x)):
                point_2D=np.array([[ROI_edge_x[p],ROI_edge_y[p]]])
                coor_2D[0][p]=np.array(point_2D[:2]).reshape(1,2)[0]

                ## Check if in lines(1,2,3,4,5,6)
                pred_y1=lines_a[0]+point_2D[0][0]*lines_b[0]
                pred_y2=lines_a[1]+point_2D[0][0]*lines_b[1]
                pred_y3=lines_a[2]+point_2D[0][0]*lines_b[2]
                pred_y4=lines_a[3]+point_2D[0][0]*lines_b[3]
                pred_y5=lines_a[4]+point_2D[0][0]*lines_b[4]
                pred_y6=lines_a[5]+point_2D[0][0]*lines_b[5]
            #     print(point_2D[0][0],point_2D[0][1],pred_y1,pred_y2,pred_y3,pred_y4,pred_y5)

                ## Chose the point that close to line as identified measurementlocation
                if abs(pred_y1-point_2D[0][1])< min_dist1:
                    loc1=point_2D
                    min_dist1=abs(pred_y1-point_2D[0][1])
                if abs(pred_y2-point_2D[0][1])< min_dist2:
                    loc2=point_2D
                    min_dist2=abs(pred_y2-point_2D[0][1])
                if abs(pred_y3-point_2D[0][1])< min_dist3:
                    loc3=point_2D
                    min_dist3=abs(pred_y3-point_2D[0][1])
                if abs(pred_y4-point_2D[0][1])< min_dist4:
                    loc4=point_2D
                    min_dist4=abs(pred_y4-point_2D[0][1])
                if abs(pred_y5-point_2D[0][1])< min_dist5:
                    loc5=point_2D
                    min_dist5=abs(pred_y5-point_2D[0][1])
                if abs(pred_y6-point_2D[0][1])< min_dist6:
                    loc6=point_2D
                    min_dist6=abs(pred_y6-point_2D[0][1])

            ### ----[IP] Filter the potential measure point according to y and brightness -----
            ROI_points=pd.DataFrame(coor_2D[0],columns=["x","y"])# all edege point in ROI region
            ROI_points=ROI_points.sort_values(by=["y","x"])

            candi_points={}
            pix_dist={}

            for y in [loc1[0][1],loc2[0][1],loc3[0][1],loc4[0][1],loc5[0][1],loc6[0][1]]:
                cand_x=ROI_points.loc[ROI_points["y"]==y,"x"].values
                cand_xs=[]
            #     candi_points[y]=ROI_points.loc[ROI_points["y"]==y,"x"].values

                ## filter out the candidate measurepoint _x based on the brightness value. 
                for i in cand_x:
                    if any(img[int(y),int(i-2):int(i+3)]<75):#brightness value<75
                        
                        cand_xs.append(i)
                candi_points[y]=np.array(cand_xs)

            locations=["A","B","C","D","E","F"]
            for (i,y) in zip(locations,candi_points.keys()):
                # For the detected total candi_points=0, nothing detected, then assign pixel_distance as 0 
                if candi_points[y].size==0:
                    pix_dist[i]=np.array([0])
                    candi_points[y]=np.array([0,0])#start=end.
                
                # For the detected total candi_points=1, then assign pixel_distance as 0 
                elif candi_points[y].size==1: 
                    pix_dist[i]=np.array([0])
                    candi_points[y]=np.array([candi_points[y][0],candi_points[y][0]])#start=end.

                elif candi_points[y].size==2:
                    pix_dist[i]=np.diff(candi_points[y])
                # For the detected total candi_points>2
                else:
                    if sum(np.diff(candi_points[y])) <=40:

                        pix_dist[i]=np.array([sum(np.diff(candi_points[y]))])
                        candi_points[y]=np.array([candi_points[y][0],candi_points[y][-1]])
                    else:
                        pos_id=np.where(np.diff(candi_points[y])==max(np.diff(candi_points[y])[np.diff(candi_points[y])<40]))[0][0]
                        pix_dist[i]=np.array([np.diff(candi_points[y])[pos_id]])
                        candi_points[y]=np.array([candi_points[y][pos_id],candi_points[y][pos_id-1]])

            pixel_out=pd.DataFrame.from_dict(candi_points,orient="index").reset_index()
            pixel_dist=pd.DataFrame.from_dict(pix_dist,orient="index",columns=["pix_dist"]).reset_index()
            pixel_out=pd.concat([pixel_out,pixel_dist],axis=1)

            pixel_out.columns=["y","x_1","x_2","loc","pix_dist"]
            pixel_out["run_no"]=int(run_no)
            pixel_out=pixel_out[["run_no","loc","y","x_1","x_2","pix_dist"]]

    #         print("{}-Camera, Run={}".format(camera,run_no))
            print(pixel_out)

            df_pixels=pd.concat([df_pixels,pixel_out],axis=0)

            ### [OutPut] save pixel count file for each run

            # #Check directory and create
            # if os.path.exists(csv_path+camera+"/") ==False:
            #     os.mkdir(csv_path+camera+"/")
            #     print("Directory {} created".format(csv_path+camera+"/"))

            # pixel_out.to_csv(csv_path+camera+"/"+"Run"+run_no+"-"+camera+"-pixels.csv")
            # print("Saved to "+csv_path+camera+"/"+"Run"+run_no+"-"+camera+"-pixels.csv")


        df_pixels["camera"]=camera
        df_pixels_all=pd.concat([df_pixels_all,df_pixels],axis=0).sort_values(by=["run_no","camera"]).reset_index(drop=True)

    ### [OutPut] save pixel counts
    df_pixels_all.to_csv(csv_path+"allcameras_pixels_count_fromImageProcess.csv",index=False)

    #### ---------------Gap & Flush Prediction-----------------------------

    df_pixels_new=df_pixels_all.loc[df_pixels_all["run_no"]==int(run_no),]
    if len(set(df_pixels_new.camera.unique()))==len(cameras):
        print("Predicting gap and flush for run ={}.........".format(run_no))
        pred_path=out_dir+"prediction/"

        df_pred_new=pred.pred_gap_flush(df_pixels_new,pred_path,run_no)
        print("--------------------------Run={} Gap&Fush Prediction ------------------------".format(run_no))
        print(df_pred_new)
        
         ####
        for camera in valid_cameras:
            plot_para=[img,bin_edg,df_pred_new,ROI_loc[camera]]
            df_pixes_camera=df_pixels_new.loc[df_pixels_new["camera"]==camera,].reset_index(drop=True)
            fig1=pr.plot_result(images[camera],bin_edg_images[camera],df_pixes_camera,df_pred_new,ROI_loc[camera])
            fig1.suptitle("Gap & Flush prediction:Run={},{} Camera".format(run_no,camera))

            ### [OutPut] save ROI Measurement points image
            #Check directory and create if not
            if os.path.exists(fig_path+camera+"/") ==False:
                os.mkdir(fig_path+camera+"/")
                print("Directory {} created".format(fig_path+camera+"/"))
            fig1.savefig(fig_path+camera+"/"+"Run"+run_no+"-"+camera+"-ROI-MeasurePoint.png")
            # fig1=plt.imread(fig_path+camera+"/"+"Run"+run_no+"-"+camera+"-ROI-MeasurePoint.png")
            # plt.show()
            plt.close("all")
            figs[camera]=fig1

        ##

    else:
        missed_camera=set(cameras)-set(set(df_pixels_new.camera.unique()))
        print("No pixel count from camera {},Gap and flush prediction will not run.".format(missed_camera)) 

    #### ---------------Control Chart-----------------------------
    gap_flush_pred_path=out_dir+"prediction/"+"gap_flush_prediction.csv"
    if os.path.exists(gap_flush_pred_path) ==True:
        df_pred=pd.read_csv(gap_flush_pred_path)
        if df_pred.shape[0]>0:
            for loc in df_pred["loc"].unique():
                df_pred_loc=df_pred[df_pred["loc"]==loc].sort_values(by="run_no")
                
                if df_pred_loc.shape[0]>=3:
                    y=df_pred_loc[["run_no","Gap_pred(mm)","Flush_pred(mm)"]]
                    y.set_index("run_no",inplace=True)
                    chart1=control_chart(y, alpha=0.01, legend_right=True);
                    chart2=univariate_control_chart(y, legend_right=True);     
                    chart1.figure.savefig(chart_path+loc+"-hotelling-chart.png")
                    # chart2.suptitle(loc)  
                    chart2.savefig(chart_path+loc+"-Uni-chart.png")
                else:
                    print("No control chart generated for loc{},since less than 3 samples".format(loc))       
    else:
        print("No control chart since no file found of {}".format(gap_flush_pred_path))

