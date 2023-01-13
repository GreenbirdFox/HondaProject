
## Compare to imagePre_newV1.py, the logic to detect measure point optimized with brightness value : from line 241
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
import py.pred_gap_flush as pred_bottom
# import py.plot_result as pr
import py.pred_gap_top as pred_top
import py.plot_result_top as pr_top
import py.plot_result as pr_bottom
from py.baweraopen import baweraopen

from hotelling.plots import control_chart, control_stats, univariate_control_chart
import warnings
# from py.ewma_control_chart import ewma_control_chart
from py.ewma import Ewma

warnings.filterwarnings('ignore', '.*masked element*', )


from tkinter import *   ##button
# from tkinter import messagebox

 #### ----------Function-----------------------------------------------
def ref_area_detect(template,image):
    ''' Find the location of reference area that match with template on image
    return
    -------
    pt:the left top corner index of detect ROI area.
    temp_w: width of template (or ROI)
    temp_h: height of template (or ROI)
        '''
    temp_w, temp_h = template.shape[::-1]
    res = cv2.matchTemplate(image,template,cv2.TM_CCOEFF_NORMED)
    threshold = np.max(res)
    print("Template Matching NCC=",threshold)
    loc = np.where( res >= threshold)
    pt=np.array([loc[1],loc[0]]).flatten()
    return (pt,temp_w, temp_h,threshold)

def ROI_area(pt,temp_w,temp_h,shift_x,shift_y,loc,camera):
    '''Function to find the ROI area according the refered template
    '''
    if loc in ["A","B"]:
        ROI_x_start,ROI_x_end= pt[0]+shift_x[camera][loc],pt[0]+temp_w
        ROI_y_start,ROI_y_end= pt[1]+shift_y[camera][loc],pt[1]+shift_y[camera][loc]+temp_h
    else:
        ROI_x_start,ROI_x_end= pt[0]+shift_x[camera][loc],pt[0]+temp_w
        ROI_y_start,ROI_y_end= pt[1]-int(0.5*temp_h)+shift_y[camera][loc],pt[1]+shift_y[camera][loc]+int(0.5*temp_h)
    return(ROI_x_start,ROI_x_end, ROI_y_start,ROI_y_end)

def filter_ROI_edge_pt(edge_x,edge_y,ROI_loc):
    '''funtion to filter the edge point in the ROI region
    edge_x,edge_y: edge points of binary edge images
    ROI_x_start,ROI_x_end,ROI_y_start,ROI_y_end: ROI loc
    '''
    ROI_x_start,ROI_x_end,ROI_y_start,ROI_y_end= ROI_loc
    ROI_edge_x=edge_x[np.where((edge_x>ROI_x_start)& (edge_x<ROI_x_end)&(edge_y>ROI_y_start)&(edge_y<ROI_y_end))]
    ROI_edge_y=edge_y[np.where((edge_x>ROI_x_start)& (edge_x<ROI_x_end)&(edge_y>ROI_y_start)&(edge_y<ROI_y_end))]
    if ROI_edge_x.size==0:
        print("--ERROR!!: No edges detected in ROI! No output created for this run......")
    else:
        print("Edges detected in ROI - Good :)")
        # print(ROI_edge_x,ROI_edge_y)
    return ROI_edge_x,ROI_edge_y

def filter_potential_mpt_x_based(ROI_edge_x,ROI_edge_y,locx,ROI_loc,bright_thresh,img):
    '''function to filter the potential measurment point from ROI edges points
    ROI_edge_x,ROI_edge_y:
    locy: locy_c, locy_d, locy_e
    ROI_loc: the location of ROI
    black_thresh: the brighness threshold for each ROI_black_open
    '''
    if ROI_edge_x.size>0:
        ROI_x_start,ROI_x_end,ROI_y_start,ROI_y_end=ROI_loc
        ROI_bright=(img[ROI_y_start:ROI_y_end,ROI_x_start:ROI_x_end]>bright_thresh).astype(float)# Middle 40, ML 30,Left-50
        plt.imshow(ROI_bright)
        ROI_bright_open=baweraopen(ROI_bright.astype(np.uint8),6000)

        coor_2D=np.zeros((1, len(ROI_edge_x), 2), np.float32)
        for p in range(len(ROI_edge_x)):
            point_2D=np.array([[ROI_edge_x[p],ROI_edge_y[p]]])
            coor_2D[0][p]=np.array(point_2D[:2]).reshape(1,2)[0]

        ROI_points=pd.DataFrame(coor_2D[0],columns=["x","y"])# all edege point in ROI region
        ROI_points=ROI_points.sort_values(by=["x","y"])
        ROI_points
        ## filter pontential mpt_x ,if there is less than 2 edge points find then move y and search 40 neighbors of x.
        locys=[]
        shift=0
        locx_neb=locx+shift
        while (sum(np.diff(np.array(locys)))<1 or len(locys)<2) and abs(shift)<100 :
            locx_neb=max(0,locx+shift)
            mpt_ys=ROI_points.loc[ROI_points["x"]==locx_neb,"y"].values
            # print(locx_neb,mpt_ys)
            
            locys=[]
            for i in mpt_ys:
                # filter out first line the candidate measurepoint _x based on the brightness value. 
                if any(ROI_bright_open[int(i-ROI_y_start-10):int(i-ROI_y_start),int(locx_neb-ROI_x_start),]>0):
                    if len(locys)>0:
                        if i <locys[-1]+25: #if multiple line detected, save the last one
                            locys[-1]=i
                        else:
                            pass
                    else:
                        locys.append(i)
                # disgard second line the candidate measurepoint _x based on the brightness value. if the 35 neighbors not in bright image.
                elif any(ROI_bright_open[int(i-ROI_y_start-55):int(i-ROI_y_start+1),int(locx_neb-ROI_x_start),]>0):#middle/middle Left=35
                    if len(locys)>0 and i >locys[-1]+40:#30
                        locys.append(i)
                else:
                    pass
                        
            shift=-shift+(shift<=0)
            # print("locys",locys)
#             print("sum",sum(np.diff(np.array(locys))))
        locx_final= locx_neb
        cand_ys=locys
        return(locx_final,np.array(cand_ys))
    else:
        return(0,np.array([0]))
def filter_potential_mpt_y_based(ROI_edge_x,ROI_edge_y,locy,ROI_loc,bright_thresh,img):
    '''function to filter the potential measurment point from ROI edges points
    ROI_edge_x,ROI_edge_y:
    locy: locy_c, locy_d, locy_e
    ROI_loc: the location of ROI
    black_thresh: the brighness threshold for each ROI_black_open
    '''
    if ROI_edge_x.size>0:
        ROI_x_start,ROI_x_end,ROI_y_start,ROI_y_end=ROI_loc
        ROI_black=(img[ROI_y_start:ROI_y_end,ROI_x_start:ROI_x_end]<bright_thresh).astype(float)
        ROI_black_open=baweraopen(ROI_black.astype(np.uint8),6000)

        coor_2D=np.zeros((1, len(ROI_edge_x), 2), np.float32)
        for p in range(len(ROI_edge_x)):
            point_2D=np.array([[ROI_edge_x[p],ROI_edge_y[p]]])
            coor_2D[0][p]=np.array(point_2D[:2]).reshape(1,2)[0]

        ROI_points=pd.DataFrame(coor_2D[0],columns=["x","y"])# all edege point in ROI region
        ROI_points=ROI_points.sort_values(by=["y","x"])
        ROI_points
        ## filter pontential mpt_y ,if there is less than 2 edge points find then move y and search 20 neighbors of y.
        locxs=[]
        shift=0
        locy_neb=locy+shift
        while (sum(np.diff(np.array(locxs)))<30 or len(locxs)<2) and abs(shift)<40 :
            locy_neb=locy+shift
            mpt_xs=ROI_points.loc[ROI_points["y"]==locy_neb,"x"].values
            # print(locy_neb,mpt_xs)
            
            locxs=[]
            for i in mpt_xs:
                # filter out the candidate measurepoint _x based on the brightness value. 
                if any(ROI_black_open[int(locy_neb-ROI_y_start),int(i-ROI_x_start-2):int(i-ROI_x_start+11)]>0):#left edge  ### Change
                    if len(locxs)>0:
                        if i <locxs[-1]+15: #if multiple line detected, save the last one
                            locxs[-1]=i
                        else:
                            locxs.append(i)
                    else:
                        locxs.append(i)
                if any(ROI_black_open[int(locy_neb-ROI_y_start),int(i-ROI_x_start-14):int(i-ROI_x_start+2)]>0):#right edge  ### Change
                    if len(locxs)>0:
                        if i >locxs[-1]+25: #if multiple line detected, save the last one
                            locxs.append(i)
                    else:
                        pass

            shift=-shift+(shift<=0)
            # print("after",locxs)
            # print("sum",sum(np.diff(np.array(locxs))))
            # print(shift)
            
        locy_final= locy_neb
        cand_xs=locxs
        return(locy_final,np.array(cand_xs))
    else:
        return(0,np.array([0]))


######------------- Main functions---------------
def imageprocess_top(run_no):
    # print(run_no)
    ### Need set
    data_dir='pictures/TopCamera/'
    out_dir='output/' ## Need set

    # run_no='1'#sys.argv[1]
    # run_no="15"


    #### ---------Check and create directory to store outputs.--------------
    fig_dir='fig/'
    fig_path= os.path.join(out_dir, fig_dir)
    if os.path.exists(fig_path) ==False:
        os.mkdir(fig_path)
        print("Directory '%s' created" %fig_path)

    csv_dir= 'pixelscount/'
    csv_path= os.path.join(out_dir, csv_dir)
    if os.path.exists(csv_path) ==False:
        os.mkdir(csv_path)
        print("Directory '%s' created" %csv_path)

    chart_dir='controlchart/'
    chart_path= os.path.join(out_dir, chart_dir)
    if os.path.exists(chart_path) ==False:
        os.mkdir(chart_path)
        print("Directory '%s' created" %chart_path)


    #### ----------Parameters ---------------------------------------------

    canny_thresh={"Right":[20,40],"Left":[30,60],"Middle Left":[40,80],"Middle":[30,80]}

    bright_thresh={"Right":{"A":100,"B":70,"C":60,"D":70,"E":70},
              "Left":{"A":150,"B":70,"C":50,"D":70,"E":70},
              "Middle":{"A":120,"B":70,"C":40,"D":50,"E":50},
              "Middle Left":{"A":160,"B":70,"C":30,"D":50,"E":50}}


    #### --------- Main -----------------------------------------------------

    ## Create data frame or csv to store result
    pix_counts_file=csv_path+"allcameras_pixels_count_top2_fromImageProcess.csv"
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
    imgL_path=data_dir+'/run-'+run_no+"-3.jpg"
    imgR_path=data_dir+'/run-'+run_no+"-2.jpg"
    imgML_path=data_dir+'/run-'+run_no+"-0.jpg"
    imgM_path=data_dir+'/run-'+run_no+"-1.jpg"

    images = glob.glob(data_dir+'Run'+run_no+'/*')

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
            sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3) # Sobel Edge Detection on the Y axis
            bin_edg=((sobely[:,:]>12)).astype(float)
            # edges = cv2.Canny(image=img_blur, threshold1=canny_thresh[cameras[i]][0], threshold2=canny_thresh[cameras[i]][1]) # Canny Edge Detection
            # bin_edg=(edges[:,:]>0).astype(float)
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
            img_blur=cv2.GaussianBlur(img,(7,7),0)
            img_dark=(img_blur<75).astype(float)
            img_dark_open=baweraopen(img_dark.astype(np.uint8),300)
            # plt.imshow(img_dark_open)
            # plt.show()

            ##  ---- [IP] - Template matching  ----

            template1=cv2.imread(out_dir+"template/template-TopGap-"+camera+".png",0)
            subtemplate1=cv2.imread(out_dir+"template/template-TopGapSubtemplate-"+camera+".png",0)
            template2=cv2.imread(out_dir+"template/template-Step-"+camera+".png",0)

            pt1,temp_w1, temp_h1,NCC1=ref_area_detect(template1,img)
            pt2,temp_w2, temp_h2,NCC2=ref_area_detect(template2,img)


            fig, ax = plt.subplots(1,figsize=(16,16))
            ax.imshow(img)
            rect = patches.Rectangle((pt1[0], pt1[1]),temp_w1, temp_h1, linewidth=1, edgecolor='b', facecolor='none')
            ax.add_patch(rect); ax.text(pt1[0], pt1[1],"ref-A NCC="+str(round(NCC1,4)),fontsize=14)
            rect = patches.Rectangle((pt2[0], pt2[1]),temp_w2, temp_h2, linewidth=1, edgecolor='b', facecolor='none')
            ax.add_patch(rect); ax.text(pt2[0], pt2[1]+100,"ref-B NCC="+str(round(NCC2,4)),fontsize=14)


            ### ---- [IP] - Set ROI (Region of Interest)----

            # #### Set ROI automatically with relative position to template
            shift_x={"Right":{"A":0,"B":0,"C":0,"D":100,"E":0},
                      "Left":{"A":0,"B":0,"C":0,"D":100,"E":0},
                      "Middle":{"A":0,"B":0,"C":0,"D":100,"E":0},
                      "Middle Left":{"A":0,"B":0,"C":100,"D":200,"E":0}}

            shift_y={"Right":{"A":0,"B":0,"C":200,"D":1150,"E":0},
                      "Left":{"A":0,"B":0,"C":0,"D":1150,"E":0},
                      "Middle":{"A":0,"B":0,"C":0,"D":1050,"E":0},
                      "Middle Left":{"A":0,"B":0,"C":0,"D":1060,"E":0}}

            ROI_x_start1,ROI_x_end1,ROI_y_start1,ROI_y_end1=ROI_area(pt1,temp_w1,temp_h1,shift_x,shift_y,"A",camera)
            ROI_x_start2,ROI_x_end2,ROI_y_start2,ROI_y_end2=ROI_area(pt2,temp_w2,temp_h2,shift_x,shift_y,"B",camera)

            # #### Set ROI mannually

            # ROI_x_start,ROI_x_end=2200,2500
            # ROI_y_start,ROI_y_end=2800,3800

            ### Plot ROI area
            fig, ax = plt.subplots(1,figsize=(16,16))
            ax.imshow(img)
            rect1 = patches.Rectangle((ROI_x_start1, ROI_y_start1), ROI_x_end1-ROI_x_start1
                                     , ROI_y_end1-ROI_y_start1, linewidth=1, edgecolor='b', facecolor='none')

            rect2 = patches.Rectangle((ROI_x_start2, ROI_y_start2), ROI_x_end2-ROI_x_start2
                                     , ROI_y_end2-ROI_y_start2, linewidth=1, edgecolor='r', facecolor='none')

            ax.add_patch(rect1)
            ax.text(ROI_x_start1, ROI_y_start1,"ROI-A",fontsize=14)
            ax.add_patch(rect2)
            ax.text(ROI_x_start2, ROI_y_start2,"ROI-B",fontsize=14)
            # plt.show()

            ### For Measurement Point A

            ROI_img1=img[ROI_y_start1:ROI_y_end1,ROI_x_start1:ROI_x_end1]
            pt11,temp_w11, temp_h11,NCC11=ref_area_detect(subtemplate1,ROI_img1)

            ROI_x_start1,ROI_x_end1,ROI_y_start1,ROI_y_end1=ROI_x_start1,ROI_x_end1,ROI_y_start1,ROI_y_start1+pt11[1]

            ROI_x_start,ROI_x_end=min(ROI_x_start1,ROI_x_start2),max(ROI_x_end1,ROI_x_end2)
            ROI_y_start,ROI_y_end=min(ROI_y_start1,ROI_y_start2),max(ROI_y_end1,ROI_y_end2)
            ROI_loc[camera]=[ROI_x_start,ROI_x_end,ROI_y_start,ROI_y_end]

            if camera=="Right":
                locx_a=ROI_x_start1+0
            else:
                locx_a=ROI_x_start1+300

            plt.axvline(locx_a,color='red')

            ### [OutPut] save ROI image

            # Check directory and create if not
            if os.path.exists(fig_path+camera+"/") ==False:
                os.mkdir(fig_path+camera+"/")
                print("Directory {} created".format(fig_path+camera+"/"))

            plt.savefig(fig_path+camera+"/"+"Run"+run_no+"-"+camera+"-Top2-ROI-Area.png")
            plt.close("all")

            # Filter the edge point in the ROI area
            ROI_loc_a=(ROI_x_start1,ROI_x_end1,ROI_y_start1,ROI_y_end1)
            ROI_edge_x1,ROI_edge_y1=filter_ROI_edge_pt(edge_x,edge_y,ROI_loc_a)


            ### ----[IP] Filter the potential measure point according to y and brightness ----

            locx_a,cand_ys_a=filter_potential_mpt_x_based(ROI_edge_x1,ROI_edge_y1,locx_a,ROI_loc_a,bright_thresh[camera]["A"],img)

            candi_points={locx_a:cand_ys_a}

            pix_dist={}
            locations=["A"]
            for (i,x) in zip(locations,candi_points.keys()):
            #     print(i,candi_points[y].size,candi_points[y])
                
                # For the detected total candi_points=0, nothing detected, then assign pixel_distance as 0 
                if candi_points[x].size==0:
                    pix_dist[i]=np.array([0])
                    candi_points[x]=np.array([0,0])#start=end.
                
                # For the detected total candi_points=1, then assign pixel_distance as 0 
                elif candi_points[x].size==1: 
                    pix_dist[i]=np.array([0])
                    candi_points[x]=np.array([candi_points[x][0],candi_points[x][0]])#start=end.
                    
                elif candi_points[x].size==2:
                    pix_dist[i]=np.diff(candi_points[x])
                # For the detected total candi_points>2
                else:
                    if sum(np.diff(candi_points[x])) <=50:##35 run=35 ml-
            #             print(i,candi_points[y])
                        pix_dist[i]=np.array([sum(np.diff(candi_points[x]))])
                        candi_points[x]=np.array([candi_points[x][0],candi_points[x][-1]])
                    else:
                        # print(">50:")
                        # print(i,candi_points[x])
                        pos_id=np.where(np.diff(candi_points[x])==max(np.diff(candi_points[x])[np.diff(candi_points[x])<150]))[0][0]
                        pix_dist[i]=np.array([np.diff(candi_points[x])[pos_id]])
                        # print(pos_id)
                        candi_points[x]=np.array([candi_points[x][pos_id],candi_points[x][pos_id+1]])
                        # print(i,candi_points[x])

            #[out]-save result for measurement point A 
            pixel_out1=pd.DataFrame.from_dict(candi_points,orient="index").reset_index()
            pixel_dist1=pd.DataFrame.from_dict(pix_dist,orient="index",columns=["pix_dist"]).reset_index()
            pixel_out1=pd.concat([pixel_out1,pixel_dist1],axis=1)
            pixel_out1.columns=["x_1","y_1","y_2","loc","pix_dist"]
            pixel_out1["x_2"]=pixel_out1["x_1"]
            pixel_out1=pixel_out1[["loc","x_1","y_1","x_2","y_2","pix_dist"]]

            ## For measurement point B

            ROI_img_blur2=img_blur[ROI_y_start2:ROI_y_end2,ROI_x_start2:ROI_x_end2]
            sobelx = cv2.Sobel(src=ROI_img_blur2, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3) # Sobel Edge Detection on the X axis
            sobely = cv2.Sobel(src=ROI_img_blur2, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3) # Sobel Edge Detection on the Y axis
            sobely_edg=((sobely[:,:]>12)&(sobelx[:,:]<15)).astype(float)
            sobely_edg_open=baweraopen(sobely_edg.astype(np.uint8),3000)

            ## Find connected component of left edge and right edge
            nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(sobely_edg_open)
            # print(nlabels,np.unique(labels),stats,centroids)
            box1=stats[1,:4]
            box2=stats[2,:4]

            if box1[0]<box2[0]:
                left_box=box1
                right_box=box2
                left_edge_y,left_edge_x=np.where(labels==1)
                right_edge_y,right_edge_x=np.where(labels==2)
            else:
                left_box=box2
                right_box=box1
                left_edge_y,left_edge_x=np.where(labels==2)
                right_edge_y,right_edge_x=np.where(labels==1)

            ROI_edges=sobely_edg_open
            fig, ax = plt.subplots(1,figsize=(16,16))
            ax.imshow(sobely_edg_open)
            rect1 = patches.Rectangle((left_box[0], left_box[1]), left_box[2]
                                     ,  left_box[3], linewidth=1, edgecolor='b', facecolor='none')
            rect2 = patches.Rectangle((right_box[0], right_box[1]), right_box[2]
                                     ,  right_box[3], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect1)
            ax.add_patch(rect2)

            ## Fit lines to the sobel edge
            lines = cv2.HoughLinesP(ROI_edges,1,np.pi/180,20,minLineLength=150,maxLineGap=30)
            print("lines found in sobel of B location",len(lines))
            valid_lines_left=[]
            valid_lines_right=[]
            i = 0
            for i in range(lines.shape[0]):
                for x1,y1,x2,y2 in lines[i]:
                    i+=1
                    slop=(y2-y1)/(x2-x1)
                    if abs(slop)<0.3 and abs(slop)>=0:
                        plt.plot([x1,x2],[y1,y2],'r')
            #             print(slop)
                        plt.text(x1,y1,str(i),bbox=dict(facecolor='red', alpha=0.5))
                    if x2<=left_box[0]+left_box[2]:
                        valid_lines_left.append([x1,y1,x2,y2])
                    else:
                        valid_lines_right.append([x1,y1,x2,y2])

            if os.path.exists(fig_path+camera+"/") ==False:
                os.mkdir(fig_path+camera+"/")
                print("Directory {} created".format(fig_path+camera+"/"))

            plt.savefig(fig_path+camera+"/"+"Run"+run_no+"-"+camera+"-Top2-Step-ROI-Area.png")
            plt.close("all")


            x_min_left,x_max_left=np.min(np.array(valid_lines_left)[:,[0,2]]),np.max(np.array(valid_lines_left)[:,[0,2]])
            x_min_right,x_max_right=np.min(np.array(valid_lines_right)[:,[0,2]]),np.max(np.array(valid_lines_right)[:,[0,2]])

            y_min_left,y_max_left=np.min(np.array(valid_lines_left)[:,[1,3]]),np.max(np.array(valid_lines_left)[:,[1,3]])
            y_min_right,y_max_right=np.min(np.array(valid_lines_right)[:,[1,3]]),np.max(np.array(valid_lines_right)[:,[1,3]])

            ## Filter the edge point in the ROI region
            count_left_edge_x=left_edge_x[np.where((left_edge_x>=left_box[0]+left_box[2]-25)& (left_edge_x<=left_box[0]+left_box[2])
                                         & (left_edge_y<=y_max_left+5)& (left_edge_y>=y_min_left-5))]
            count_left_edge_y=left_edge_y[np.where((left_edge_x>=left_box[0]+left_box[2]-20)&(left_edge_x<=left_box[0]+left_box[2])
                                                  & (left_edge_y<=y_max_left+5)& (left_edge_y>=y_min_left-5))]
            if count_left_edge_x.size==0:
                print("ERROR: No edges detected in Left!!!---")
            else:
                print("Good: Edges detected in left!")
                # print(count_left_edge_x,count_left_edge_y)

            centroid_left_x=np.mean(count_left_edge_x)+ROI_x_start2
            centroid_left_y=np.mean(count_left_edge_y)+ROI_y_start2

            ## Filter the edge point in the ROI region
            count_right_edge_x=right_edge_x[np.where((right_edge_x>=right_box[0])& (right_edge_x<=right_box[0]+20)
                                                    & (right_edge_y<=y_max_right+5)& (right_edge_y>=y_min_right-5))]
            count_right_edge_y=right_edge_y[np.where((right_edge_x>=right_box[0])&(right_edge_x<=right_box[0]+20)
                                                    & (right_edge_y<=y_max_right+5)& (right_edge_y>=y_min_right-5))]
            if count_right_edge_x.size==0:
                print("ERROR: No edges detected in right!!!---")
            else:
                print("Good: Edges detected in right!")
                # print(count_right_edge_x,count_right_edge_y)

            centroid_right_x=np.mean(count_right_edge_x)+ROI_x_start2
            centroid_right_y=np.mean(count_right_edge_y)+ROI_y_start2
            pix_dist_y=centroid_right_y-centroid_left_y

            #[out]-save result for measurement point B    
            pix_dist={}
            pix_dist["B"]=[centroid_left_x,centroid_left_y
                           ,centroid_right_x,centroid_right_y,pix_dist_y]
            pixel_out2=pd.DataFrame.from_dict(pix_dist,orient="index").reset_index()
            pixel_out2.columns=["loc","x_1","y_1","x_2","y_2","pix_dist"]


            pixel_out=pd.concat([pixel_out1,pixel_out2])
            pixel_out["run_no"]=int(run_no)
            pixel_out=pixel_out[["run_no","loc","x_1","y_1","x_2","y_2","pix_dist"]]
            
            print("{}-Camera, Run={}".format(camera,run_no))
            print(pixel_out)
            print("\n")
            df_pixels=pd.concat([df_pixels,pixel_out],axis=0)

            ### [OutPut] save pixel count file for each run

            #Check directory and create
            if os.path.exists(csv_path+camera+"/") ==False:
                os.mkdir(csv_path+camera+"/")
                print("Directory {} created".format(csv_path+camera+"/"))

            pixel_out.to_csv(csv_path+camera+"/"+"Run"+run_no+"-"+camera+"Top2-pixels.csv")
            print("Saved to "+csv_path+camera+"/"+"Run"+run_no+"-"+camera+"Top2-pixels.csv")


            df_pixels["camera"]=camera
            df_pixels_all=pd.concat([df_pixels_all,df_pixels],axis=0).sort_values(by=["run_no","camera"]).reset_index(drop=True)

        ### [OutPut] save pixel counts
        df_pixels_all.to_csv(csv_path+"allcameras_pixels_count_top2_fromImageProcess.csv",index=False)

        #### ---------------Gap & Flush Prediction-----------------------------

        df_pixels_new=df_pixels_all.loc[df_pixels_all["run_no"]==int(run_no),]
        # if len(set(df_pixels_new.camera.unique()))==len(cameras):
        print("Predicting gap and flush for run ={}.........".format(run_no))
        pred_path=out_dir+"prediction/"
        pred_csv="gap_flush_prediction.csv"
        print(df_pixels_new)

        df_pred_new=pred_top.pred_gap_top2(df_pixels_new,pred_path,run_no,pred_csv)
        print("--------------------------Run={} Gap&Fush Prediction ------------------------".format(run_no))
        print(df_pred_new)
        print("\n")
        df_pred_new.to_csv(pred_path+"pred_gap_flush_run"+run_no+"_top2.csv")


         #### [OUTPUT] PLOT ROI
        for camera in valid_cameras:
            plot_para=[img,bin_edg,df_pred_new,ROI_loc[camera]]
            df_pixes_camera=df_pixels_new.loc[df_pixels_new["camera"]==camera,].reset_index(drop=True)
            fig1=pr_top.plot_result_top(images[camera],bin_edg_images[camera],df_pixes_camera,df_pred_new,ROI_loc[camera])
            fig1.suptitle("Gap & Flush prediction:Run={},{} Camera".format(run_no,camera))

            ### [OutPut] save ROI Measurement points image
            #Check directory and create if not
            if os.path.exists(fig_path+camera+"/") ==False:
                os.mkdir(fig_path+camera+"/")
                print("Directory {} created".format(fig_path+camera+"/"))
            fig1.savefig(fig_path+camera+"/"+"Run"+run_no+"-"+camera+"-Top2-ROI-MeasurePoint.png")
            # fig1=plt.imread(fig_path+camera+"/"+"Run"+run_no+"-"+camera+"-ROI-MeasurePoint.png")
            # plt.show()
            plt.close("all")
            figs[camera]=fig1

            ##


        # else:
        #     missed_camera=set(cameras)-set(set(df_pixels_new.camera.unique()))
        #     df_pred_new=pd.DataFrame()
        #     print("No pixel count from camera {},Gap and flush prediction will not run.".format(missed_camera))


     #### ---------------Control Chart-----------------------------
    gap_flush_pred_path=out_dir+"prediction/"+pred_csv
    if os.path.exists(gap_flush_pred_path) ==True:
        df_pred=pd.read_csv(gap_flush_pred_path)
        if df_pred.shape[0]>0:
            for loc in df_pred["loc"].unique():
                df_pred_loc=df_pred[df_pred["loc"]==loc].sort_values(by="run_no")
                
                if df_pred_loc.shape[0]>=3:
                    y=df_pred_loc[["run_no","Gap_pred(mm)"]]

                    y.set_index("run_no",inplace=True)
                    chart1=control_chart(y, alpha=0.01, legend_right=True);
                    plt.xticks(range(len(y.index)),labels=y.index)
                    chart1.figure.savefig(chart_path+loc+"-Top2-hotelling-chart.png")

                    # chart2=univariate_control_chart(y, legend_right=True);   
                    # # chart2.suptitle(loc)  
                    # chart2.savefig(chart_path+loc+"-Uni-chart.png")

                    ewma_object=Ewma()
                    chart3=ewma_object.ewma_control_chart(y);   
                    # chart2.suptitle(loc)  
                    chart3.savefig(chart_path+loc+"-Top2-EWMA-chart.png")

                else:
                    print("No control chart generated for loc{},since less than 3 samples".format(loc))       
    else:
        print("No control chart since no file found of {}".format(gap_flush_pred_path))
    
    return(df_pred_new)


def imageprocess_bottom(run_no):
    # print(run_no)
    ### Need set
    data_dir='pictures/BottomCamera/'
    out_dir='output/' ## Need set

    # run_no='1'#sys.argv[1]
    # run_no="15"


    #### ---------Check and create directory to store outputs.--------------
    fig_dir='fig/'
    fig_path= os.path.join(out_dir, fig_dir)
    if os.path.exists(fig_path) ==False:
        os.mkdir(fig_path)
        print("Directory '%s' created" %fig_path)

    csv_dir= 'pixelscount/'
    csv_path= os.path.join(out_dir, csv_dir)
    if os.path.exists(csv_path) ==False:
        os.mkdir(csv_path)
        print("Directory '%s' created" %csv_path)

    chart_dir='controlchart/'
    chart_path= os.path.join(out_dir, chart_dir)
    if os.path.exists(chart_path) ==False:
        os.mkdir(chart_path)
        print("Directory '%s' created" %chart_path)

    pred_dir='prediction/'
    pred_path= os.path.join(out_dir, pred_dir)
    if os.path.exists(pred_path) ==False:
        os.mkdir(pred_path)
        print("Directory '%s' created" %chart_path)


    #### ----------Parameters ---------------------------------------------

    canny_thresh={"Right":[20,40],"Left":[20,40],"Middle Left":[25,50],"Middle":[25,50]}
    # shift_par={"Right":2.8,"Left":2.05,"Middle Left":2.2,"Middle":2.4}

    bright_thresh={"Right":{"C":60,"D":70,"E":70},
              "Left":{"C":50,"D":70,"E":70},
              "Middle":{"C":40,"D":50,"E":50},
              "Middle Left":{"C":30,"D":50,"E":50}}


    #### ----------Function-----------------------------------------------

    #### --------- Main -----------------------------------------------------

    ## Create data frame or csv to store result
    pix_counts_file=csv_path+"allcameras_pixels_count_bottom_fromImageProcess.csv"
    if os.path.exists(pix_counts_file) ==True:
        df_pixels_all=pd.read_csv(pix_counts_file)
    else:
        df_pixels_all=pd.DataFrame()

    if df_pixels_all.shape[0]>0 and (int(run_no) in df_pixels_all["run_no"].unique()):
        print("Warning: The pixel count result is avaliable for run= {}, it will be replaced.".format(run_no))
        df_pixels_all=df_pixels_all[df_pixels_all["run_no"]!=int(run_no)].copy(deep=True)
    else:
        print("Pixel counting for run ={}, Check if the image have been created from cameras...".format(run_no))

    ## Set Image directory of 4 Cameras
    cameras=["Left","Right","Middle Left","Middle"]
    imgL_path=data_dir+'/run-'+run_no+"-3.jpg"
    imgR_path=data_dir+'/run-'+run_no+"-2.jpg"
    imgML_path=data_dir+'/run-'+run_no+"-0.jpg"
    imgM_path=data_dir+'/run-'+run_no+"-1.jpg"

    images = glob.glob(data_dir+'Run'+run_no+'/*')

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
            img_blur=cv2.GaussianBlur(img,(7,7),0)
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
            img_blur=cv2.GaussianBlur(img,(7,7),0)
            img_dark=(img_blur<75).astype(float)
            img_dark_open=baweraopen(img_dark.astype(np.uint8),300)
            # plt.imshow(img_dark_open)
            # plt.show()

            ##  ---- [IP] - Template matching  ----

            template1=cv2.imread(out_dir+"template/template-bottom-C"+camera+".png",0)
            template2=cv2.imread(out_dir+"template/template-bottom-D"+camera+".png",0)
            template3=cv2.imread(out_dir+"template/template-bottom-E"+camera+".png",0)

            pt1,temp_w1, temp_h1,NCC1=ref_area_detect(template1,img)
            pt2,temp_w2, temp_h2,NCC2=ref_area_detect(template2,img)
            pt3,temp_w3, temp_h3,NCC3=ref_area_detect(template3,img)


            fig, ax = plt.subplots(1,figsize=(16,16))
            ax.imshow(img)
            rect = patches.Rectangle((pt1[0], pt1[1]),temp_w1, temp_h1, linewidth=1, edgecolor='b', facecolor='none')
            ax.add_patch(rect); ax.text(pt1[0], pt1[1],"ref-C")
            rect = patches.Rectangle((pt2[0], pt2[1]),temp_w2, temp_h2, linewidth=1, edgecolor='b', facecolor='none')
            ax.add_patch(rect); ax.text(pt2[0], pt2[1],"ref-D")
            rect = patches.Rectangle((pt3[0], pt3[1]),temp_w3, temp_h3, linewidth=1, edgecolor='b', facecolor='none')
            ax.add_patch(rect); ax.text(pt3[0], pt3[1],"ref-E")


            ### ---- [IP] - Set ROI (Region of Interest)----

            # #### Set ROI automatically with relative position to template
            shift_x={"Right":{"C":200,"D":150,"E":0},
                      "Left":{"C":0,"D":100,"E":0},
                      "Middle":{"C":300,"D":100,"E":0},
                      "Middle Left":{"C":500,"D":200,"E":0}}

            shift_y={"Right":{"C":200,"D":1350,"E":0},
                      "Left":{"C":0,"D":1150,"E":0},
                      "Middle":{"C":0,"D":1050,"E":0},
                      "Middle Left":{"C":0,"D":1060,"E":0}}

            ROI_x_start1,ROI_x_end1,ROI_y_start1,ROI_y_end1=ROI_area(pt1,temp_w1,temp_h1,shift_x,shift_y,"C",camera)
            ROI_x_start2,ROI_x_end2,ROI_y_start2,ROI_y_end2=ROI_area(pt2,temp_w2,temp_h2,shift_x,shift_y,"D",camera)
            ROI_x_start3,ROI_x_end3,ROI_y_start3,ROI_y_end3=ROI_area(pt3,temp_w3,temp_h3,shift_x,shift_y,"E",camera)

            # #### Set ROI mannually

            # ROI_x_start,ROI_x_end=2200,2500
            # ROI_y_start,ROI_y_end=2800,3800
            ROI_x_start,ROI_x_end=min(ROI_x_start1,ROI_x_start2,ROI_x_start3),max(ROI_x_end1,ROI_x_end2,ROI_x_end3)
            ROI_y_start,ROI_y_end=min(ROI_y_start1,ROI_y_start2,ROI_x_start3),max(ROI_y_end1,ROI_y_end2,ROI_y_end3)
            ROI_loc[camera]=[ROI_x_start,ROI_x_end,ROI_y_start,ROI_y_end]


            ### Plot ROI area
            fig, ax = plt.subplots(1,figsize=(16,16))
            ax.imshow(img)
            rect1 = patches.Rectangle((ROI_x_start1, ROI_y_start1), ROI_x_end1-ROI_x_start1
                                     , ROI_y_end1-ROI_y_start1, linewidth=1, edgecolor='b', facecolor='none')
            rect2 = patches.Rectangle((ROI_x_start2, ROI_y_start2), ROI_x_end2-ROI_x_start2
                                     , ROI_y_end2-ROI_y_start2, linewidth=1, edgecolor='r', facecolor='none')
            rect3 = patches.Rectangle((ROI_x_start3, ROI_y_start3), ROI_x_end3-ROI_x_start3
                                     , ROI_y_end3-ROI_y_start3, linewidth=1, edgecolor='g', facecolor='none')
            ax.add_patch(rect1)
            ax.add_patch(rect2)
            ax.add_patch(rect3)

            locy_c,locy_d,locy_e=pt1[1]+shift_y[camera]["C"],pt2[1]+shift_y[camera]["D"],pt3[1]+shift_y[camera]["E"]
            locys=[locy_c,locy_d,locy_e]
            for y in locys:
                plt.axhline(y,color='red')

            ### [OutPut] save ROI image

            # Check directory and create if not
            if os.path.exists(fig_path+camera+"/") ==False:
                os.mkdir(fig_path+camera+"/")
                print("Directory {} created".format(fig_path+camera+"/"))

            plt.savefig(fig_path+camera+"/"+"Run"+run_no+"-"+camera+"-bottom-ROI-Area.png")
            plt.close("all")

            # Filter the edge point in the ROI area
            ROI_loc_c=(ROI_x_start1,ROI_x_end1,ROI_y_start1,ROI_y_end1)
            ROI_loc_d=(ROI_x_start2,ROI_x_end2,ROI_y_start2,ROI_y_end2)
            ROI_loc_e=(ROI_x_start3,ROI_x_end3,ROI_y_start3,ROI_y_end3)
            ROI_edge_x1,ROI_edge_y1=filter_ROI_edge_pt(edge_x,edge_y,ROI_loc_c)
            ROI_edge_x2,ROI_edge_y2=filter_ROI_edge_pt(edge_x,edge_y,ROI_loc_d)
            ROI_edge_x3,ROI_edge_y3=filter_ROI_edge_pt(edge_x,edge_y,ROI_loc_e)


            ### ----[IP] Filter the potential measure point according to y and brightness ----
            locy_c,cand_xs_c=filter_potential_mpt_y_based(ROI_edge_x1,ROI_edge_y1,locy_c,ROI_loc_c,bright_thresh[camera]["C"],img)# Middle 40, ML 30,Left-50
            locy_d,cand_xs_d=filter_potential_mpt_y_based(ROI_edge_x2,ROI_edge_y2,locy_d,ROI_loc_d,bright_thresh[camera]["D"],img)#50
            locy_e,cand_xs_e=filter_potential_mpt_y_based(ROI_edge_x3,ROI_edge_y3,locy_e,ROI_loc_e,bright_thresh[camera]["E"],img)#50

            candi_points={locy_c:cand_xs_c,locy_d:cand_xs_d,locy_e:cand_xs_e}

            pix_dist={}
            locations=["C","D","E"]
            for (i,y) in zip(locations,candi_points.keys()):
            #     print(i,candi_points[y].size,candi_points[y])
                
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
                    if sum(np.diff(candi_points[y])) <=50:##35 run=35 ml-
            #             print(i,candi_points[y])
                        pix_dist[i]=np.array([sum(np.diff(candi_points[y]))])
                        candi_points[y]=np.array([candi_points[y][0],candi_points[y][-1]])
                    else:
                        # print(">50:")
                        # print(i,candi_points[y])
                        pos_id=np.where(np.diff(candi_points[y])==max(np.diff(candi_points[y])[np.diff(candi_points[y])<150]))[0][0]
                        pix_dist[i]=np.array([np.diff(candi_points[y])[pos_id]])
                        candi_points[y]=np.array([candi_points[y][pos_id],candi_points[y][pos_id+1]])

            pixel_out=pd.DataFrame.from_dict(candi_points,orient="index").reset_index()
            pixel_dist=pd.DataFrame.from_dict(pix_dist,orient="index",columns=["pix_dist"]).reset_index()
            pixel_out=pd.concat([pixel_out,pixel_dist],axis=1)

            pixel_out.columns=["y_1","x_1","x_2","loc","pix_dist"]
            pixel_out["y_2"]=pixel_out["y_1"]
            pixel_out["run_no"]=int(run_no)
            pixel_out=pixel_out[["run_no","loc","x_1","y_1","x_2","y_2","pix_dist"]]
        #         print("{}-Camera, Run={}".format(camera,run_no))
            print(pixel_out)
            print("\n")

            df_pixels=pd.concat([df_pixels,pixel_out],axis=0)

            ### [OutPut] save pixel count file for each run

            #Check directory and create
            if os.path.exists(csv_path+camera+"/") ==False:
                os.mkdir(csv_path+camera+"/")
                print("Directory {} created".format(csv_path+camera+"/"))

            pixel_out.to_csv(csv_path+camera+"/"+"Run"+run_no+"-"+camera+"Bottom-pixels.csv")
            print("Saved to "+csv_path+camera+"/"+"Run"+run_no+"-"+camera+"Bottom-pixels.csv")


            df_pixels["camera"]=camera
            df_pixels_all=pd.concat([df_pixels_all,df_pixels],axis=0).sort_values(by=["run_no","camera"]).reset_index(drop=True)

        ### [OutPut] save pixel counts
        df_pixels_all.to_csv(csv_path+"allcameras_pixels_count_bottom_fromImageProcess.csv",index=False)

    #     #### ---------------Gap & Flush Prediction-----------------------------

        df_pixels_new=df_pixels_all.loc[df_pixels_all["run_no"]==int(run_no),]
        # if len(set(df_pixels_new.camera.unique()))==len(cameras):
        print("Predicting gap and flush for run ={}.........".format(run_no))
        pred_path=out_dir+"prediction/"
        pred_csv="gap_flush_prediction.csv"
        print(df_pixels_new)

        df_pred_new=pred_bottom.pred_gap_flush(df_pixels_new,pred_path,run_no,pred_csv)
        print("--------------------------Run={} Gap&Fush Prediction ------------------------".format(run_no))
        print(df_pred_new)
        print("\n")
        df_pred_new.to_csv(pred_path+"pred_gap_flush_run"+run_no+"_bottom.csv")


         #### [OUTPUT] PLOT ROI
        for camera in valid_cameras:
            plot_para=[img,bin_edg,df_pred_new,ROI_loc[camera]]
            df_pixes_camera=df_pixels_new.loc[df_pixels_new["camera"]==camera,].reset_index(drop=True)
            fig1=pr_bottom.plot_result(images[camera],bin_edg_images[camera],df_pixes_camera,df_pred_new,ROI_loc[camera])
            fig1.suptitle("Gap & Flush prediction:Run={},{} Camera".format(run_no,camera))

            ### [OutPut] save ROI Measurement points image
            #Check directory and create if not
            if os.path.exists(fig_path+camera+"/") ==False:
                os.mkdir(fig_path+camera+"/")
                print("Directory {} created".format(fig_path+camera+"/"))
            fig1.savefig(fig_path+camera+"/"+"Run"+run_no+"-"+camera+"-Bottom-ROI-MeasurePoint.png")
            # fig1=plt.imread(fig_path+camera+"/"+"Run"+run_no+"-"+camera+"-ROI-MeasurePoint.png")
            # plt.show()
            plt.close("all")
            figs[camera]=fig1

            ##

        # else:
        #     missed_camera=set(cameras)-set(set(df_pixels_new.camera.unique()))
        #     df_pred_new=pd.DataFrame()
        #     print("No pixel count from camera {},Gap and flush prediction will not run.".format(missed_camera))


     #### ---------------Control Chart-----------------------------
    gap_flush_pred_path=out_dir+"prediction/"+pred_csv
    if os.path.exists(gap_flush_pred_path) ==True:
        df_pred=pd.read_csv(gap_flush_pred_path)
        if df_pred.shape[0]>0:
            for loc in df_pred["loc"].unique():
                df_pred_loc=df_pred[df_pred["loc"]==loc].sort_values(by="run_no")
                
                if df_pred_loc.shape[0]>=3:
                    y=df_pred_loc[["run_no","Gap_pred(mm)","Flush_pred(mm)"]]

                    y.set_index("run_no",inplace=True)
                    chart1=control_chart(y, alpha=0.01, legend_right=True);
                    plt.xticks(range(len(y.index)),labels=y.index)
                    chart1.figure.savefig(chart_path+loc+"-Bottom-hotelling-chart.png")

                    chart2=univariate_control_chart(y, legend_right=True);   
                    # chart2.suptitle(loc)  
                    chart2.savefig(chart_path+loc+"-Bottom-Uni-chart.png")

                    ewma_object=Ewma()
                    chart3=ewma_object.ewma_control_chart(y);   
                    # chart2.suptitle(loc)  
                    chart3.savefig(chart_path+loc+"-Bottom-EWMA-chart.png")

                else:
                    print("No control chart generated for loc{},since less than 3 samples".format(loc))       
    else:
        print("No control chart since no file found of {}".format(gap_flush_pred_path))
    
    return(df_pred_new)

