
from tkinter import *   ##button
from tkinter import messagebox
from py.imagePro_Gui import imageprocess_top,imageprocess_bottom
from functools import partial  
from tkinter import font  as tkfont 
import tkinter as tk
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image,ImageTk
from py.take_pic import take_pic_top,take_pic_bottom
from py.control_chart import plot_control_chart
import csv

from datetime import datetime
import pytz

# current time


def click_take_pic(n1,label_message1,label_message2,camera_loc):
    now = datetime.now(pytz.timezone('US/Eastern'))
    format_time=now.strftime("%Y-%m-%d-%H%M%S%d")
    if not n1.get():
        run_no=format_time
    else:
        num1 = (n1.get()) 
        run_no = str(int(num1))
    
    if camera_loc =="Top":
        save_dir_top=take_pic_top(str(run_no),format_time)
        label_message1.config(text="saving to %s." %save_dir_top )
    if camera_loc =="Bottom":
        save_dir_bottom=take_pic_bottom(str(run_no),format_time)
        label_message2.config(text="saving to %s." %save_dir_bottom )
    return
   
def call_result(label_result,label_message, n1):#,df_Pred):
    now = datetime.now(pytz.timezone('US/Eastern'))
    format_time=now.strftime("%Y-%m-%d-%H%M%S%d")
    if not n1.get():
        run_no=format_time
    else:
        num1 = (n1.get()) 
        run_no = str(int(num1))

    label_message.config(text="Computing for run = %s....." %run_no ) 
    ## Call Top Cameras result
    start_time=datetime.now()
    df_result0=imageprocess_top(str(run_no))
    end_time=datetime.now()
    print("Runing time for image process for top cameras is {}".format(end_time-start_time))
    df_result0[["Right","Middle","Left","Middle Left"]]=df_result0[["Right","Middle","Left","Middle Left"]].apply(lambda x :round(x,2))
    df_result0["Gap_pred(mm)"]=df_result0["Gap_pred(mm)"].apply(lambda x :round(x,5))
    df_pred0=list(df_result0.values)
    
    entrieslist = []
    for i, row in enumerate(df_pred0):
        entrieslist.append(row[0])
        for col in range(0, 7):
            if col in [0,1,2]:
                tk.Label(root, text=row[col]).place(x=30+col*50,y=350+i*30)
            elif col_name in [6,7]:
                tk.Label(root, text=row[col]).place(x=400+(i-6)*110,y=350+i*30)
            else:
                 tk.Label(root, text=row[col]).place(x=180+(col-3)*80,y=350+i*30)

    ## Call bottom Cameras result
    # label_message.config(text="Computing for run = %d,bottom cameras....." %run_no ) 
    start_time=datetime.now()
    df_result1=imageprocess_bottom(str(run_no))
    end_time=datetime.now()
    print("Runing time for image process for top cameras is {}".format(end_time-start_time))
    df_result1["Gap_pred(mm)"]=df_result1["Gap_pred(mm)"].apply(lambda x :round(x,5))
    df_result1["Flush_pred(mm)"]=df_result1["Flush_pred(mm)"].apply(lambda x :round(x,5))
    df_pred1=list(df_result1.values)

    label_result.config(text="Result for Run= %s" %run_no)

    ### show bottom Camera result on screen
    entrieslist = []
    for i, row in enumerate(df_pred1):
        entrieslist.append(row[0])
        for col in range(0, 8):
            if col in [0,1,2]:
                tk.Label(root, text=row[col]).place(x=30+col*50,y=410+i*30)
            elif col_name in [6,7]:
                tk.Label(root, text=row[col]).place(x=400+(i-6)*110,y=410+i*30)
            else:
                 tk.Label(root, text=row[col]).place(x=180+(col-3)*80,y=410+i*30)
    return 

img=None
fig=None
def call_plot(camera,n1,camera_loc):

    global img
    global fig
    out_dir='output/'
    fig_dir= os.path.join(out_dir, 'fig/')
    if not n1.get():
        run_no=format_time
    else:
        num1 = (n1.get()) 
        run_no = str(int(num1))

    leaf1=tk.Toplevel(root)
    leaf1.geometry("1600x800")  

    fig_path=fig_dir+camera+"/"+"Run"+str(run_no)+"-"+camera+"-"+camera_loc+"-ROI-MeasurePoint.png"
    img=Image.open(fig_path)
    print(fig_path)
    fig=ImageTk.PhotoImage(img)
    imglabel=Label(leaf1,image=fig).grid(row=0,column=0,columnspan=3)#place(x=30,y=800)
    leaf1.mainloop()


img1,img2=None,None
fig1,fig2=None,None
def call_chart(loc,n1,camera_loc):

    global img1,img2
    global fig1,fig2
    out_dir='output/'
    chart_dir=os.path.join(out_dir, 'controlchart/')
    if not n1.get():
        run_no=format_time
    else:
        num1 = (n1.get()) 
        run_no = str(int(num1))

    leaf2=tk.Toplevel(root)
    # leaf2.geometry("1000x2400") 
    sbar1= tk.Scrollbar(leaf2)
    sbar1.pack(side=RIGHT, fill=Y)  
    sbar1.grid(row=0, column=1, sticky="ns")

    popCanv = Canvas(leaf2, width=1000, height = 2800,
                     scrollregion=(0,0,1000,2800)) #width=1256, height = 1674)
    popCanv.grid(row=0, column=0, sticky="nsew") #added sticky

    sbar1.config(command=popCanv.yview)
    popCanv.config(yscrollcommand = sbar1.set)

    chart=plot_control_chart(loc)

    if loc=="all":
        chart_path1=chart_dir+loc+"-hotelling-chart.png"
    elif loc in ["A","B","C","D","E"]:
        chart_path1=chart_dir+loc+"-EWMA-chart.png"
    
    img1=Image.open(chart_path1)
    fig1=ImageTk.PhotoImage(img1)
    if loc in ["all","A","B"]:
        image2 = popCanv.create_image(500, 300, image=fig1) #correct way of adding an image to canvas
    else:
        image2 = popCanv.create_image(500, 500, image=fig1) #correct way of adding an image to canvas
    title_font3 = tkfont.Font(family='Helvetica', size=18, weight="bold", slant="roman")
    popCanv.create_text(500,30,text="Control Plot for loc = "+loc,font=title_font3)

    # chart_path2=chart_dir+loc+"-EWMA-chart.png"
    # img2=Image.open(chart_path2)
    # fig2=ImageTk.PhotoImage(img2)
    # image2 = popCanv.create_image(500, 1070, image=fig2) #correct way of adding an image to canvas

    # if camera_loc=="Bottom":
    #     chart_path3=chart_dir+loc+"-"+camera_loc+"-Uni-chart.png"
    #     img3=Image.open(chart_path3)
    #     fig3=ImageTk.PhotoImage(img3)
    #     image3 = popCanv.create_image(500, 2000, image=fig3) #correct way of adding an image to canvas
    #     
    #     

    leaf2.rowconfigure(0, weight=1) 
    leaf2.columnconfigure(0, weight=1)

    leaf2.mainloop()

root = tk.Tk()  
root.geometry("700x1000") 
  
root.title("SEAL-Vision-1.0")
zoom1 = 0.38
zoom2 = 0.08
image1 = Image.open("icon/honda.png")
image2 = Image.open("icon/osu-seal.png")
pixels_x, pixels_y = tuple([int(zoom1 * x)  for x in image1.size])
icon1 = ImageTk.PhotoImage(image1.resize((pixels_x, pixels_y)))
pixels_x, pixels_y = tuple([int(zoom2 * x)  for x in image2.size])
icon2 = ImageTk.PhotoImage(image2.resize((pixels_x, pixels_y)))
label_icon1 = tk.Label(image=icon1)
label_icon1.image = icon1
label_icon2 = tk.Label(image=icon2)
label_icon2.image = icon2

# Position brand image
label_icon1.place(x=550,y=25)
label_icon2.place(x=610,y=25)

title_font0 = tkfont.Font(family='Helvetica', size=16, weight="bold", slant="roman")  
title_font1 = tkfont.Font(family='Helvetica', size=14, weight="bold", slant="italic")
title_font2 = tkfont.Font(family='Helvetica', size=12, weight="bold", slant="roman")

msg = Label(root, text = "Welcome to SEAL-Vision-1.0 Measurement Application!",font=title_font0).place(x=100,y=30)

number = tk.StringVar()  
  
labelNum = tk.Label(root, text="Input run_no",font=title_font2).place(x = 30,y = 105)  
entryNum = tk.Entry(root, textvariable=number,width=10).place(x = 130, y = 100)
labeltext = tk.Label(root, text="(Optional)").place(x = 230,y = 105)  

### Labels and Buttons for taking pics
labelMessage1 = tk.Label(root)#label for picture directory
labelMessage1.place(x=220,y=145)
labelMessage2 = tk.Label(root)
labelMessage2.place(x=220,y=185)

call_picture_top = partial(click_take_pic,number, labelMessage1, labelMessage2,"Top")
call_picture_bottom = partial(click_take_pic,number, labelMessage1, labelMessage2,"Bottom")

TopCamera = tk.Label(root,text='Top Cameras',font=title_font2).place(x=30,y=145)
BottomCamera = tk.Label(root,text='Bottom Cameras',font=title_font2).place(x=30,y=185)
buttonTop = tk.Button(root, text="Click", command=call_picture_top,activeforeground = "blue"
    ,activebackground = "pink",relief="sunken").place(x=130,y=140)
buttonBottom = tk.Button(root, text="Click", command=call_picture_bottom,activeforeground = "blue"
    ,activebackground = "pink",relief="sunken").place(x=130,y=180)

### Hover (Not work)
# tip1= tk.Balloon(root)
# tip2= tk.Balloon(root)
# tip1.bind_widget(buttonTop,balloonmsg="Click to take picture for Top camera")
# tip2.bind_widget(buttonBottom ,balloonmsg="Click to take picture for Bottom camera")

##Labels and Button for computing (image processing)

labelMessage = tk.Label(root)#label for call result
labelMessage.place(x=230,y=230)

labelResult = tk.Label(root,font=title_font2)
labelResult.place(x=250,y=290)# label location for "Result for Run="

##Button for computing
call_results = partial(call_result, labelResult, labelMessage,number)
buttonRun = tk.Button(root, text="Compute", command=call_results,
    activeforeground = "blue",activebackground = "pink",relief="sunken").place(x=130,y=220)

## Label text for each title
label_Pred = tk.Label(root,text='Gap, Flush & Door Alignment Prediction',font=title_font1).place(x=30,y=270)
label_Camera = tk.Label(root,text='Visualize Image',font=title_font1).place(x=30,y=540)
label_TopCamera = tk.Label(root,text='Top Cameras',font=title_font2).place(x=30,y=590)
label_BottomCamera = tk.Label(root,text='Bottom Cameras',font=title_font2).place(x=30,y=640)
label_Conchart = tk.Label(root,text='Control Chart (Hotelling T^2 & EWMA)',font=title_font1).place(x=30,y=720)
label_Conchart1 = tk.Label(root,text='Hotelling T^2 ',font=title_font2).place(x=30,y=785)
label_Conchart2 = tk.Label(root,text='EWMA ',font=title_font2).place(x=30,y=835)
label_Measureloc= tk.Label(root, text="(Measurement Location)").place(x = 200,y = 860)

##Button for plot of each camera
call_plotTL= partial(call_plot, "Left",number,"Top2")  
call_plotTR= partial(call_plot, "Right",number,"Top2")   
call_plotTM= partial(call_plot, "Middle",number,"Top2")  
call_plotTML= partial(call_plot, "Middle Left",number,"Top2")   

call_plotBL= partial(call_plot, "Left",number,"Bottom")  
call_plotBR= partial(call_plot, "Right",number,"Bottom")   
call_plotBM= partial(call_plot, "Middle",number,"Bottom")   
call_plotBML= partial(call_plot, "Middle Left",number,"Bottom")  

call_plotT2= partial(call_chart, "all",number,"Top2")  
call_plotA= partial(call_chart, "A",number,"Top2")    
call_plotB= partial(call_chart, "B",number,"Top2")  
call_plotC= partial(call_chart, "C",number,"Bottom")  
call_plotD= partial(call_chart, "D",number,"Bottom")  
call_plotE= partial(call_chart, "E",number,"Bottom")  
# call_plotF= partial(call_chart, "F",number)  

b1 = tk.Button(root,text = "Left",command=call_plotTL,activeforeground = "red",activebackground = "pink").place(x = 150, y = 590)  
b2 = tk.Button(root, text = "Middle Left",command=call_plotTML,activeforeground = "blue",activebackground = "pink").place(x = 250, y = 590)
b3 = tk.Button(root, text = "Middle",command=call_plotTM,activeforeground = "green",activebackground = "pink").place(x = 390, y = 590)
b4 = tk.Button(root, text = "Right",command=call_plotTR,activeforeground = "yellow",activebackground = "pink").place(x = 500, y = 590)  

b5 = tk.Button(root,text = "Left",command=call_plotBL,activeforeground = "red",activebackground = "pink").place(x = 150, y = 640)  
b6 = tk.Button(root, text = "Middle Left",command=call_plotBML,activeforeground = "blue",activebackground = "pink").place(x = 250, y = 640)
b7 = tk.Button(root, text = "Middle",command=call_plotBM,activeforeground = "green",activebackground = "pink").place(x = 390, y = 640)
b8 = tk.Button(root, text = "Right",command=call_plotBR,activeforeground = "yellow",activebackground = "pink").place(x = 500, y = 640)  


##Button for each Location
bc0 = tk.Button(root,text = "All",command=call_plotT2,activeforeground = "red",activebackground = "pink").place(x = 120, y = 780)  
bc1 = tk.Button(root,text = "A",command=call_plotA,activeforeground = "red",activebackground = "pink").place(x = 120, y = 830)  
bc2 = tk.Button(root, text = "B",command=call_plotB,activeforeground = "blue",activebackground = "pink").place(x = 190, y = 830)  
bc3 = tk.Button(root, text = "C",command=call_plotC,activeforeground = "green",activebackground = "pink").place(x = 260, y = 830)  
bc4 = tk.Button(root, text = "D",command=call_plotD,activeforeground = "yellow",activebackground = "pink").place(x = 330, y = 830)  
bc5 = tk.Button(root, text = "E",command=call_plotE,activeforeground = "green",activebackground = "pink").place(x = 400, y = 830)  
# bc6 = tk.Button(root, text = "F",command=call_plotF,activeforeground = "yellow",activebackground = "pink").place(x = 500, y = 850)   


# Define column Labels.
col_names = ("run_no", "loc", "Left", "Middle Left", "Middle","Right", "Gap/V_diff(mm)",
             "Flush(mm)")
for i, col_name in enumerate(col_names):
    if col_name in ["run_no", "loc", "Left"]:
        tk.Label(root, text=col_name).place(x=30+i*50,y=320)
    elif col_name in ["Gap/V_diff(mm)","Flush(mm)"]:
        tk.Label(root, text=col_name).place(x=400+(i-6)*110,y=320)
    else:
        tk.Label(root, text=col_name).place(x=180+(i-3)*80,y=320)

closebutton = Button(root, text='Exit', command=root.destroy,activeforeground = "red",activebackground = "yellow",padx=15,pady=10).place(x = 480, y = 880)


root.mainloop()  