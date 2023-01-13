import os
import sys
import pandas as pd


def take_pic_top(run_no,time_stamp):
	# run_no = sys.argv[1]
	pic_dir="pictures/TopCamera/"

	imgL_path=pic_dir+"/run-"+run_no+"-3.jpg"
	imgR_path=pic_dir+"/run-"+run_no+"-2.jpg"
	imgML_path=pic_dir+"/run-"+run_no+"-0.jpg"
	imgM_path=pic_dir+"/run-"+run_no+"-1.jpg"
	# print(imgL_path)

	## File to save log
	log_file=pic_dir+"log_file.csv"
	if os.path.exists(log_file) ==True:
	    df_log=pd.read_csv(log_file)
	else:
	    df_log=pd.DataFrame()

	if df_log.shape[0]>0 and (run_no in df_log["run_no"].unique()):
	    print("Warning: The result is avaliable for run= {}, it will be replaced.".format(run_no))
	    df_log=df_log[df_log["run_no"]!=int(run_no)].copy(deep=True)
	else:
	    print("Taking pictures for run ={}...".format(run_no))


	# Set to single channel 0
	status=False
	
	try:
		if os.system("i2cset -y 10 0x24 0x24 0x02")==0:
			os.system("i2cset -y 10 0x24 0x24 0x02")
			take_picture_0 = os.system("libcamera-still -t 5000 -o run-{}-0.jpg".format(run_no))
			print("The exit code was: %d" % take_picture_0)
			status=True
		else:
			raise Exception('command does not exist')

	except:
		print("The image was not taken with TopCamera-MiddleLeft Successfully!")

	log_dic={"run_no":run_no,"time":time_stamp,"camera":"Middle Left","camera":"Middle Left","path":imgML_path,"Successfull":status}
	df_log=df_log=pd.concat([df_log,pd.DataFrame.from_dict(log_dic,orient="index").T])

	# Set to single channel 1
	status=False
	
	try:
		if os.system("i2cset -y 10 0x24 0x24 0x12")==0:
			os.system("i2cset -y 10 0x24 0x24 0x12")
			take_picture_1 = os.system("libcamera-still -t 5000 -o run-{}-1.jpg".format(run_no))
			print("The exit code was: %d" % take_picture_1)
			status=True
		else:
			raise Exception('command does not exist')
	except:
		print("The image was not taken with TopCamera-Middle Successfully!")

	log_dic={"run_no":run_no,"time":time_stamp,"camera":"Middle","path":imgM_path,"Successfull":status}
	df_log=df_log=pd.concat([df_log,pd.DataFrame.from_dict(log_dic,orient="index").T])


	# Set to single channel 2
	status=False
	try:
		if os.system("i2cset -y 10 0x24 0x24 0x22")==0:
			os.system("i2cset -y 10 0x24 0x24 0x22")
			take_picture_2 = os.system("libcamera-still -t 5000 -o run-{}-2.jpg".format(run_no))
			print("The exit code was: %d" % take_picture_2)
			status=True
		else:
			raise Exception('command does not exist')
	except:
		print("The image was not taken with TopCamera-Right Successfully!")
	log_dic={"run_no":run_no,"time":time_stamp,"camera":"Right","path":imgR_path,"Successfull":status}
	df_log=df_log=pd.concat([df_log,pd.DataFrame.from_dict(log_dic,orient="index").T])

	# Set to single channel 3
	status=False
	
	try:
		if os.system("i2cset -y 10 0x24 0x24 0x32")==0:
			os.system("i2cset -y 10 0x24 0x24 0x32")
			take_picture_3 = os.system("libcamera-still -t 5000 -o run-{}-3.jpg".format(run_no))
			print("The exit code was: %d" % take_picture_3)
			status=True
		else:
			raise Exception('command does not exist')
	except:
		print("The image was not taken with TopCamera-Left Successfully!")

	log_dic={"run_no":run_no,"time":time_stamp,"camera":"Left","path":imgL_path,"Successfull":status}
	df_log=df_log=pd.concat([df_log,pd.DataFrame.from_dict(log_dic,orient="index").T])

	## Save log
	df_log.to_csv(pic_dir+"log_file.csv",index=False)

	return pic_dir

def take_pic_bottom(run_no,time_stamp):
	# run_no = sys.argv[1]
	pic_dir="pictures/BottomCamera/"


	imgL_path=pic_dir+"/run-"+run_no+"-3.jpg"
	imgR_path=pic_dir+"/run-"+run_no+"-2.jpg"
	imgML_path=pic_dir+"/run-"+run_no+"-0.jpg"
	imgM_path=pic_dir+"/run-"+run_no+"-1.jpg"
	# print(imgL_path)

	## File to save log
	log_file=pic_dir+"log_file.csv"
	if os.path.exists(log_file) ==True:
	    df_log=pd.read_csv(log_file)
	else:
	    df_log=pd.DataFrame()

	if df_log.shape[0]>0 and (run_no in df_log["run_no"].unique()):
	    print("Warning: The result is avaliable for run= {}, it will be replaced.".format(run_no))
	    df_log=df_log[df_log["run_no"]!=int(run_no)].copy(deep=True)
	else:
	    print("Taking pictures for run ={}...".format(run_no))


	# Set to single channel 0
	status=False
	
	try:
		if os.system("i2cset -y 10 0x24 0x24 0x02")==0:
			os.system("i2cset -y 10 0x24 0x24 0x02")
			take_picture_0 = os.system("libcamera-still -t 5000 -o run-{}-0.jpg".format(run_no))
			print("The exit code was: %d" % take_picture_0)
			status=True
		else:
			raise Exception('command does not exist')

	except:
		print("The image was not taken with BottomCamera-MiddleLeft Successfully!")

	log_dic={"run_no":run_no,"time":time_stamp,"camera":"Middle Left","camera":"Middle Left","path":imgML_path,"Successfull":status}
	df_log=df_log=pd.concat([df_log,pd.DataFrame.from_dict(log_dic,orient="index").T])

	# Set to single channel 1
	status=False
	
	try:
		if os.system("i2cset -y 10 0x24 0x24 0x12")==0:
			os.system("i2cset -y 10 0x24 0x24 0x12")
			take_picture_1 = os.system("libcamera-still -t 5000 -o run-{}-1.jpg".format(run_no))
			print("The exit code was: %d" % take_picture_1)
			status=True
		else:
			raise Exception('command does not exist')
	except:
		print("The image was not taken with BottomCamera-Middle Successfully!")

	log_dic={"run_no":run_no,"time":time_stamp,"camera":"Middle","path":imgM_path,"Successfull":status}
	df_log=df_log=pd.concat([df_log,pd.DataFrame.from_dict(log_dic,orient="index").T])


	# Set to single channel 2
	status=False
	try:
		if os.system("i2cset -y 10 0x24 0x24 0x22")==0:
			os.system("i2cset -y 10 0x24 0x24 0x22")
			take_picture_2 = os.system("libcamera-still -t 5000 -o run-{}-2.jpg".format(run_no))
			print("The exit code was: %d" % take_picture_2)
			status=True
		else:
			raise Exception('command does not exist')
	except:
		print("The image was not taken with BottomCamera-Right Successfully!")
	log_dic={"run_no":run_no,"time":time_stamp,"camera":"Right","path":imgR_path,"Successfull":status}
	df_log=df_log=pd.concat([df_log,pd.DataFrame.from_dict(log_dic,orient="index").T])

	# Set to single channel 3
	status=False
	
	try:
		if os.system("i2cset -y 10 0x24 0x24 0x32")==0:
			os.system("i2cset -y 10 0x24 0x24 0x32")
			take_picture_3 = os.system("libcamera-still -t 5000 -o run-{}-3.jpg".format(run_no))
			print("The exit code was: %d" % take_picture_3)
			status=True
		else:
			raise Exception('command does not exist')
	except:
		print("The image was not taken with BottomCamera-Left Successfully!")

	log_dic={"run_no":run_no,"time":time_stamp,"camera":"Left","path":imgL_path,"Successfull":status}
	df_log=df_log=pd.concat([df_log,pd.DataFrame.from_dict(log_dic,orient="index").T])

	## Save log
	df_log.to_csv(pic_dir+"log_file.csv",index=False)

	return pic_dir