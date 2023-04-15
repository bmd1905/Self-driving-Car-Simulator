import numpy as np
import time

error_arr = np.zeros(5)
error_sp = np.zeros(5)

pre_t = time.time()
pre_t_spd = time.time()

def Calc_error(image):
    try:
        arr=[]
        height = 12
        lineRow = image[height,:]
        for x, y in enumerate(lineRow):
            if y[0] == 100:
                arr.append(x)

        if len(arr) > 0:
            center = int( (min(arr) + max(arr))/2 )
            error = int(image.shape[1]/2) - center
            return error
        else:
            print("Return +19")
            return 40
    except:
         print("Return 0")
         return 0

def PID(error, p, i, d): #0.43,0,0.02
    global pre_t
    global error_arr
    error_arr[1:] = error_arr[0:-1]
    error_arr[0] = error
    P = error*p
    delta_t = time.time() - pre_t
    pre_t = time.time()
    D = (error-error_arr[1])/delta_t*d
    I = np.sum(error_arr)*delta_t*i
    angle = P + I + D

    if abs(angle)>25:
        angle = np.sign(angle)*25
        
    return int(angle)


def calc_speed(angle):
    if abs(angle) < 10:
        speed = 60
    elif 10 <= abs(angle) <= 20:
        speed = 30
    else:
        speed = 20
    return speed


def PID_speed(error, p, i, d, speed_max):
	global pre_t_spd
	global error_sp

	error_sp[1:] = error_sp[0:-1]
	error_sp[0] = error
	P = error*p
	delta_t = time.time() - pre_t_spd
	dif = time.time()
	D = (error-error_sp[1])/delta_t*d
	I = np.sum(error_sp)*delta_t*i
	speed = P + I + D
	if speed > speed_max: 
		speed = speed_max
	if speed < 0: 
		speed = -5
	return speed