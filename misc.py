import numpy as np 
from scipy import interpolate
import tensorflow as tf 

def schedule(epoch,lr):
    if epoch > 50:
        lr *= 0.98
    return lr 
    
def exp_flux_sampling(time_array,df_exp,FullScanT,PortionAnalyzed=0.75):
    interpolated_flux = np.zeros_like(time_array)
    time_array_length = len(time_array)
    df_exp_forward = df_exp.iloc[:int(len(df_exp)*0.5)]
    df_exp_reverse = df_exp.iloc[int(len(df_exp)*0.5):int(len(df_exp)*PortionAnalyzed)]
    forward_scan_time_array = time_array[time_array<FullScanT*0.5]
    reverse_scan_time_array = time_array[time_array>=FullScanT*0.5]
    assert time_array_length == len(forward_scan_time_array) + len(reverse_scan_time_array)
    f_forward = interpolate.interp1d(np.linspace(0,FullScanT*0.5,num=len(df_exp_forward)),df_exp_forward.iloc[:,1])
    f_reverse = interpolate.interp1d(np.linspace(FullScanT*0.5,FullScanT*PortionAnalyzed,num=len(df_exp_reverse)),df_exp_reverse.iloc[:,1])
    interpolated_flux[:len(forward_scan_time_array)] = f_forward(forward_scan_time_array)
    interpolated_flux[len(forward_scan_time_array):] = f_reverse(reverse_scan_time_array)
    return np.array([[v] for v in interpolated_flux])
