# Compute vs model profile as a function of age based on a Half Space Cooling Model.
# Error function wrong, fits Ye's 2013 results.
import numpy as np
from scipy.special import erf
#import math
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage.filters import gaussian_filter
import pycpt
import sys
from matplotlib.ticker import MultipleLocator
sys.path.append('/work3/wang/code_bkup/pyMCinv/temp_2_vel')
import Mantle_temp_vel
import burn_thermal_ela

"""
def vs_profile(age, M):
    Ts = 0.
    Tm = 1300.
    kappa = 1.e-6
    depth = np.linspace(0.5,100,M)
    Pressure = 3.3e6 * depth * 9.8
    temps    = Ts+(Tm-Ts)*erf(depth*1.e3/(2*np.sqrt(kappa*1.e6*age*365*24*3600)))
    mant_mod = Mantle_temp_vel.OceanMant()
    vs_noQ = mant_mod.get_vs(Pressure, temps+273.15)
    Qs = cal_Q(18.,0.1,depth,temps+273.15,T=1.)
    vs_mant = vs_noQ * (1.-1./(2.*Qs*np.tan(np.pi/2.*0.15)))
    #vs_mant = mant_mod.get_Vanel(Pressure,temps+273.15)
    #Qs = mant_mod.Qs
    return vs_noQ, vs_mant, Qs, temps
"""

def vs_profile(age,M):
    Ts = 0.
    Tm = 1365.
    kappa = 1.e-6
    depth = np.linspace(0.5,200.,M)
    Pressure = 3.3e6 * depth * 9.8
    temps    = Ts+(Tm-Ts)*erf(depth*1.e3/(2*np.sqrt(kappa*1.e6*age*365*24*3600)))
    #mant_mod = burn_thermal_ela.OceanMant()
    #vs_noQ = mant_mod.get_vs_aha(Pressure, temps+273.15)
    vs_noQ = 4.77+0.038*Pressure*1.e-9-0.000378*(temps+273.15-300)
    Qs = cal_Q(10,0.1,depth,temps+273.15,T=16.)
    vs_mant = vs_noQ * (1.-1./(2.*Qs*np.tan(np.pi/2.*0.1)))
    return vs_noQ, vs_mant, Qs, temps

def cal_Q(A,alpha,z,temp,T=2*np.pi):
    return A * ((2*np.pi/T) ** alpha) * np.exp(alpha*(2.5e5+z*3.3*9.8*10)/temp/8.314)

def plot_profile(age_arr,z_arr,vs_arr,nc=5, vmin=4.1,vmax=4.55, map_r=False,title='',label='',showfig=False):
    xi  = np.linspace(np.min(age_arr),np.max(age_arr),500)
    yi  = np.linspace(np.max(z_arr[:,0]),np.min(z_arr[:,-1]),500)
    xx, yy = np.meshgrid(xi,yi)
    vsi = griddata((age_arr.flatten(),z_arr.flatten()),vs_arr.flatten(),(xx,yy),method='cubic')
    plt.figure()
    plt.rcParams['xtick.top']=True
    plt.rcParams['ytick.right']=True
    xminorLocator = MultipleLocator(5.)
    vs = gaussian_filter(vsi,5,order=0)
    my_cmap = pycpt.load.gmtColormap('./cv_original.cpt')
    if map_r:
        my_cmap = my_cmap.reversed()
    cm = plt.pcolormesh(xx,yy,vs,cmap=my_cmap,shading='gouraud',vmin=vmin,vmax=vmax)
    CS = plt.contour(xx, yy, vs, nc, colors='gray',linestyles='dashed')
    plt.gca().clabel(CS, inline=1, fontsize=10)
    plt.gca().yaxis.set_minor_locator(xminorLocator)
    #plt.ylim(ymin=10,ymax=50)
    #plt.xlim(xmin=0.5,xmax=3.5)
    plt.xlabel('Age (ma)', fontsize=12)
    plt.ylabel('Depth (km)', fontsize=12)
    plt.gca().invert_yaxis()
    cb2 = plt.colorbar(cm, orientation="horizontal", aspect=40, pad=0.13, format='%.3f')
    cb2.set_label(label, fontsize=10, rotation=0)
    cb2.set_alpha(1)
    cb2.draw_all()
    plt.title(title)
    if showfig:
        plt.show()
    return

if __name__ == "__main__":
    N = 70
    M = 400
    age_arr = np.zeros([N,M],dtype=np.float64)
    vs_arr  = np.zeros([N,M],dtype=np.float64)
    vs_noQ  = np.zeros([N,M],dtype=np.float64)
    Qs      = np.zeros([N,M],dtype=np.float64)
    temp_arr= np.zeros([N,M],dtype=np.float64)
    z_arr   = np.zeros([N,M],dtype=np.float64)
    depths = np.linspace(0.5,100.,M)
    ages = np.linspace(0.1,10.,N)
    for i,age in enumerate(ages):
        z_arr[i,:] = depths
        age_arr[i,:] = age
        vs_noQ[i,:],vs_arr[i,:],Qs[i,:],temp_arr[i,:] = vs_profile(age,M)
    
    #vs_3 = 4.77+0.038*z_arr*3.3e-3*9.8-0.000378*(temp_arr+273.15-300)
    #Qs = cal_Q(0.5,0.25,z_arr,temp_arr+273.15,T=1.)
    #vs_31 = vs_3 * (1.-1./(2.*Qs*np.tan(np.pi/2.*0.1)))
    #plot_profile(age_arr,z_arr,Qs,nc=0,vmin=30.,vmax=200.,title='Q')
    #plot_profile(age_arr,z_arr,vs_arr,title='With Q')
    #plot_profile(age_arr,z_arr,vs_3,title='Ref')
    plot_profile(age_arr,z_arr,temp_arr,vmin=0,vmax=1300,map_r=True,title='Temperature',label='Temperature (Celcius)')
    plot_profile(age_arr,z_arr,Qs,nc=[10,20,30,50,80,100,200],vmin=30,vmax=200,title='Q',label='Q',showfig=False)
    plot_profile(age_arr,z_arr,vs_noQ,nc=np.array([4.25,4.4,4.5,4.6]),vmin=4.,vmax=4.75,title='Anharmonic velocity',label='km/s')
    plot_profile(age_arr,z_arr,vs_arr,nc=np.array([4.25,4.3,4.4,4.5,4.6]),vmin=4.,vmax=4.75,title='Shear Velocity (km/s)',label='km/s',showfig=True)
    #plot_profile(age_arr,z_arr,vs_noQ,title='No Q',showfig=False)
    #diff = vs_noQ - vs_3
    #plot_profile(age_arr,z_arr,diff,vmin=None,vmax=None,title='Diff No Q',showfig=True)
    
    
    """
    plt.rcParams['xtick.top']=True
    plt.figure()
    #plt.plot(temp_arr[4,:]+273.15,z_arr[4,:],label='0.5 Ma',color='red')
    #plt.plot(temp_arr[14,:]+273.15,z_arr[14,:],label='1.5 Ma',color='green')
    plt.plot(temp_arr[-1,:],z_arr[-1,:],label='7 Ma',color='blue')
    plt.xlim([800,1400])
    plt.ylim([10,100])
    plt.gca().invert_yaxis()
    plt.legend()
    plt.show()
    """
    """
    Qs1 = cal_Q(30,0.1,z_arr[4,:],temp_arr[4,:]+273.15,T=10.)
    Qs11= cal_Q(15,0.1,z_arr[4,:],temp_arr[4,:]+273.15,T=10.)
    Qs12= cal_Q(50,0.1,z_arr[4,:],temp_arr[4,:]+273.15,T=10.)
    Qs2 = cal_Q(30,0.1,z_arr[14,:],temp_arr[14,:]+273.15,T=10.)
    Qs21= cal_Q(15,0.1,z_arr[14,:],temp_arr[14,:]+273.15,T=10.)
    Qs22= cal_Q(50,0.1,z_arr[14,:],temp_arr[14,:]+273.15,T=10.)
    Qs3 = cal_Q(30,0.1,z_arr[29,:],temp_arr[29,:]+273.15,T=10.)
    Qs31= cal_Q(15,0.1,z_arr[29,:],temp_arr[29,:]+273.15,T=10.)
    Qs32= cal_Q(50,0.1,z_arr[29,:],temp_arr[29,:]+273.15,T=10.)
    
    plt.figure()
    plt.plot(Qs1,z_arr[4,:],label='0.5 Ma',color='red')
    plt.plot(Qs11,z_arr[4,:],color='red')
    plt.plot(Qs12,z_arr[4,:],color='red')
    plt.plot(Qs2,z_arr[14,:],label='1.5 Ma',color='green')
    plt.plot(Qs21,z_arr[14,:],color='green')
    plt.plot(Qs22,z_arr[14,:],color='green')
    plt.plot(Qs3,z_arr[29,:],label='3 Ma',color='blue')
    plt.plot(Qs31,z_arr[29,:],color='blue')
    plt.plot(Qs32,z_arr[29,:],color='blue')
    plt.text(90, 55, 'A=15', color='k')
    plt.text(200, 55, 'A=30', color='k')
    plt.text(330, 55, 'A=50', color='k')
    plt.legend()
    plt.xlim([0,600])
    plt.ylim([10,100])
    plt.gca().invert_yaxis()
    plt.show()
    """
