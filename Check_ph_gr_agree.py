# Check if the phase and group velocity curves are consistant.

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import sys
import ocean_surf_dbase

if __name__ == "__main__":
    N = 50000
    if len(sys.argv) != 2:
        raise ValueError("Usage: python Check_ph_gr_agree.py loc")
    loc = sys.argv[1]
    #gr_disp_f = 'JdF_Profile_temp_both/'+loc+'_disp_gr.txt'
    #ph_disp_f = 'JdF_Profile_temp_both/'+loc+'_disp_ph.txt'
    #T_grv = np.loadtxt(gr_disp_f)[:,:-1]
    #T_phv = np.loadtxt(ph_disp_f)[:,:-1]
    dset = ocean_surf_dbase.invhdf5('test_surf_dbase.h5')
    T_grv = np.transpose(dset[loc]["disp_gr_ray"].value[:-1,:])
    T_phv = np.transpose(dset[loc]["disp_ph_ray"].value[:-1,:])
    gr_savg = savgol_filter(T_grv[:,1], 5, 2, deriv=0)
    ph_savg = savgol_filter(T_phv[:,1], 5, 2, deriv=0)
    spl = InterpolatedUnivariateSpline(T_phv[:,0], T_phv[:,1])
    spl_savg = InterpolatedUnivariateSpline(T_phv[:,0], ph_savg)
    xs = np.linspace(T_phv[:,0].min(),T_phv[:,0].max(),N)
    vg = spl(xs)/(1+xs/spl(xs)*np.gradient(spl(xs),xs[1]-xs[0]))
    vg_savg = spl_savg(xs)/(1+xs/spl_savg(xs)*np.gradient(spl_savg(xs),xs[1]-xs[0]))
    plt.plot(xs, spl(xs), 'g-', label='cubic')
    plt.plot(xs, spl_savg(xs), 'b-', label='cubic savg')
    plt.plot(T_phv[:,0], T_phv[:,1],'g.', label='ph v')
    plt.plot(T_phv[:,0], ph_savg, 'gv', label='ph savg')
    plt.plot(xs, vg, 'r-', label='gr cal')
    plt.plot(xs, vg_savg, 'b-', label='gr cal savg')
    plt.plot(T_grv[:,0], T_grv[:,1], 'r.', label='gr v')
    plt.plot(T_grv[:,0], gr_savg, 'rv', label='gr savg')
    plt.xlabel('Period (sec)',fontsize=14)
    plt.ylabel('vel (km/s)',fontsize=14)
    plt.title(loc,fontsize=16)
    plt.legend(fontsize=12)
    plt.show()
