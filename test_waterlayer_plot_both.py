import numpy as np
import mcpost
import sys

if len(sys.argv) != 2:
    raise ValueError("Usage: python test_waterlayer_plot.py loc")

loc = sys.argv[1]
f = open('JdF_Profile_both/water_'+loc+'.mod','r')
#f = open('hongda_all/water_'+loc+'.mod_temp','r')
lines = f.readlines()
age = float(lines[0].strip())
waterdepth = float(lines[1].strip().split()[2])
f.close()

vpr = mcpost.postvpr(factor=1., thresh=0.2, age=age, waterdepth = waterdepth)
vpr.numbp = np.array([1, 2, 4, 5])
vpr.mtype = np.array([5, 4, 2, 2])

vpr.read_inv_data('water_hongda_both/mc_inv.'+loc+'.npz')
vpr.read_data('water_hongda_both/mc_data.'+loc+'.npz')
#vpr.read_inv_data('hongda_all/mc_inv.'+loc+'.npz')
#vpr.read_data('hongda_all/mc_data.'+loc+'.npz')

vpr.get_vmodel()
vpr.plot_disp(disptype='both',showfig=False)
vpr.plot_profile(showfig=False)
vpr.plot_hist(pindex=0, bins=50, title='Sediment top', xlabel='vs (km/s)', showfig=False)
vpr.plot_hist(pindex=1, bins=50, title='Sediment bottom', xlabel='vs (km/s)', showfig=False)
vpr.plot_hist(pindex=2, bins=50, title='Crust 1', xlabel='vs (km/s)', showfig=False)
vpr.plot_hist(pindex=3, bins=50, title='Crust 2', xlabel='vs (km/s)', showfig=False)
vpr.plot_hist(pindex=4, bins=50, title='Crust 3', xlabel='vs (km/s)', showfig=False)
vpr.plot_hist(pindex=5, bins=50, title='Crust 4', xlabel='vs (km/s)', showfig=False)
vpr.plot_hist(pindex=6, bins=50, title='Mantle 1', xlabel='vs (km/s)', showfig=False)
vpr.plot_hist(pindex=7, bins=50, title='Mantle 2', xlabel='vs (km/s)', showfig=False)
vpr.plot_hist(pindex=8, bins=50, title='Mantle 3', xlabel='vs (km/s)', showfig=False)
vpr.plot_hist(pindex=9, bins=50, title='Mantle 4', xlabel='vs (km/s)', showfig=False)
vpr.plot_hist(pindex=10, bins=50, title='Mantle 5', xlabel='vs (km/s)', showfig=False)
vpr.plot_hist(pindex=11, bins=50, title='Water thickness', xlabel='thickness (km)', showfig=False)
vpr.plot_hist(pindex=12, bins=50, title='Sediment thickness', xlabel='thickness (km)', showfig=False)
vpr.plot_hist(pindex=13, bins=50, title='Crust thickness', xlabel='thickness (km)', showfig=True)
