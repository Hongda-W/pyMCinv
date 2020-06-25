import numpy as np
import mcpost
import sys

if len(sys.argv) != 2:
    raise ValueError("Usage: python test_waterlayer_plot_HOBITSS.py loc")

loc = sys.argv[1]
f = open('HOBITSS_inv_test/water_'+loc+'.mod','r')
#f = open('hongda_all/water_'+loc+'.mod_temp','r')
lines = f.readlines()
age = float(lines[0].strip())
#waterdepth = float(lines[1].strip().split()[2])
waterdepth = 2.
f.close()

vpr = mcpost.postvpr(factor=1., thresh=1., age=age, waterdepth = 2.)
vpr.numbp = np.array([2, 2, 1])
vpr.mtype = np.array([4, 4, 1])
#vpr.numbp = np.array([2, 2, 1])
#vpr.mtype = np.array([4, 4, 1])

vpr.read_inv_data(infname='HOBITSS_inv_test/mc_inv.'+loc+'.npz', Nmax=5000, Nmin=1000)
vpr.read_data('HOBITSS_inv_test/mc_data.'+loc+'.npz')
p_name ="HOBITSS_inv_priori/mc_inv."+loc+".npz"
#vpr.read_inv_data('hongda_all/mc_inv.'+loc+'.npz')
#vpr.read_data('hongda_all/mc_data.'+loc+'.npz')

vpr.get_vmodel()
vpr.plot_disp(disptype='gr', mindisp=False, showfig=False)
vpr.plot_profile(minvpr=False, showfig=False)
vpr.plot_hist(pindex=0, bins=50, title='Sediment top', xlabel='vs (km/s)', priori=True, p_name=p_name, minline=False, showfig=False)
vpr.plot_hist(pindex=1, bins=50, title='Sediment bottom', xlabel='vs (km/s)', priori=True, p_name=p_name, minline=False, showfig=False)
vpr.plot_hist(pindex=2, bins=50, title='Crust top', xlabel='vs (km/s)', priori=True, p_name=p_name, minline=False, showfig=False)
vpr.plot_hist(pindex=3, bins=50, title='Crust bottom', xlabel='vs (km/s)', priori=True, p_name=p_name, minline=False, showfig=False)
vpr.plot_hist(pindex=4, bins=50, title='Mantle', xlabel='vs (km/s)', priori=True, p_name=p_name, minline=False, showfig=False)
vpr.plot_hist(pindex=5, bins=50, title='Sediment thickness', xlabel='thickness (km)', priori=True, p_name=p_name, minline=False, showfig=False)
vpr.plot_hist(pindex=6, bins=50, title='Crust thickness', xlabel='thickness (km)', priori=True, p_name=p_name, minline=False, showfig=True)
#vpr.plot_hist(pindex=5, bins=50, title='Sediment thickness', xlabel='thickness (km)', priori=True, p_name=p_name, minline=False, showfig=False)
##vpr.plot_hist(pindex=6, bins=50, title='Crust thickness', xlabel='thickness (km)', priori=True, p_name=p_name, minline=False, showfig=True)
