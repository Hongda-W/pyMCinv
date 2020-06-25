import numpy as np
import mcpost
import sys
import ocean_surf_dbase

if len(sys.argv) != 2:
    raise ValueError("Usage: python test_waterlayer_plot.py loc")

dset = ocean_surf_dbase.invhdf5('test_surf_dbase.h5')
loc = sys.argv[1]

#f = open('JdF_Profile_temp_both/water_'+loc+'.mod','r')
#f = open('hongda_all/water_'+loc+'.mod_temp','r')
#lines = f.readlines()
#age = float(lines[0].strip())
#waterdepth = float(lines[1].strip().split()[2])
#f.close()

age = dset[loc].attrs['litho_age']
waterdepth = -dset[loc].attrs['topo']
print("Age: %g Ma"%(age))
print("Water depth: %g km"%(waterdepth))

vpr = mcpost.postvpr(factor=1., thresh=0.5, age=age, waterdepth = waterdepth)
vpr.numbp = np.array([1, 2, 4, 3])
vpr.mtype = np.array([5, 4, 2, 6])

vpr.read_inv_data('inv_workingdir_temp/mc_inv.'+loc+'.npz')
vpr.read_data('inv_workingdir_temp/mc_data.'+loc+'.npz')
#vpr.read_inv_data('water_hongda_temp_both/mc_inv.'+loc+'.npz')
#vpr.read_data('water_hongda_temp_both/mc_data.'+loc+'.npz')
#vpr.read_inv_data('hongda_all/mc_inv.'+loc+'.npz')
#vpr.read_data('hongda_all/mc_data.'+loc+'.npz')

vpr.get_vmodel()
try:
    gper_size = vpr.data.dispR.gper.size
except AttributeError:
    gper_size = 0.
try:
    pper_size = vpr.data.dispR.pper.size
except AttributeError:
    pper_size = 0.
if gper_size == 0:
    if pper_size == 0:
        raise ValueError("Error: No dispersion data!!")
    else:
        disptype = 'ph'
else:
    if pper_size == 0:
        disptype = 'gr'
    else:
        disptype = 'both'
vpr.plot_disp(disptype=disptype,showfig=False)
vpr.plot_profile(showfig=False)
vpr.plot_temp(showfig=False)
vpr.plot_hist(pindex=2, bins=50, title='Crust vs 1', xlabel='Speed (km/s)', showfig=False)
vpr.plot_hist(pindex=3, bins=50, title='Crust vs 2', xlabel='Speed (km/s)', showfig=False)
vpr.plot_hist(pindex=4, bins=50, title='Crust vs 3', xlabel='Speed (km/s)', showfig=False)
vpr.plot_hist(pindex=5, bins=50, title='Crust Vs 4', xlabel='Speed (km/s)', showfig=False)
vpr.plot_hist(pindex=6, bins=50, title='Cooling Age', xlabel='Age (Ma)', showfig=False)
vpr.plot_hist(pindex=7, bins=50, title='Top of Mantle Temperature', xlabel='Temp (Celsius)', showfig=False)
vpr.plot_hist(pindex=8, bins=50, title='Potential Temperature', xlabel='Temp', showfig=False)
vpr.plot_hist(pindex=0, bins=50, title='Sediment top', xlabel='vs (km/s)', showfig=False)
vpr.plot_hist(pindex=1, bins=50, title='Sediment bottom', xlabel='vs (km/s)', showfig=False)
vpr.plot_hist(pindex=10, bins=50, title='Sediment thickness', xlabel='thickness (km)', showfig=False)
vpr.plot_hist(pindex=11, bins=50, title='Crust thickness', xlabel='thickness (km)', showfig=True)

#vpr.plot_hist_age(bins=50)
