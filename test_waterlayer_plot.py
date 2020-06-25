import mcpost
import sys

if len(sys.argv) != 2:
    raise ValueError("Usage: python test_waterlayer_plot.py loc")

loc = sys.argv[1]
f = open('JdF_Profile_temp/water_'+loc+'.mod','r')
#f = open('hongda_all/water_'+loc+'.mod_temp','r')
lines = f.readlines()
age = float(lines[0].strip())
waterdepth = float(lines[1].strip().split()[2])
f.close()

vpr = mcpost.postvpr(age=age, waterdepth = waterdepth)

vpr.read_inv_data('water_hongda_temp/mc_inv.'+loc+'.npz')
vpr.read_data('water_hongda_temp/mc_data.'+loc+'.npz')
#vpr.read_inv_data('hongda_all/mc_inv.'+loc+'.npz')
#vpr.read_data('hongda_all/mc_data.'+loc+'.npz')

vpr.get_vmodel()
vpr.plot_disp(disptype='ph',showfig=False)
vpr.plot_profile(showfig=False)
vpr.plot_temp(showfig=False)
vpr.plot_hist(pindex=6, bins=50, title='Cooling Age', xlabel='Age (Ma)', showfig=False)
vpr.plot_hist(pindex=7, bins=50, title='Potential Temperature', xlabel='Temp (Celsius)', showfig=True)
#vpr.plot_hist_age(bins=50)
