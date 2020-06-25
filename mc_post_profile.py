""" Process the inversion result for gird points on a profile
"""
import mcpost
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage.filters import gaussian_filter
import pycpt

inv_fnames = glob.glob("./water_hongda_temp/mc_inv.M*.npz")
# inv_fnames = glob.glob("./water_hongda_temp/mc_inv.N*.npz")
my_cmap = pycpt.load.gmtColormap('./cv_original.cpt')
N = len(inv_fnames)
M = 51
age_arr = np.zeros([N,M],dtype=np.float64)
vs_arr  = np.zeros([N,M],dtype=np.float64)
z_arr   = np.zeros([N,M],dtype=np.float64)
c_age  = np.zeros(N,dtype=np.float64) # cooling age

for i, inv_fname in enumerate(inv_fnames):
	loc = inv_fname.split('.')[-2]
	f = open('JdF_Profile_temp_old/water_'+loc+'.mod','r')
	lines = f.readlines()
	age = float(lines[0].strip())
	waterdepth = float(lines[1].strip().split()[2])
	f.close()
	vpr = mcpost.postvpr(age=age, waterdepth = waterdepth)
	vpr.numbp = np.array([1, 2, 4, 2])
	vpr.mtype = np.array([5, 4, 2, 6])
	vpr.read_inv_data(inv_fname,verbose=False)
	vpr.get_vmodel()
	# i_N = vpr.avg_model.grid_zArr.size
	vs_arr[i,:]  = vpr.avg_model.grid_VsvArr[-M:]
	z_arr[i,:]   = vpr.avg_model.grid_zArr[-M:]
	age_arr[i,:] = age
	c_age[i]        = vpr.avg_model.isomod.cvel[0,-1]
# print(z_arr[-1,:])
plt.figure()
plt.gca().set_facecolor('gray')
# sc = plt.scatter(age_arr, z_arr, c=vs_arr,cmap=my_cmap,vmin=4.15,vmax=4.65)
sc = plt.scatter(age_arr, z_arr, c=vs_arr,cmap=my_cmap)
plt.ylim(ymin=10.,ymax=100.)
plt.xlabel('Age (ma)', fontsize=12)
plt.ylabel('Depth (km)', fontsize=12)
plt.gca().invert_yaxis()
cb = plt.colorbar(sc, orientation="horizontal", aspect=40, pad=0.13, format='%.3f')
cb.set_label('Vs (km/s)', fontsize=10, rotation=0)
cb.set_alpha(1)
cb.draw_all()

xi  = np.linspace(np.min(age_arr),np.max(age_arr),500)
yi  = np.linspace(np.max(z_arr[:,0]),np.min(z_arr[:,-1]),500)
# dx = (np.max(age_arr[~mask]) - np.min(age_arr[~mask]) ) / 500.
# dy = (np.min(z_arr[~mask]) - np.max(z_arr[~mask]) ) / 500.
xx, yy = np.meshgrid(xi,yi)
vsi = griddata((age_arr.flatten(),z_arr.flatten()),vs_arr.flatten(),(xx,yy),method='cubic')
plt.figure()
# cm = plt.contourf(xx[:-1,:-1]+dx/2.,yy[:-1,:-1]+dy/2.,vsi[:-1,:-1],cmap=my_cmap,shading='gouraud')
vs = gaussian_filter(vsi,5,order=0)
# cm = plt.pcolormesh(xx,yy,vs,cmap=my_cmap,shading='gouraud',vmin=4.15,vmax=4.65)
cm = plt.pcolormesh(xx,yy,vs,cmap=my_cmap,shading='gouraud')
CS = plt.contour(xx, yy, vs, 5, colors='gray',linestyles='dashed')
plt.gca().clabel(CS, inline=1, fontsize=10)
plt.ylim(ymin=10,ymax=100)
plt.xlabel('Age (ma)', fontsize=12)
plt.ylabel('Depth (km)', fontsize=12)
plt.gca().invert_yaxis()
cb2 = plt.colorbar(cm, orientation="horizontal", aspect=40, pad=0.13, format='%.3f')
cb2.set_label('Vs (km/s)', fontsize=10, rotation=0)
cb2.set_alpha(1)
cb2.draw_all()

plt.figure()
plt.plot(age_arr[:,0],c_age,'.')
plt.plot(np.linspace(0.2,6,20),np.linspace(0.2,6,20),color='gray')
plt.xlim(xmin=0.)
plt.ylim(ymin=0.)
plt.xlabel('Lithospherical age (ma)', fontsize=12)
plt.ylabel('Cooling age (Ma)', fontsize=12)
plt.show()
