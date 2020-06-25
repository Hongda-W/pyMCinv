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

inv_fnames_M = glob.glob("./water_hongda_temp_both/mc_inv.M*.npz")
inv_fnames_N = glob.glob("./water_hongda_temp_both/mc_inv.N*.npz")
inv_fnames_S = glob.glob("./water_hongda_temp_both/mc_inv.S*.npz")

num_N = len(inv_fnames_N)
num_M = len(inv_fnames_M)
num_S = len(inv_fnames_S)

age_N = np.zeros(num_N,dtype=np.float64)
age_M = np.zeros(num_M,dtype=np.float64)
age_S = np.zeros(num_S,dtype=np.float64)

c_age_N  = np.zeros(num_N,dtype=np.float64) # cooling age
c_age_M  = np.zeros(num_M,dtype=np.float64)
c_age_S  = np.zeros(num_S,dtype=np.float64)

A_N = np.zeros(num_N,dtype=np.float64) # Amplitude of Q
A_M = np.zeros(num_M,dtype=np.float64)
A_S = np.zeros(num_S,dtype=np.float64)

H_N = np.zeros(num_N,dtype=np.float64) # crustal thickness
H_M = np.zeros(num_M,dtype=np.float64)
H_S = np.zeros(num_S,dtype=np.float64)

for i, inv_fname in enumerate(inv_fnames_N):
	loc = inv_fname.split('.')[-2]
	f = open('JdF_Profile_temp_both/water_'+loc+'.mod','r')
	lines = f.readlines()
	age_N[i] = float(lines[0].strip())
	waterdepth = float(lines[1].strip().split()[2])
	f.close()
	vpr = mcpost.postvpr(factor=1., thresh=0.5, age=float(lines[0].strip()), waterdepth = waterdepth)
	vpr.numbp = np.array([1, 2, 4, 3])
	vpr.mtype = np.array([5, 4, 2, 6])
	vpr.read_inv_data(inv_fname,verbose=False)
	vpr.get_vmodel()
	c_age_N[i]     = vpr.avg_model.isomod.cvel[0,-1]
	A_N[i] = vpr.avg_model.isomod.cvel[2,-1]
	H_N[i] = vpr.avg_model.isomod.thickness[-2]

for i, inv_fname in enumerate(inv_fnames_M):
    loc = inv_fname.split('.')[-2]
    f = open('JdF_Profile_temp_both/water_'+loc+'.mod','r')
    lines = f.readlines()
    age_M[i] = float(lines[0].strip())
    waterdepth = float(lines[1].strip().split()[2])
    f.close()
    vpr = mcpost.postvpr(factor=1., thresh=0.5, age=float(lines[0].strip()), waterdepth = waterdepth)
    vpr.numbp = np.array([1, 2, 4, 3])
    vpr.mtype = np.array([5, 4, 2, 6])
    vpr.read_inv_data(inv_fname,verbose=False)
    vpr.get_vmodel()
    c_age_M[i]     = vpr.avg_model.isomod.cvel[0,-1]
    A_M[i] = vpr.avg_model.isomod.cvel[2,-1]
    H_M[i] = vpr.avg_model.isomod.thickness[-2]

for i, inv_fname in enumerate(inv_fnames_S):
    loc = inv_fname.split('.')[-2]
    f = open('JdF_Profile_temp_both/water_'+loc+'.mod','r')
    lines = f.readlines()
    age_S[i] = float(lines[0].strip())
    waterdepth = float(lines[1].strip().split()[2])
    f.close()
    vpr = mcpost.postvpr(factor=1., thresh=0.5, age=float(lines[0].strip()), waterdepth = waterdepth)
    vpr.numbp = np.array([1, 2, 4, 3])
    vpr.mtype = np.array([5, 4, 2, 6])
    vpr.read_inv_data(inv_fname,verbose=False)
    vpr.get_vmodel()
    c_age_S[i]     = vpr.avg_model.isomod.cvel[0,-1]
    A_S[i] = vpr.avg_model.isomod.cvel[2,-1]
    H_S[i] = vpr.avg_model.isomod.thickness[-2]

plt.figure()
plt.plot(age_N,c_age_N,'b.',label='N',markersize=10)
plt.plot(age_M,c_age_M,'g.',label='M',markersize=10)
plt.plot(age_S,c_age_S,'r.',label='S',markersize=10)
plt.plot(np.linspace(0.2,8.5,20),np.linspace(0.2,8.5,20),color='gray')
plt.xlim(xmin=0.)
plt.ylim(ymin=0.)
plt.xlabel('Lithospherical age (ma)', fontsize=12)
plt.ylabel('Cooling age (Ma)', fontsize=12)
plt.legend()
plt.title("Cooling age vs. litho age",fontsize=14)

plt.figure()
plt.plot(age_N,A_N,'b.',label='N',markersize=10)
plt.plot(age_M,A_M,'g.',label='M',markersize=10)
plt.plot(age_S,A_S,'r.',label='S',markersize=10)
plt.xlim(xmin=0.)
plt.legend()
plt.xlabel('Lithospherical age (ma)', fontsize=12)
plt.ylabel('A', fontsize=12)
plt.title("Amp of Q vs. litho age",fontsize=14)

plt.figure()
plt.plot(age_N,H_N,'b.',label='N',markersize=10)
plt.plot(age_M,H_M,'g.',label='M',markersize=10)
plt.plot(age_S,H_S,'r.',label='S',markersize=10)
plt.xlim(xmin=0.)
plt.legend()
plt.xlabel('Lithospherical age (ma)', fontsize=12)
plt.ylabel('Crustal thickness', fontsize=12)
plt.title("Crustal thickness vs. litho age",fontsize=14)
plt.show()
