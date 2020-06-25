import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

indata1 = np.loadtxt("177.6_-38.4_JeffersonModel.txt")
indata1_2 = np.loadtxt("/work3/wang/code_bkup/pyMCinv/Eberhart-PhillipsModel/Velprofile_177.6_-38.4.txt")
indata2 = np.loadtxt("178.0_-38.8_JeffersonModel.txt")
indata2_2 = np.loadtxt("/work3/wang/code_bkup/pyMCinv/Eberhart-PhillipsModel/Velprofile_178.0_-38.8.txt")
indata3 = np.loadtxt("178.6_-38.8_JeffersonModel.txt")
indata3_2 = np.loadtxt("/work3/wang/code_bkup/pyMCinv/Eberhart-PhillipsModel/Velprofile_178.6_-38.8.txt")

itp_func1_1 = interp1d(indata1[:,2], indata1[:,4])
itp_func1_2 = interp1d(indata1_2[:,-1], indata1_2[:,3])
itp_func2_1 = interp1d(indata2[:,2], indata2[:,4])
itp_func2_2 = interp1d(indata2_2[:,-1], indata2_2[:,3])
itp_func3_1 = interp1d(indata3[:,2], indata3[:,4])
itp_func3_2 = interp1d(indata3_2[:,-1], indata3_2[:,3])

h=np.linspace(1, 85, 100)

plt.figure(figsize=(8,12))
plt.plot((itp_func1_1(h)-itp_func1_2(h))/itp_func1_2(h)*100, h, color='purple', lw=4)
plt.plot((itp_func2_1(h)-itp_func2_2(h))/itp_func2_2(h)*100, h, color='brown', lw=4)
plt.plot((itp_func3_1(h)-itp_func3_2(h))/itp_func3_2(h)*100, h, color='red', lw=4)
plt.grid(True)
plt.xlabel('Vs difference (%)', fontsize=20)
plt.ylabel('Depth (km)', fontsize=20)
plt.xlim(-15, 15)
plt.ylim(0, 85)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title("Vp difference", fontsize=28)
plt.gca().invert_yaxis()
plt.show()
