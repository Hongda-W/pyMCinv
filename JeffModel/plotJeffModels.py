import numpy as np
import matplotlib.pyplot as plt

indata1 = np.loadtxt("177.6_-38.4_JeffersonModel.txt")
indata1_2 = np.loadtxt("/work3/wang/code_bkup/pyMCinv/Eberhart-PhillipsModel/Velprofile_177.6_-38.4.txt")
indata2 = np.loadtxt("178.0_-38.8_JeffersonModel.txt")
indata2_2 = np.loadtxt("/work3/wang/code_bkup/pyMCinv/Eberhart-PhillipsModel/Velprofile_178.0_-38.8.txt")
indata3 = np.loadtxt("178.6_-38.8_JeffersonModel.txt")
indata3_2 = np.loadtxt("/work3/wang/code_bkup/pyMCinv/Eberhart-PhillipsModel/Velprofile_178.6_-38.8.txt")

plt.figure(figsize=(8,12))
plt.plot(indata1[:,3], indata1[:,2], color='purple', lw=4)
plt.plot(indata1_2[:,2], indata1_2[:,-1], '--', color='purple', lw=4)
plt.plot(indata2[:,3], indata2[:,2], color='brown', lw=4)
plt.plot(indata2_2[:,2], indata2_2[:,-1], '--', color='brown', lw=4)
plt.plot(indata3[:,3], indata3[:,2], color='red', lw=4)
plt.plot(indata3_2[:,2], indata3_2[:,-1], '--', color='red', lw=4)
"""
indata1 = np.loadtxt("177.6_-38.4_MergeModel.txt")
indata2 = np.loadtxt("178.0_-38.8_MergeModel.txt")
indata3 = np.loadtxt("178.6_-38.8_MergeModel.txt")

plt.figure(figsize=(8,12))
plt.plot(indata1[:,1], indata1[:,0], color='purple', lw=4)
plt.plot(indata2[:,1], indata2[:,0], color='brown', lw=4)
plt.plot(indata3[:,1], indata3[:,0], color='red', lw=4)
"""
plt.grid(True)
plt.xlabel('vp (km/s)', fontsize=20)
plt.ylabel('Depth (km)', fontsize=20)
plt.xlim(1.5, 10.6)
plt.ylim(-5, 100)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title("Shear wave velocity profiles", fontsize=28)
plt.gca().invert_yaxis()
plt.show()
