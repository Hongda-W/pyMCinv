import numpy as np
import mcpost
import sys
import matplotlib.pyplot as plt

# get covariance matrix of the parameters from posterior distribution

if len(sys.argv) != 2:
    raise ValueError("Usage: python CovarianceMatrix.py loc")

loc = sys.argv[1]
f = open('HOBITSS_inv_test/water_'+loc+'.mod','r')
lines = f.readlines()
age = float(lines[0].strip())
waterdepth = float(lines[1].strip().split()[2])
f.close()

vpr = mcpost.postvpr(factor=1., thresh=1., age=age, waterdepth = waterdepth)
vpr.numbp = np.array([1, 2, 2, 1])
vpr.mtype = np.array([5, 4, 4, 1])

vpr.read_inv_data('HOBITSS_inv_test/mc_inv.'+loc+'.npz')
vpr.read_data('HOBITSS_inv_test/mc_data.'+loc+'.npz')
p_name ="HOBITSS_inv_priori/mc_inv."+loc+".npz"

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

paraval = vpr.invdata[vpr.ind_thresh, 2:(vpr.npara+2)]
if not paraval.shape[1] == 8:
    raise ValueError("Shape of the paraval array incorrect!")

cov = np.corrcoef(paraval, rowvar=False)
im = plt.imshow(cov, cmap="seismic", vmin=-1, vmax=1)
plt.xticks(np.arange(8), ("vs1", "vs2", "vc1", "vc2", "vm", "Wh", "Sh", "Ch"))
plt.yticks(np.arange(8), ("vs1", "vs2", "vc1", "vc2", "vm", "Wh", "Sh", "Ch"))
plt.colorbar(im)
plt.show()
