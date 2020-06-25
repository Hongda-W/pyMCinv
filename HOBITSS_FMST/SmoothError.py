import numpy as np


if __name__ == "__main__":
    lon = 177.6
    lat = -38.4
    loc = "%.1f_%.1f"%(lat, lon)
    outname = "../HOBITSS_inv_test/%.1f_%.1f_disp_gr.txt"%(lon, lat)
    outfile = open(outname, 'w') 
    indata = np.loadtxt("FTAN_curve_%s.txt"%(loc))
    bins   = indata[::5,0]
    for i in range(bins.size-1):
        period = ( bins[i] + bins[i+1] ) / 2
        inds =np.logical_and( indata[:,0] >= bins[i] ,indata[:,0] < bins[i+1] )
        vals = indata[inds,1]
        vel = np.mean(vals)
        sem = np.std(vals) / np.sqrt(vals.size)
        std = np.std(vals)
        outfile.write("%g %g %g\n"%(period, vel, std))
    outfile.close()
