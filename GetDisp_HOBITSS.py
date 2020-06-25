# get Rayleigh wave group velocity measurements around a certain point
import numpy as np
import matplotlib.pyplot as plt

def smooth(data, window_len=3, window="hanning"):
    s = np.r_[data[window_len-1:0:-1], data, data[-2:-window_len-1:-1]]
    w = eval("np."+window+"(window_len)")
    data_smooth = np.convolve(w/w.sum(), s, mode="valid")
    hl = int(window_len / 2)
    return data_smooth[hl:-hl]

def get_dists(latArr, lonArr, lat0, lon0):
    R = 6371.008 # Earth's radius
    latArr = np.radians(latArr)
    lonArr = np.radians(lonArr)
    lat0, lon0 = map(np.radians, [lat0, lon0])
    a = np.sin((latArr-lat0)/2)**2 + np.cos(latArr)*np.cos(lat0)*np.sin((lonArr-lon0)/2)**2
    c = 2*np.arctan2(a**0.5, (1-a)**0.5)
    return R*c

def rwa(data, window_len=7):
    s = np.r_[data[window_len-1:0:-1], data, data[-2:-window_len-1:-1]]
    out = np.convolve(s, np.ones((window_len,))/window_len, mode='valid')
    hl = int(window_len / 2)
    return out[hl:-hl]


if __name__ == "__main__":
    pers = np.arange(4,12.01,0.4)
    lon1 = 178.7
    lat1 = -38.8
    lon2 = 177.5
    lat2 = -38.3
    lon3 = 177.8
    lat3 = -38.9 
    #vels1 = np.zeros((pers.size, 9), dtype=float)
    #vels2 = np.zeros((pers.size, 9), dtype=float)
    pers1 = np.array([])
    pers2 = np.array([])
    pers3 = np.array([])
    velm1 = np.array([])
    velm2 = np.array([])
    velm3 = np.array([])
    vele1 = np.array([])
    vele2 = np.array([])
    vele3 = np.array([])
    outname1 = "HOBITSS_inv_test/%.1f_%.1f_disp_gr.txt"%(lon1, lat1)
    outname2 = "HOBITSS_inv_test/%.1f_%.1f_disp_gr.txt"%(lon2, lat2)
    outname3 = "HOBITSS_inv_test/%.1f_%.1f_disp_gr.txt"%(lon3, lat3)
    outfile1 = open(outname1, 'w')
    outfile2 = open(outname2, 'w')
    outfile3 = open(outname3, 'w')
    for i, per in enumerate(pers):
        filename = "/work3/wang/code_bkup/AgeJdF/ray_tomo_working_HOBITSS/%g_gr/QC_300_20_40_%g.vel_minden10"%(per, per)
        inarr = np.loadtxt(filename)
        latArr =  inarr[:,1].reshape(51,71)
        lonArr = inarr[:,0].reshape(51,71)
        velArr = inarr[:,2].reshape(51,71)
        #ind1_1 = np.where(latArr[:,0] == lat1)[0][0]
        #ind1_2 = np.where(lonArr[0,:] == lon1)[0][0]
        #ind2_1 = np.where(latArr[:,0] == lat2)[0][0]
        #ind2_2 = np.where(lonArr[0,:] == lon2)[0][0]
        #vels1[i,:] = velArr[ind1_1-1:ind1_1+2, ind1_2-1:ind1_2+2].flatten()
        #vels2[i,:] = velArr[ind2_1-1:ind2_1+2, ind2_2-1:ind2_2+2].flatten()
        #velv1 = vels1[i, ~np.isnan(vels1[i,:])]
        #velv2 = vels2[i, ~np.isnan(vels2[i,:])]
        #if velv1.size > 4:
        #    pers1 = np.append(pers1, per)
        #    velm1 = np.append(velm1, velv1.mean())
        #    vele1 = np.append(vele1, velv1.std())
        #if velv2.size > 4:
        #    pers2 = np.append(pers2, per)
        #    velm2 = np.append(velm2, velv2.mean())
        #    vele2 = np.append(vele2, velv2.std())
        distArr1 = get_dists(latArr, lonArr, lat1, lon1)
        distArr2 = get_dists(latArr, lonArr, lat2, lon2)
        distArr3 = get_dists(latArr, lonArr, lat3, lon3)
        inds1 = distArr1 < 20.
        inds2 = distArr2 < 20.
        inds3 = distArr3 < 20.
        velv1 = velArr[inds1]
        velv2 = velArr[inds2]
        velv3 = velArr[inds3]
        pers1 = np.append(pers1, per)
        velm1 = np.append(velm1, velv1[velv1==velv1].mean())
        vele1 = np.append(vele1, velv1[velv1==velv1].std())
        pers2 = np.append(pers2, per)
        velm2 = np.append(velm2, velv2[velv2==velv2].mean())
        vele2 = np.append(vele2, velv2[velv2==velv2].std())
        pers3 = np.append(pers3, per)
        velm3 = np.append(velm3, velv3[velv3==velv3].mean())
        vele3 = np.append(vele3, velv3[velv3==velv3].std())
    #vel1_smooth = smooth(velm1, window_len=9, window="hanning")
    #vel2_smooth = smooth(velm2, window_len=9, window="hanning")
    #vel3_smooth = smooth(velm3, window_len=9, window="hanning")
    # running window average
    window_len = 9
    vel1_smooth = rwa(velm1, window_len)
    vele1_smt   = rwa(vele1, window_len)
    vel2_smooth = rwa(velm2, window_len)
    vele2_smt   = rwa(vele2, window_len)
    vel3_smooth = rwa(velm3, window_len)
    vele3_smt   = rwa(vele3, window_len)
    for j in range(pers1.size):
        outfile1.write("%g %g %g\n"%(pers1[j], vel1_smooth[j], vele1_smt[j]))
    outfile1.close()
    for k in range(pers2.size):
        outfile2.write("%g %g %g\n"%(pers2[k], vel2_smooth[k], vele2_smt[k]))
    outfile2.close()
    for l in range(pers3.size):
        outfile3.write("%g %g %g\n"%(pers3[l], vel3_smooth[l], vele3_smt[l]))
    outfile3.close()
    plt.errorbar(pers1, vel1_smooth, yerr=vele1_smt)
    plt.errorbar(pers2, vel2_smooth, yerr=vele2_smt)
    plt.errorbar(pers3, vel3_smooth, yerr=vele3_smt)
    plt.show()
