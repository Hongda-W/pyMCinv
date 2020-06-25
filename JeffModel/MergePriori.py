import numpy as np
import sys

if __name__ == "__main__":
    loc = sys.argv[1]
    deepfname = "%s_JeffersonModel.txt"%(loc)
    shalfname = "%s_shallow_estimate.txt"%(loc)
    arrdeep = np.loadtxt(deepfname)[:,[2,4,3]]
    data = np.loadtxt(shalfname)[:,[0,2,1]]
    inds = arrdeep[:,0]>data[-1,0]
    outdata = np.vstack((data,arrdeep[inds,:]))
    np.savetxt("%s_MergeModel.txt"%(loc), outdata, fmt="%.3f")
