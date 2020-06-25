import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    infname = "./HOBITSS_inv_test/178.6_-38.8_disp_gr.txt"
    infits   = ["JeffModel/178.6_-38.8_synDisp_5kmsed.txt",
                "JeffModel/178.6_-38.8_synDisp_6kmsed.txt",
                "JeffModel/178.6_-38.8_synDisp_4kmsed.txt",
                "JeffModel/178.6_-38.8_synDisp_5kmsed1.5.txt",
                "JeffModel/178.6_-38.8_synDisp_1.76kmwater.txt",
                "JeffModel/178.6_-38.8_synDisp_0kmwater.txt",]
    labels   = ["5km 1.0km/s sediment","6km 1.0km/s sediment", "4km 1.0km/s sediment", "5km 1.5km/s sediment", "0.76+1km water", "no water"]
    indata = np.loadtxt(infname)
    _,caps,_ = plt.errorbar(indata[:,0], indata[:,1], yerr=indata[:,2], fmt='.', elinewidth=2, capthick=3, uplims=True, lolims=True)
    for cap in caps:
        cap.set_marker('_')
    for i, infname in enumerate(infits):
        infitdata = np.loadtxt(infname)
        plt.plot(infitdata[:,0], infitdata[:,1], lw=2, label=labels[i])
    plt.xlabel('Periods (sec)', fontsize=20)
    plt.ylabel('Velocity (km/s)', fontsize=20)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title("lon=178.6 lat=-38.8", fontsize=28)
    plt.show()
