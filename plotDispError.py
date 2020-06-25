import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    infnames = ["./HOBITSS_inv_test/177.6_-38.4_disp_gr.txt",
                "./HOBITSS_inv_test/178.0_-38.8_disp_gr.txt",
                "./HOBITSS_inv_test/178.6_-38.8_disp_gr.txt",]
    infits   = ["JeffModel/177.6_-38.4_synDisp.txt",
                "JeffModel/178.0_-38.8_synDisp.txt",
                "JeffModel/178.6_-38.8_synDisp.txt",]
    #infits2  = ["Eberhart-PhillipsModel/177.6_-38.4_synDisp.txt",
    #            "Eberhart-PhillipsModel/178.0_-38.8_synDisp.txt",
    #            "Eberhart-PhillipsModel/178.6_-38.8_synDisp.txt",]
    colors = ['purple', 'brown', 'red']
    for i, infname in enumerate(infnames):
        indata = np.loadtxt(infname)
        infit = np.loadtxt(infits[i])
        #infit2= np.loadtxt(infits2[i])
        _,caps,_ = plt.errorbar(indata[:,0], indata[:,1], yerr=indata[:,2], fmt='.', color=colors[i], ecolor=colors[i], elinewidth=2, capthick=3, uplims=True, lolims=True)
        plt.plot(infit[:,0], infit[:,1], color=colors[i], lw=2)
       # plt.plot(infit2[:,0], infit2[:,1], '--', color=colors[i], lw=2)
        for cap in caps:
            cap.set_marker('_')
    plt.xlabel('Periods (sec)', fontsize=20)
    plt.ylabel('Velocity (km/s)', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title("Phase velocity dispersion curves", fontsize=28)
    plt.show()
