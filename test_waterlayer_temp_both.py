# invert for vs using both Rayleigh wave phase and group velocities.
import vprofile
import modparam
import sys

if len(sys.argv) != 2:
    raise ValueError("Usage: python test_waterlayer.py loc")

loc = sys.argv[1]

vpr = vprofile.vprofile1d()
vpr.readdisp(infname='JdF_Profile_temp_both/'+loc+'_disp_ph.txt',dtype='ph')
vpr.readdisp(infname='JdF_Profile_temp_both/'+loc+'_disp_gr.txt',dtype='gr')
vpr.readmod(infname='JdF_Profile_temp_both/water_'+loc+'.mod')
vpr.getpara()

vpr.mc_joint_inv_iso_mp(outdir='./water_hongda_temp_both', dispdtype='both', wdisp=1., rffactor=40., pfx=loc, numbrun=30000, verbose=True)
