import vprofile
import modparam
import sys

if len(sys.argv) != 2:
    raise ValueError("Usage: python test_waterlayer.py loc")

loc = sys.argv[1]

vpr = vprofile.vprofile1d()
vpr.readdisp(infname='JdF_Profile_temp/'+loc+'_disp.txt')
vpr.readmod(infname='JdF_Profile_temp/water_'+loc+'.mod')
#vpr.readdisp(infname='hongda_all/'+loc+'_disp.txt')
#vpr.readmod(infname='hongda_all/water_'+loc+'.mod_temp')
vpr.getpara()

vpr.mc_joint_inv_iso_mp(outdir='./water_hongda_temp', wdisp=1., rffactor=40., pfx=loc, numbrun=30000, verbose=True)
#vpr.mc_joint_inv_iso_mp(outdir='./hongda_all', wdisp=1., rffactor=40., pfx=loc, numbrun=150000, verbose=True)

