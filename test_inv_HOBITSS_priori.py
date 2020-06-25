import vprofile
import modparam
import sys

if len(sys.argv) != 2:
    raise ValueError("Usage: python test_HOBITSS.py loc")

loc = sys.argv[1]

vpr = vprofile.vprofile1d()
vpr.readdisp(infname='HOBITSS_inv_test/'+loc+'_disp_gr.txt',dtype='gr')
vpr.readmod(infname='HOBITSS_inv_test/water_'+loc+'.mod')
vpr.getpara()

vpr.mc_joint_inv_iso_mp(outdir='HOBITSS_inv_priori', dispdtype='gr', wdisp=2., rffactor=40., pfx=loc, step4uwalk=1500, numbrun=150000, verbose=True)
