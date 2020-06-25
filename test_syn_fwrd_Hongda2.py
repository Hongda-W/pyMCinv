import vprofile
import modparam, copy
from scipy.interpolate import interp1d
import numpy as np

loc="178.6_-38.8"
h0=0.
vpr = vprofile.vprofile1d()
vpr.readdisp(infname='./HOBITSS_inv_test/%s_disp_gr.txt'%(loc))
vpr.readmod(infname='./JeffModel/%s.mod1'%(loc))
vpr.getpara()


vpr.get_period()
vpr.update_mod()
vpr.get_vmodel()
vpr.model.isomod.mod2para()

infname = "Eberhart-PhillipsModel/Velprofile_%s.txt"%(loc)
hrange=[8, 80]
inmodel = np.loadtxt(infname)
inds = np.logical_and(inmodel[:,-1]>=hrange[0], inmodel[:,-1]<=hrange[1])
modelh = inmodel[inds,-1]+h0
funcvs = interp1d(modelh, inmodel[inds,3], kind='cubic', fill_value=np.nan, bounds_error=False)
funcvp = interp1d(modelh, inmodel[inds,2], kind='cubic', fill_value=np.nan, bounds_error=False)
hmodel = vpr.model.h.cumsum()
modelvs = funcvs(hmodel)
modelvp = funcvp(hmodel)
vpr.model.vsv[modelvs==modelvs] = modelvs[modelvs==modelvs]
vpr.model.vpv[modelvp==modelvp] = modelvp[modelvp==modelvp]

modelArr=np.array([np.append(0,vpr.model.h.cumsum()),np.append(vpr.model.vsv[0],vpr.model.vsv),np.append(vpr.model.vpv[0],vpr.model.vpv)]).T
with open("Eberhart-PhillipsModel/%s_Model_mod.txt"%(loc), 'w') as outf:
    outf.write('\n'.join([' '.join(["%.2f" % x for x in ele]) for ele in modelArr]))

vpr.compute_fsurf()

vpr.data.dispR.pvelo    = vpr.data.dispR.pvelp.copy()
vpr.data.dispR.writedisptxt('./Eberhart-PhillipsModel/%s_synDisp.txt'%(loc), predisp=False)

