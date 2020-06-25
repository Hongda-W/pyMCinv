###############################################################################################
"""
Calculate shear velocities from a temperature profile for the oceanic upper mantle using burnman
geodynamics python package.
Compositional model for the oceanic upper mantle: 75% Olivine, 21% Orthopyroxne,
				3.5% Clinopyroxene, 0.5% Spinel with an Iron-to-(Magnesium+Iron) ration of 10%.
															[Ritzwoller et al. 2004 EPSL]
"""
###############################################################################################
import os
import sys
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('/work3/wang/code_bkup/burnman'))
import numpy as np# test burnman for oceanic upper mantle, composition model from [Shapiro & Ritzwoller, 2004].
import matplotlib.pyplot as plt
import burnman
from burnman import minerals

class OceanMant(object):
	def __init__(self, amt_of_minerals=np.array([0.75,0.21,0.035,0.005]),iron=0.1):
		""" initiate the oceanic upper mantle mineralogical compositions
		Parameters: amt_of_minerals  --  mineralogical compostion of the four minerals: olivine, orthopyroxene, clinopyroxene and spinel
					iron             --  iron to (iron+mag) ration in the minerals
		"""
		olivine=minerals.SLB_2011.ol()
		# print(olivine.endmembers)
		olivine.set_composition([1.0 - iron, iron])
	
		ortho=minerals.SLB_2011.orthopyroxene()
		ortho.set_composition([1.0-iron, iron,0.,0.])
	
		clino=minerals.SLB_2011.clinopyroxene()
		clino.set_composition([1.0-iron, iron,0.,0.,0.])
	
		spinel=minerals.SLB_2011.spinel_group()
		spinel.set_composition([1.0 - iron, iron])
		amt_of_minerals=np.array([0.75,0.21,0.035,0.005])
		self.rock = burnman.Composite([olivine, ortho, clino, spinel],amt_of_minerals)
		self.rock.set_method('slb3')
		return
	def get_vs_aha(self,pressure,temperature):
		""" get aharmonic shear velocities in the mantle
		"""
		density, vp, vs, vphi, K, G = self.rock.evaluate(['density','v_p','v_s','v_phi','K_S','G'], pressure, temperature)
		# vs_ref = 4.77+0.0380*pressure/1.e9-0.000378*(temperature-300.) # approximation from [Stixrude et al. 2005]
		# plt.figure()
		# plt.subplot(1,3,1)
		# plt.plot(pressure/1.e9,vs/1.e3,color='b',linestyle='-',marker='o', markerfacecolor='b',markersize=4,label='computation')
		# plt.plot(pressure/1.e9,vs_ref,color='g',linestyle='-',marker='o', markerfacecolor='g',markersize=4,label='approximation')
		# plt.title("S wave speed (km/s)")
		# plt.xlim(min(pressure)/1.e9,max(pressure)/1.e9)
		# plt.xlabel('pressure (GPa)')
		# plt.legend(loc='lower right')
		# plt.show()
		return vs/1.e3
