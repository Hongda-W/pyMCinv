# -*- coding: utf-8 -*-
"""
A revised Module of Lili's modparam.py where I implement temperaturen parameterization for the oceanic mantle
"""
import numpy as np
import numba
import math
import random
from scipy.optimize import lsq_linear
from scipy.special import erf
from scipy.interpolate import interp1d, CubicSpline
import sys
sys.path.append('/work3/wang/code_bkup/pyMCinv/temp_2_vel')
import Mantle_temp_vel
import burn_thermal_ela
	
class para1d(object):
	"""
	An object for handling parameter perturbations
	=====================================================================================================================
	::: parameters :::
	:   values  :
	npara       - number of parameters for perturbations
	maxind      - maximum number of index for each parameters
	isspace     - if space array is computed or not
	:   arrays  :
	paraval     - parameter array for perturbation
	paraindex   - index array indicating numerical setup for each parameter
				1.  isomod
					paraindex[0, :] - type of parameters
										0   - velocity coefficient for splines or
															  (thermal age and potential temperature of oceanic mantle)
										1   - thickness
										-1   - vp/vs ratio
					paraindex[1, :] - index for type of amplitude for parameter perturbation
										1   - absolute
										else- relative
					paraindex[2, :] - amplitude for parameter perturbation (absolute/relative)
					paraindex[3, :] - step for parameter space 
					paraindex[4, :] - index for the parameter in the model group   
					paraindex[5, :] - index for spline basis/grid point, ONLY works when paraindex[0, :] == 0
				2.  ttimod
					paraindex[0, :] - type of parameters
										0   - vph coefficient for splines
										1   - vpv coefficient for splines
										2   - vsh coefficient for splines
										3   - vsv coefficient for splines
										4   - eta coefficient for splines
										5   - dip
										6   - strike
										
										below are currently not used yet
										7   - rho coefficient for splines
										8   - thickness
										-1  - vp/vs ratio
					paraindex[1, :] - index for type of amplitude for parameter perturbation
										1   - absolute
										else- relative
					paraindex[2, :] - amplitude for parameter perturbation (absolute/relative)
					paraindex[3, :] - step for parameter space 
					paraindex[4, :] - index for the parameter in the model group   
					paraindex[5, :] - index for spline basis/grid point, ONLY works when paraindex[0, :] == 0
	space       - space array for defining perturbation range
					space[0, :]     - min value
					space[1, :]     - max value
					space[2, :]     - step, used as sigma in Gaussian random generator
	=====================================================================================================================
	"""
	
	def __init__(self):
		self.npara          = 0
		self.maxind         = 6
		self.isspace        = False
		return
	
	def init_arr(self, npara):
		"""
		initialize the arrays
		"""
		self.npara          = npara
		self.paraval        = np.zeros(self.npara, dtype=np.float64)
		self.paraindex      = np.zeros((self.maxind, self.npara), dtype = np.float64)
		self.space          = np.zeros((3, self.npara), dtype = np.float64)
		return
	
	def readparatxt(self, infname):
		"""
		read txt perturbation parameter file
		==========================================================================
		::: input :::
		infname - input file name
		==========================================================================
		"""
		npara       = 0
		i           = 0
		for l1 in open(infname,"r"):
			npara   += 1
		print("Number of parameters for perturbation: %d " % npara)
		self.init_arr(npara)
		with open(infname, 'r') as fid:
			for line in fid.readlines():
				temp                        = np.array(line.split(), dtype=np.float64)
				ne                          = temp.size
				for j in range(ne):
					self.paraindex[j, i]    = temp[j]
				i                           += 1
		# print "read para over!"
		return
		
		
	def write_paraval_txt(self, outfname):
		np.savetxt(outfname, self.paraval, fmt='%g')
		return
	
	def read_paraval_txt(self, infname):
		self.paraval  = np.loadtxt(infname, dtype=np.float64)
		return
	
	def new_paraval(self, ptype):
		"""
		perturb parameters in paraval array
		===============================================================================
		::: input :::
		ptype   - perturbation type
					0   - uniform random value generated from parameter space
					1   - Gauss random number generator given mu = oldval, sigma=step
		===============================================================================
		"""
		if not self.isspace:
			print('Parameter space for perturbation has not been initialized yet!')
			return False
		if ptype == 0:
			for i in range(self.npara):
				if int(self.paraindex[1, i]) == 0:
					continue
				tval            = random.random()
				self.paraval[i] = tval * ( self.space[1, i] - self.space[0, i] ) + self.space[0, i]
		elif ptype == 1:
			for i in range(self.npara):
				oldval 	= self.paraval[i]
				step 	= self.space[2, i]
				run 	= True
				j		= 0
				while (run and j<10000): 
					newval      = random.gauss(oldval, step)
					if (newval >= self.space[0, i] and newval <= self.space[1, i]):
						run     = False
					j           += 1
				self.paraval[i] = newval
		else:
			raise ValueError('Unexpected perturbation type!')
		return True

####################################################
# auxiliary functions
####################################################

@numba.jit(numba.float64[:, :](numba.int64, numba.int64, numba.float64, numba.float64, numba.int64, numba.int64))
def bspl_basis(nBs, degBs, zmin_Bs, zmax_Bs, disfacBs, npts):
    """
    function that generate B spline basis
    """
    #-------------------------------- 
    # defining the knot vector
    #--------------------------------
    m           = nBs-1+degBs
    t           = np.zeros(m+1, dtype=np.float64)
    for i in range(degBs):
        t[i]    = zmin_Bs + i*(zmax_Bs-zmin_Bs)/10000.
    for i in range(degBs,m+1-degBs):
        n_temp  = m+1-degBs-degBs+1
        if (disfacBs !=1):
            temp= (zmax_Bs-zmin_Bs)*(disfacBs-1)/(math.pow(disfacBs,n_temp)-1)
        else:
            temp= (zmax_Bs-zmin_Bs)/n_temp
        t[i]    = temp*math.pow(disfacBs,(i-degBs)) + zmin_Bs
    for i in range(m+1-degBs,m+1):
        t[i]    = zmax_Bs-(zmax_Bs-zmin_Bs)/10000.*(m-i)
    # depth array
    step        = (zmax_Bs-zmin_Bs)/(npts-1)
    depth       = np.zeros(npts, dtype=np.float64)
    for i in range(npts):
        depth[i]= np.float64(i) * np.float64(step) + np.float64(zmin_Bs)
    # arrays for storing B spline basis
    obasis      = np.zeros((np.int64(m), np.int64(npts)), dtype = np.float64)
    nbasis      = np.zeros((np.int64(m), np.int64(npts)), dtype = np.float64)
    #-------------------------------- 
    # computing B spline basis
    #--------------------------------
    for i in range (m):
        for j in range (npts):
            if (depth[j] >=t[i] and depth[j]<t[i+1]):
                obasis[i][j]= 1
            else:
                obasis[i][j]= 0
    for pp in range (1,degBs):
        for i in range (m-pp):
            for j in range (npts):
                nbasis[i][j]= (depth[j]-t[i])/(t[i+pp]-t[i])*obasis[i][j] + \
                        (t[i+pp+1]-depth[j])/(t[i+pp+1]-t[i+1])*obasis[i+1][j]
        for i in range (m-pp):
            for j in range (npts):
                obasis[i][j]= nbasis[i][j]
    nbasis[0][0]            = 1
    nbasis[nBs-1][npts-1]   = 1
    return nbasis


class isomod(object):
	"""
	An object for handling parameterization of 1d isotropic model for the inversion
	=====================================================================================================================
	::: parameters :::
	:   numbers     :
	nmod        - number of model groups
	maxlay      - maximum layers for each group (default - 100)
	maxspl      - maximum spline coefficients for each group (default - 20)
	:   1D arrays   :
	numbp       - number of control points/basis (1D int array with length nmod)
	mtype       - model parameterization types (1D int array with length nmod)
					1   - layer         - nlay  = numbp, hArr = ratio*thickness, vs = cvel
					2   - B-splines     - hArr  = thickness/nlay, vs    = (cvel*spl)_sum over numbp
					4   - gradient layer- nlay is defined depends on thickness
											hArr  = thickness/nlay, vs  = from cvel[0, i] to cvel[1, i]
					5   - water         - nlay  = 1, vs = 0., hArr = thickness
	thickness   - thickness of each group (1D float array with length nmod)
	nlay        - number of layres in each group (1D int array with length nmod)
	vpvs        - vp/vs ratio in each group (1D float array with length nmod)
	isspl       - flag array indicating the existence of basis B spline (1D int array with length nmod)
					0 - spline basis has NOT been computed
					1 - spline basis has been computed
	:   multi-dim arrays    :
	t           - knot vectors for B splines (2D array - [:(self.numb[i]+degBs), i]; i indicating group id)
	spl         - B spline basis array (3D array - [:self.numb[i], :self.nlay[i], i]; i indicating group id)
					ONLY used for mtype == 2
	ratio       - array for the ratio of each layer (2D array - [:self.nlay[i], i]; i indicating group id)
					ONLY used for mtype == 1
	cvel        - velocity coefficients (2D array - [:self.numbp[i], i]; i indicating group id)
					layer mod   - input velocities for each layer
					spline mod  - coefficients for B spline
					gradient mod- top/bottom layer velocity
	:   model arrays        :
	vs          - vs velocity arrays (2D array - [:self.nlay[i], i]; i indicating group id)
	hArr        - layer arrays (2D array - [:self.nlay[i], i]; i indicating group id)
	:   para1d  :
	para        - object storing parameters for perturbation
	=====================================================================================================================
	"""
	
	def __init__(self):
		self.nmod       = 0
		self.maxlay     = 100
		self.maxspl     = 20
		self.para       = para1d()
		return
	
	def init_arr(self, nmod, age):
		"""
		initialization of arrays
		"""
		self.nmod       = nmod
		self.age        = age
		# arrays of size nmod
		self.numbp      = np.zeros(self.nmod, dtype=np.int64)
		self.mtype      = np.zeros(self.nmod, dtype=np.int64)
		self.thickness  = np.zeros(self.nmod, dtype=np.float64)
		self.nlay       = np.ones(self.nmod, dtype=np.int64)*20
		self.vpvs       = np.ones(self.nmod, dtype=np.float64)*1.75
		self.isspl      = np.zeros(self.nmod, dtype=np.int64)
		# arrays of size maxspl, nmod
		self.cvel       = np.zeros((self.maxspl, self.nmod), dtype = np.float64)
		# arrays of size maxlay, nmod
		self.ratio      = np.zeros((self.maxlay, self.nmod), dtype = np.float64)
		self.vs         = np.zeros((self.maxlay, self.nmod), dtype = np.float64)
		self.hArr       = np.zeros((self.maxlay, self.nmod), dtype = np.float64)
		# arrays of size maxspl, maxlay, nmod
		self.spl        = np.zeros((self.maxspl, self.maxlay, self.nmod), dtype = np.float64)
		return
	
	def readmodtxt(self, infname):
		"""
		Read model parameterization from a txt file
		first line: oceanic age, continental location if > 180  
		column 1: id
		column 2: flag  - layer(1)/B-splines(2/3)/gradient layer(4)/water(5)/temperature parameterization(6)
		column 3: thickness
		column 4: number of control points for the group
		column 5 - (4+tnp): value
		column 4+tnp - 4+2*tnp: ratio
		column -1: vpvs ratio
		==========================================================================
		::: input :::
		infname - input file name
		==========================================================================
		"""
		nmod        = 0
		for l1 in open(infname,"r"):
			nmod    += 1
		print("Number of model parameter groups: %d " % (nmod-1))
		# step 1
		for num, l1 in enumerate(open(infname,"r")):
			if num == 0: # the first line gives age
				self.init_arr(nmod-1, float(l1.split('#',1)[0].rstrip()))
				continue
			l1 			            = l1.rstrip()
			l2 			            = l1.split()
			iid 		            = int(l2[0])
			flag		            = int(l2[1])
			thickness	            = float(l2[2])
			tnp 		            = int(l2[3]) # number of parameters
			# step 2
			self.mtype[iid]	        = flag
			self.thickness[iid]     = thickness
			self.numbp[iid]         = tnp 
			if (int(l2[1]) == 5):  # water layer			
				if (tnp != 1):
					print('Water layer! Only one value for Vp')
					return False
			if (int(l2[1]) == 4):
				if (tnp != 2):
					print('Error: only two values needed for gradient type, and one value for vpvs')
					print(tnp)
					return False
			if (int(l2[1]) == 6):
				if (tnp != 3):
					print('Error: only 3 values needed for temperature parameterization, and one value for vpvs')
					print(tnp)
					return False
			if ( (int(l2[1])==1 and len(l2) != 4+2*tnp + 1) or (int(l2[1]) == 2 and len(l2) != 4+tnp + 1) ): # tnp parameters (+ tnp ratio for layered model) + 1 vpvs parameter
				print('wrong input !!!')
				return False
			nr          = 0
			# step 3
			for i in range(tnp):
				self.cvel[i, iid]       = float(l2[4+i])
				if (int(l2[1]) ==1):  # type 1 layer
					self.ratio[nr, iid] = float(l2[4+tnp+i])
					nr  += 1
			# step 4
			self.vpvs[iid]         = (float(l2[-1]))-0.
		return True
	
	def parameterize_oceanic_plate(self, wtrdepth, sedthk, lithoage, vp_water=1.475, maxdepth=100.):
		"""
		Parameterize the oceanic temperature model according to the given conditions.
		first line: oceanic age, continental location if > 180.
		==========================================================================
		There are 4 layers for the oceanic vs modelling, water, sediment, crust and mantle
		::: input :::
		
		==========================================================================
		"""
		vel_sed0 = 0.1 # vs on the top of sediment layer
		sed_a = 0.02 # parameters used for calculating shear wave speed in sediment layer, eq 5 in Ruan el al 2014
		sed_b = 1.27
		sed_c = 0.48
		vel_sed1 = (sed_a*sedthk**2+sed_b*sedthk+sed_c*vel_sed0)/(sedthk+sed_c)# vs at bottom of sediment layer
		crstthk = 7.
		nmod        = 4 # there are 4 layers
		self.init_arr(nmod, lithoage)
		flags = [5, 4, 2, 6] # flags for types of layer
		thickness = [wtrdepth, sedthk, crstthk, maxdepth-crstthk-sedthk-wtrdepth]
		tnps = [1, 2, 4, 3]
		crust_vs = [2.5, 3.15, 3.88, 3.95]
		vpvs = [0., 2., 1.76, 1.76]
		# step 1
		
		for num in range(nmod):
			self.mtype[num]	        = flags[num]
			self.thickness[num]     = thickness[num]
			self.numbp[num]         = tnps[num]
			if (flags[num] == 5):  # water layer			
				if (tnps[num] != 1):
					print('Water layer! Only one value for Vp')
					return False
				self.cvel[0, num] = vp_water
			if (flags[num] == 4):
				if (tnps[num] != 2):
					print('Error: only two values needed for gradient type, and one value for vpvs')
					print(tnp)
					return False
				self.cvel[0, num] = vel_sed0
				self.cvel[1, num] = vel_sed1
			if (flags[num] == 2):
				for ii in range(tnps[num]):
					self.cvel[ii, num] = crust_vs[ii]
			if (flags[num] == 6):
				if (tnps[num] != 3):
					print('Error: only 3 values needed for temperature parameterization, and one value for vpvs')
					print(tnp)
					return False
				self.cvel[0, num] = lithoage
				self.cvel[1, num] = 200. # initial value for top of mantle temperature
				self.cvel[2, num] = 1315.
			self.vpvs[num]         = vpvs[num]
		return True
	
	def bspline(self, i):
		"""
		Compute B-spline basis given group id
		The actual degree is k = degBs - 1
		e.g. nBs = 5, degBs = 4, k = 3, cubic B spline
		::: output :::
		self.spl    - (nBs+k, npts)
						[:nBs, :] B spline basis for nBs control points
						[nBs:, :] can be ignored
		"""    
		if self.thickness[i] >= 150:
			self.nlay[i]    = 60
		elif self.thickness[i] < 10:
			self.nlay[i]    = 5
		elif self.thickness[i] < 20:
			self.nlay[i]    = 10
		else:
			self.nlay[i]    = 50
		if self.isspl[i]:
			print('spline basis already exists!')
			return
		if self.mtype[i] != 2:
			print('Not spline parameterization!')
			return 
		# initialize
		if i >= self.nmod:
			raise ValueError('index for spline group out of range!')
			return
		nBs         = self.numbp[i]
		if nBs < 4:
			degBs   = 3
		else:
			degBs   = 4
		zmin_Bs     = 0.
		zmax_Bs     = self.thickness[i]
		disfacBs    = 2.
		npts        = self.nlay[i]
		nbasis      = bspl_basis(nBs, degBs, zmin_Bs, zmax_Bs, disfacBs, npts)
		m           = nBs-1+degBs
		if m > self.maxspl:
			raise ValueError('number of splines is too large, change default maxspl!')
		self.spl[:nBs, :npts, i]= nbasis[:nBs, :]
		self.isspl[i]           = True
		return True
	
	def mant_PT(self):
		""" get mantle pressure & temperature and amplitude of Q model from the mantle temperature parameterization
		-- deprecated
		"""
		therm_age = self.cvel[0,-1] # thermal cooling age in Ma
		pot_temp  = self.cvel[1,-1] # potential temperature for the convective mantle
		Ts        = 0. # temperature at ocean bottom in celsius
		# Tm        = 1315. # mantle temperaturn in celsius
		Tm        = 1315
		# kappa     = 1.e-6 # thermal diffusivity
		adia_k    = 0.4 # temperature gradient for the adiabatic layer
		self.update_depth()
		h_bottom = np.append(self.hArr[:self.nlay[1], 1],self.hArr[:self.nlay[2], 2])# layer thickness beneath ocean bottom
		h_bottom = np.append(h_bottom,self.hArr[:self.nlay[3], 3])
		h_surf   = np.append(self.hArr[:self.nlay[0], 0], h_bottom) # layer thickness under ocean surface
		rho_lays = np.append(1.02*np.ones(self.nlay[0]), 1.65*np.ones(self.nlay[1]))
		rho_lays = np.append(rho_lays, 3.*np.ones(self.nlay[2]))
		rho_lays = np.append(rho_lays, 3.4*np.ones(self.nlay[3])) # density estimates for each layer that we'll use to calculate pressure
		z_bottom = h_bottom.cumsum()
		Pressure = (rho_lays*1.e6*h_surf*9.8).cumsum()
		temps    = Ts+(Tm-Ts)*erf(z_bottom*1.e3/(2*np.sqrt(therm_age*365*24*3600))) # kappa*1.e-6*therm_age
		temps_p  = Ts+(Tm-Ts)*erf((z_bottom+0.1)*1.e3/(2*np.sqrt(therm_age*365*24*3600)))
		dTdz     = (temps_p - temps) / 0.1
		# ind1     = np.where(dTdz > 2*adia_k)[0][-1]
		try:
			# ind1     = np.where(temps < 1100.)[0][-1] # conductive layer ends at 1100 C
			ind1 = np.where(dTdz > 10*adia_k)[0][-1] # conductive layer ends when T-Z slope too small
		except IndexError: # no temperature was less than 1100 C.
			ind1     = 0
		ind2     = np.where(z_bottom < z_bottom[ind1] * 1.7)[0][-1] # transition layer has 70% the thickness of the conductive layer
		if ind2 < z_bottom.size-1: # there's still room for adiabatic layer
		# if True:
			# if temps[ind1] > pot_temp + adia_k*z_bottom[ind1]: # temperature should not decrease with depth
			# 	pot_temp = temps[ind1] - adia_k*z_bottom[ind1]
			# 	self.cvel[1,-1] = pot_temp                       # put this selection process in isgood() function
			# dTdz1 = dTdz[ind1]
			# temps[ind2+1:] = pot_temp + adia_k*z_bottom[ind2+1:] # adiabatic thermal gradient in convective layer
			# mask = np.zeros(z_bottom.shape,dtype=bool)
			# A = np.array([[1.,z_bottom[ind1],z_bottom[ind1]**2,z_bottom[ind1]**3], [1,z_bottom[ind2+1],z_bottom[ind2+1]**2,z_bottom[ind2+1]**3],\
			# 	[0.,1.,2.*z_bottom[ind1],3.*z_bottom[ind1]**2], [0.,1.,2.*z_bottom[ind2+1],3.*z_bottom[ind2+1]**2]])
			# B = np.array([temps[ind1],temps[ind2+1],dTdz1,0.5])
			# aa,bb,cc,dd = np.linalg.solve(A, B)
			# temps[mask] = aa+bb*z_bottom[mask]+cc*z_bottom[mask]**2+dd*z_bottom[mask]**3
			temps[ind2+1:] = pot_temp + adia_k*z_bottom[ind2+1:] # adiabatic thermal gradient in convective layer
			cs = CubicSpline(np.append(z_bottom[ind1-1:ind1+1],z_bottom[ind2+1:]), np.append(temps[ind1-1:ind1+1],temps[ind2+1:]))
			temps[ind1+1:ind2+1] = cs(z_bottom[ind1+1:ind2+1])
		self.mant_temps = temps[-self.nlay[3]:]
		self.mant_press = Pressure[-self.nlay[3]:]
		self.mant_deps  = z_bottom[-self.nlay[3]:] + self.hArr[:self.nlay[0], 0].sum() # depths from surface for mantle layers
		return Pressure[-self.nlay[3]:], temps[-self.nlay[3]:]
	
	def mant_PT_new(self):
		""" get mantle pressure & temperature and amplitude of Q model from the mantle temperature parameterization
		Temperature at top of the mantle can also be varied as a parameter
		"""
		therm_age = self.cvel[0,-1] # thermal cooling age in Ma
		top_temp  = self.cvel[1,-1] # temperature at top of the mantle
		pot_temp  = self.cvel[2,-1] # potential temperature for the convective mantle
		Tm        = 1315
		# kappa     = 1.e-6 # thermal diffusivity
		adia_k    = 0.4 # temperature gradient for the adiabatic layer
		self.update_depth()
		h_bottom = np.append(self.hArr[:self.nlay[1], 1],self.hArr[:self.nlay[2], 2])# layer thickness beneath ocean bottom
		h_bottom = np.append(h_bottom,self.hArr[:self.nlay[3], 3])
		h_surf   = np.append(self.hArr[:self.nlay[0], 0], h_bottom) # layer thickness under ocean surface
		rho_lays = np.append(1.02*np.ones(self.nlay[0]), 1.65*np.ones(self.nlay[1]))
		rho_lays = np.append(rho_lays, 3.*np.ones(self.nlay[2]))
		rho_lays = np.append(rho_lays, 3.4*np.ones(self.nlay[3])) # density estimates for each layer that we'll use to calculate pressure
		z_bottom = h_bottom.cumsum()
		Pressure = (rho_lays*1.e6*h_surf*9.8).cumsum()
		z_mant = self.hArr[:self.nlay[3], 3].cumsum() # depth from the top of the mantle
		temps    = top_temp+(Tm-top_temp)*erf(z_mant*1.e3/(2*np.sqrt(therm_age*365*24*3600))) # kappa*1.e-6*therm_age
		temps_p  = top_temp+(Tm-top_temp)*erf((z_mant+0.1)*1.e3/(2*np.sqrt(therm_age*365*24*3600)))
		dTdz     = (temps_p - temps) / 0.1
		# ind1     = np.where(dTdz > 2*adia_k)[0][-1]
		try:
			# ind1     = np.where(temps < 1100.)[0][-1] # conductive layer ends at 1100 C
			ind1 = np.where(dTdz > 10*adia_k)[0][-1] # conductive layer ends when T-Z slope too small
		except IndexError: # no temperature was less than 1100 C.
			ind1     = 0
		ind2     = np.where(z_mant < z_mant[ind1] * 1.7)[0][-1] # transition layer has 70% the thickness of the conductive layer
		if ind2 < z_mant.size-1: # there's still room for adiabatic layer
		# if True:
			# if temps[ind1] > pot_temp + adia_k*z_bottom[ind1]: # temperature should not decrease with depth
			# 	pot_temp = temps[ind1] - adia_k*z_bottom[ind1]
			# 	self.cvel[1,-1] = pot_temp                       # put this selection process in isgood() function
			# dTdz1 = dTdz[ind1]
			# temps[ind2+1:] = pot_temp + adia_k*z_bottom[ind2+1:] # adiabatic thermal gradient in convective layer
			# mask = np.zeros(z_bottom.shape,dtype=bool)
			# A = np.array([[1.,z_bottom[ind1],z_bottom[ind1]**2,z_bottom[ind1]**3], [1,z_bottom[ind2+1],z_bottom[ind2+1]**2,z_bottom[ind2+1]**3],\
			# 	[0.,1.,2.*z_bottom[ind1],3.*z_bottom[ind1]**2], [0.,1.,2.*z_bottom[ind2+1],3.*z_bottom[ind2+1]**2]])
			# B = np.array([temps[ind1],temps[ind2+1],dTdz1,0.5])
			# aa,bb,cc,dd = np.linalg.solve(A, B)
			# temps[mask] = aa+bb*z_bottom[mask]+cc*z_bottom[mask]**2+dd*z_bottom[mask]**3
			temps[ind2+1:] = pot_temp + adia_k*z_mant[ind2+1:] # adiabatic thermal gradient in convective layer
			cs = CubicSpline(np.append(z_mant[ind1-1:ind1+1],z_mant[ind2+1:]), np.append(temps[ind1-1:ind1+1],temps[ind2+1:]))
			temps[ind1+1:ind2+1] = cs(z_mant[ind1+1:ind2+1])
		self.mant_temps = temps[-self.nlay[3]:]
		self.mant_press = Pressure[-self.nlay[3]:]
		self.mant_deps  = z_bottom[-self.nlay[3]:] + self.hArr[:self.nlay[0], 0].sum() # depths from surface for mantle layers
		return Pressure[-self.nlay[3]:], temps[-self.nlay[3]:]

	def get_vs_tempmant(self):
		""" get theoretical mantle vs speed under the temperature parameterization
		"""
		# Press, temps = self.mant_PT()
		Press, temps = self.mant_PT_new()
		# mant_mod = Mantle_temp_vel.OceanMant()
		# self.vs_noQ = mant_mod.get_vs(Press,temps+273.15)
		# vs_mant  = mant_mod.get_Vanel(Press,temps+273.15) # convert temperature to Kelvin
		# self.Qs  = mant_mod.Qs
		# A  = self.cvel[2,-1]+10. # Amplitude of the Q model, if regarded as another free parameter
		A = 15.
		alpha = 0.1
		T = 12. # 12 sec period
		self.vs_noQ = 4.77+0.038*Press*1.e-9-0.000378*(temps+273.15-300) # [Stixrude and Lithgow_bertelloni, 2005]
		# mant = burn_thermal_ela.OceanMant()
		# self.vs_noQ = mant.get_vs_aha(Press,temps+273.15)
		self.Qs = A * ((2*np.pi/T) ** alpha) * np.exp(alpha*(2.5e5+Press*1.e-5)/(temps+273.15)/8.314)
		self.vs_mant = self.vs_noQ * (1.-1./(2.*self.Qs*np.tan(np.pi/2.*alpha)))
		return self.vs_mant

	def update(self):
		"""
		Update model (vs and hArr arrays), from the thickness, cvel
		"""
		for i in range(self.nmod):
			if self.nlay[i] > self.maxlay:
				print ('number of layers is too large, need change default maxlay!')
				return False
			# layered model
			if self.mtype[i] == 1:
				self.nlay[i]                = self.numbp[i]
				self.hArr[:self.nlay[i], i] = self.ratio[:self.nlay[i], i] * self.thickness[i]
				self.vs[:self.nlay[i], i]   = self.cvel[:self.nlay[i], i]
			# B spline model
			elif self.mtype[i] == 2:
				self.isspl[i]               = False
				self.bspline(i)
				# # if self.isspl[i] != 1:
				# #     self.bspline(i)
				self.vs[:self.nlay[i], i]   = np.dot( (self.spl[:self.numbp[i], :self.nlay[i], i]).T, self.cvel[:self.numbp[i], i])
				self.hArr[:self.nlay[i], i] = self.thickness[i]/self.nlay[i]
				# # for ilay in range(self.nlay[i]):
				# #     tvalue 	= 0.
				# #     for ibs in xrange(self.numbp[i]):
				# #         tvalue = tvalue + self.spl[ibs, ilay, i] * self.cvel[ibs, i]
				# #     self.vs[ilay, i]    = tvalue
				# #     self.hArr[ilay, i]  = self.thickness[i]/self.nlay[i]
			# gradient layer
			elif self.mtype[i] == 4:
				nlay 	    = 4
				if self.thickness[i] >= 20.:
					nlay    = 20
				elif self.thickness[i] > 10. and self.thickness[i] < 20.:
					nlay    = int(self.thickness[i]/1.)
				elif self.thickness[i] > 2. and self.thickness[i] <= 10.:
					nlay    = int(self.thickness[i]/0.5)
				elif self.thickness[i] < 0.5:
					nlay    = 2
				dh 	                    = self.thickness[i]/float(nlay)
				dcvel 		            = (self.cvel[1, i] - self.cvel[0, i])/(nlay - 1.)
				self.vs[:nlay, i]       = self.cvel[0, i] + dcvel*np.arange(nlay, dtype=np.float64)
				self.hArr[:nlay, i]     = dh
				self.nlay[i]            = nlay
			# water layer
			elif self.mtype[i] == 5:
				nlay                    = 1
				self.vs[0, i]           = 0.
				self.hArr[0, i]         = self.thickness[i]
				self.nlay[i]            = 1
			elif self.mtype[i] == 6: # temperature parameterization
				if self.thickness[i] >= 150:
					self.nlay[i]    = 60
				elif self.thickness[i] < 10:
					self.nlay[i]    = 5
				elif self.thickness[i] < 20:
					self.nlay[i]    = 10
				else:
					self.nlay[i]    = 50
				self.vs[:self.nlay[i], i]   = self.get_vs_tempmant()
				self.hArr[:self.nlay[i], i] = self.thickness[i]/self.nlay[i]
		return True
	
	def update_depth(self):
		"""
		Update hArr arrays only, used for paramerization of a refernce input model
		"""
		for i in range(self.nmod):
			if self.nlay[i] > self.maxlay:
				printf('number of layers is too large, need change default maxlay!')
				return False
			# layered model
			if self.mtype[i] == 1:
				self.nlay[i]                = self.numbp[i]
				self.hArr[:self.nlay[i], i] = self.ratio[:self.nlay[i], i] * self.thickness[i]
			# B spline model
			elif self.mtype[i] == 2:
				self.isspl[i]   = False
				self.bspline(i)
				self.hArr[:self.nlay[i], i] = self.thickness[i]/self.nlay[i]
			# gradient layer
			elif self.mtype[i] == 4:
				nlay 	    = 4
				if self.thickness[i] >= 20.:
					nlay    = 20
				if self.thickness[i] > 10. and self.thickness[i] < 20.:
					nlay    = int(self.thickness[i]/1.)
				if self.thickness[i] > 2. and self.thickness[i] <= 10.:
					nlay    = int(self.thickness[i]/0.5)
				if self.thickness[i] < 0.5:
					nlay    = 2
				dh 	                    = self.thickness[i]/float(nlay)
				self.hArr[:nlay, i]     = dh
				self.nlay[i]            = nlay
			# water layer
			elif self.mtype[i] == 5:
				nlay                    = 1
				self.hArr[0, i]         = self.thickness[i]
				self.nlay[i]            = 1
			elif self.mtype[i] == 6: # temperature parameterization
				if self.thickness[i] >= 150:
					self.nlay[i]    = 60
				elif self.thickness[i] < 10:
					self.nlay[i]    = 5
				elif self.thickness[i] < 20:
					self.nlay[i]    = 10
				else:
					self.nlay[i]    = 50
				self.hArr[:self.nlay[i], i] = self.thickness[i]/self.nlay[i]
		return True
	
	def parameterize_input(self, zarr, vsarr, seddepth, mohodepth, maxdepth=100.,waterdepth=0.,vp_water=1.475):
		"""
		paramerization of a refernce input model
		===============================================================================
		::: input :::
		zarr, vsarr - input depth/vs array, must be the same size (unit - km, km/s)
		mohodepth   - input moho depth (unit - km)
		seddepth    - input sediment depth (unit - km)
		maxdepth    - maximum depth for the 1-D profile (default - 100 km)
		::: output :::
		self.thickness  
		self.numbp      - [1, 2, 4, 2]
		self.mtype      - [5, 4, 2, 6]
		self.vpvs       - [0, 2., 1.75, 1.75]
		self.spl
		self.cvel       - determined from input vs profile
		===============================================================================
		"""
		if zarr.size != vsarr.size:
			raise ValueError('Inconsistent input 1-D profile depth and vs arrays!')
		if waterdepth > 0.:
			self.init_arr(4)
			self.thickness[:]   = np.array([waterdepth, seddepth, mohodepth - seddepth - waterdepth, maxdepth - mohodepth])
			self.numbp[:]       = np.array([1, 2, 4, 2])
			self.mtype[:]       = np.array([5, 4, 2, 6])
			self.vpvs[:]        = np.array([0, 2., 1.75, 1.75])
		else:
			self.init_arr(3)
			self.thickness[:]   = np.array([seddepth, mohodepth - seddepth, maxdepth - mohodepth])
			self.numbp[:]       = np.array([2, 4, 5])
			self.mtype[:]       = np.array([4, 2, 2])
			self.vpvs[:]        = np.array([2., 1.75, 1.75])
		self.update_depth()
		hArr                = np.append(self.hArr[:self.nlay[0], 0], self.hArr[:self.nlay[1], 1]) # water and sediments
		hArr                = np.append(hArr, self.hArr[:self.nlay[2], 2]) # crust
		if waterdepth > 0.:
			hArr                = np.append(hArr, self.hArr[:self.nlay[3], 3]) # mantle
		nlay_total          = self.nlay.sum()
		indlay              = np.arange(nlay_total, dtype=np.int32)
		indgrid0            = indlay*2
		indgrid1            = indlay*2 + 1
		indlay2             = np.arange(nlay_total-1,dtype=np.int32)
		indgrid2            = indlay2*2+2
		depth               = hArr.cumsum()
		zinterp             = np.zeros(2*nlay_total)
		zinterp[indgrid1]   = depth
		zinterp[indgrid2]   = dpeth[:-1]
		ind_max             = np.where(zarr >= maxdepth)[0][0]
		zarr                = zarr[:(ind_max+1)]
		vsarr               = vsarr[:(ind_max+1)]
		# remove water layer
		try:
			ind_zeros       = np.where(vsarr == 0.)[0][-1]
			zarr            = zarr[(ind_zeros+1):]
			vsarr           = vsarr[(ind_zeros+1):]
		except IndexError:
			pass
		if vsarr[0] > vsarr[1]:
			vs_temp         = vsarr[0]
			vsarr[0]        = vsarr[1]
			vsarr[1]        = vs_temp
		ind_crust           = np.where(zarr >= mohodepth)[0][0]
		vs_crust            = vsarr[:(ind_crust+1)]  
		if not np.all(vs_crust[1:] >= vs_crust[:-1]):
			print('WARNING: sort the input vs array to make it monotonically increases with depth in the crust')
			vs_crust        = np.sort(vs_crust)
			vsarr           = np.append(vs_crust, vsarr[(ind_crust+1):])
		# interpolation
		vsinterp            = np.interp(zinterp, zarr, vsarr)    
		#------------------------------------
		# determine self.cvel arrays
		#------------------------------------
		# sediments
		indz0               = np.arange(nlay_total, dtype = np.int32)*2
		indz1               = np.arange(nlay_total, dtype = np.int32)*2 + 1
		#
		# debug
		z0                  = zinterp[indz0]
		z1                  = zinterp[indz1]
		h                   = z1 - z0
		if not np.allclose(h, hArr):
			raise ValueError('Debug layer array!')
		# END debug
		indlay              = np.arange(nlay_total, dtype = np.int32)*2 + 1
		vsinterp            = vsinterp[indlay]
		#------------------------------------
		# determine self.cvel arrays
		#------------------------------------
		# water layer
		if waterdepth > 0.:
			self.cvel[0, 0] = vp_water
			vsinterp[0]     = 0.
		# sediments
		if waterdepth > 0.:
			self.cvel[0, 1] = vsinterp[self.nlay[0]]
			self.cvel[1, 1] = vsinterp[self.nlay[0] + self.nlay[1]-1]
		else:
			self.cvel[0, 0] = vsinterp[0]
			self.cvel[1, 0] = vsinterp[self.nlay[0]-1]
		#---------------------------------
		# inversion with lsq_linear
		#---------------------------------
		if waterdepth > 0.:
			# crust
			A                   = (self.spl[:self.numbp[2], :self.nlay[2], 2]).T
			b                   = vsinterp[(self.nlay[0]+self.nlay[1]):(self.nlay[0]+self.nlay[1]+self.nlay[2])]
			vs0                 = max(vsinterp[self.nlay[0]+self.nlay[1]], 3.0)
			vs1                 = min(vsinterp[self.nlay[0]+self.nlay[1]+self.nlay[2] - 1], 4.2)
			x                   = lsq_linear(A, b, bounds=(vs0, vs1)).x
			self.cvel[:4, 2]    = x[:]
			# mantle
			# A                   = (self.spl[:self.numbp[3], :self.nlay[3], 3]).T
			# b                   = vsinterp[(self.nlay[0]+self.nlay[1]+self.nlay[2]):]
			# vs0                 = max(vsinterp[(self.nlay[0]+self.nlay[1]+self.nlay[2]):].min(), 4.0)
			# vs1                 = min(vsinterp[(self.nlay[0]+self.nlay[1]+self.nlay[2]):].max(), vsarr.max())
			# x                   = lsq_linear(A, b, bounds=(vs0, vs1)).x
			# self.cvel[:5, 3] = x[:]
			code # need modification currently not used 2018 NOV.
		else:
			# crust
			A                   = (self.spl[:self.numbp[1], :self.nlay[1], 1]).T
			b                   = vsinterp[self.nlay[0]:(self.nlay[0]+self.nlay[1])]
			vs0                 = max(vsinterp[self.nlay[0]], 3.0)
			vs1                 = min(vsinterp[self.nlay[0]+self.nlay[1] - 1], 4.2)
			x                   = lsq_linear(A, b, bounds=(vs0, vs1)).x
			self.cvel[:4, 1]    = x[:]
			# mantle
			A                   = (self.spl[:self.numbp[2], :self.nlay[2], 2]).T
			b                   = vsinterp[(self.nlay[0]+self.nlay[1]):]
			vs0                 = max(vsinterp[(self.nlay[0]+self.nlay[1]):].min(), 4.0)
			vs1                 = min(vsinterp[(self.nlay[0]+self.nlay[1]):].max(), vsarr.max())
			x                   = lsq_linear(A, b, bounds=(vs0, vs1)).x
			self.cvel[:5, 2]    = x[:]
		return
	
	def get_paraind(self):
		"""
		get parameter index arrays for para
		Table 1 and 2 in Shen et al. 2012
		references:
		Shen, W., Ritzwoller, M.H., Schulte-Pelkum, V. and Lin, F.C., 2012.
			Joint inversion of surface wave dispersion and receiver functions: a Bayesian Monte-Carlo approach.
				Geophysical Journal International, 192(2), pp.807-836.
		"""
		if self.age < 180:
			self.get_paraind_ocean()
			return
		numbp_sum   = self.numbp.sum()
		npara       = numbp_sum  + self.nmod - 1
		# water layer, added May 15, 2018
		if self.mtype[0] == 5:
			npara   -= 2
		#
		self.para.init_arr(npara)
		ipara       = 0
		for i in range(self.nmod):
			# water layer, added May 15, 2018
			if self.mtype[i] == 5:
				continue
			#--------------------------------
			for j in range(self.numbp[i]):
				self.para.paraindex[0, ipara]   = 0
				if i == 0 or (i == 1 and self.mtype[0] == 5): # water layer, added May 15, 2018
					# sediment, cvel space is +- 1 km/s, different from Shen et al. 2012
					self.para.paraindex[1, ipara]   = 1
					self.para.paraindex[2, ipara]   = 1.
				else:
					# +- 20 %
					self.para.paraindex[1, ipara]   = -1
					self.para.paraindex[2, ipara]   = 20.
				# 0.05 km/s 
				self.para.paraindex[3, ipara]       = 0.05
				self.para.paraindex[4, ipara]       = i
				self.para.paraindex[5, ipara]       = j
				ipara                               +=1
		if self.nmod >= 3:
			# sediment thickness
			self.para.paraindex[0, ipara]   = 1
			self.para.paraindex[1, ipara]   = -1
			self.para.paraindex[2, ipara]   = 100.
			self.para.paraindex[3, ipara]   = 0.1
			if self.mtype[0] == 5: # water layer, added May 15, 2018
				self.para.paraindex[4, ipara]   = 1
			else:
				self.para.paraindex[4, ipara]   = 0
			ipara                           += 1
		# crustal thickness
		self.para.paraindex[0, ipara]       = 1
		self.para.paraindex[1, ipara]       = -1
		self.para.paraindex[2, ipara]       = 20.
		self.para.paraindex[3, ipara]       = 1.
		if self.nmod >= 3:
			if self.mtype[0] == 5: # water layer, added May 15, 2018
				self.para.paraindex[4, ipara]   = 2.
			else:
				self.para.paraindex[4, ipara]   = 1.
		else:
			self.para.paraindex[4, ipara]   = 0.
		return
	
	def get_paraind_ocean(self):
		""" Parameterize the priori information for oceanic settings
		"""
		numbp_sum   = self.numbp.sum()
		npara       = numbp_sum  + self.nmod - 1
		# water layer, added May 15, 2018
		if self.mtype[0] == 5: # add water depth variation and amplitude of Q 
			npara   -= 1
		#
		self.para.init_arr(npara)
		ipara       = 0
		for i in range(self.nmod):
			# water layer, added May 15, 2018
			if self.mtype[i] == 5:
				continue
			elif self.mtype[i] == 6: # temperature parameterized mantle
				self.para.paraindex[1, ipara] = -1 
				self.para.paraindex[2, ipara] = 100. # cooling age can vary 100%
				self.para.paraindex[3, ipara] = 0.05
				self.para.paraindex[4, ipara] = i
				self.para.paraindex[5, ipara] = 0
				ipara                         += 1
				self.para.paraindex[1, ipara] = 1
				self.para.paraindex[2, ipara] = 300. # top of mantle temperature can vary +-300 degrees
				self.para.paraindex[3, ipara] = 5.
				self.para.paraindex[4, ipara] = i
				self.para.paraindex[5, ipara] = 1
				ipara                         += 1
				
				self.para.paraindex[1, ipara] = 1 
				self.para.paraindex[2, ipara] = 10. # potential temperature can vary +-10 degrees
				self.para.paraindex[3, ipara] = 1.
				self.para.paraindex[4, ipara] = i
				self.para.paraindex[5, ipara] = 2
				ipara                         += 1
				continue
			#--------------------------------
			for j in range(self.numbp[i]):
				self.para.paraindex[0, ipara]   = 0
				if i == 0 or (i == 1 and self.mtype[0] == 5):
					# sediment, cvel space is +- 1 km/s, different from Shen et al. 2012
					self.para.paraindex[1, ipara]   = 1
					self.para.paraindex[2, ipara]   = 1.
					self.para.paraindex[3, ipara]   = 0.01
				elif (i == 1 and self.mtype[0] == 4) or (i == 2 and self.mtype[0] == 5): # crustal layer
					self.para.paraindex[1, ipara]   = 1
					self.para.paraindex[2, ipara]   = 0.7
					self.para.paraindex[3, ipara]   = 0.02
				#######################################################################
				# else: # mantle layer
				# 	# +- 10 %
				# 	self.para.paraindex[1, ipara]   = -1
				# 	self.para.paraindex[2, ipara] = 10.
				# 	# if j == self.numbp[i]-1: # smaller change at bottom of mantle
				# 	#     self.para.paraindex[2, ipara] = 2.
				# 	# else:
				# 	#     self.para.paraindex[2, ipara] = 4.
				#######################################################################
				self.para.paraindex[4, ipara]       = i
				self.para.paraindex[5, ipara]       = j
				ipara                               +=1
		if self.mtype[0] == 5:
			# water thickness
			self.para.paraindex[0, ipara]   = 1
			self.para.paraindex[1, ipara]   = 1
			self.para.paraindex[2, ipara]   = 1.5
			self.para.paraindex[3, ipara]   = 0.02
			self.para.paraindex[4, ipara]   = 0
			ipara                           += 1
		if self.nmod >= 3:
			# sediment thickness
			self.para.paraindex[0, ipara]   = 1
			self.para.paraindex[1, ipara]   = -1
			self.para.paraindex[2, ipara]   = 100. # sediment thickness can change 100%
			self.para.paraindex[3, ipara]   = 0.02
			if self.mtype[0] == 5: # if there's water layer, added May 15, 2018
				self.para.paraindex[4, ipara]   = 1
			else:
				self.para.paraindex[4, ipara]   = 0
			ipara                           += 1
		# crustal thickness
		self.para.paraindex[0, ipara]       = 1
		self.para.paraindex[1, ipara]       = 1
		self.para.paraindex[2, ipara]       = 1.
		self.para.paraindex[3, ipara]       = 0.1
		if self.nmod >= 3:
			if self.mtype[0] == 5: # if there's water layer, added May 15, 2018
				self.para.paraindex[4, ipara]   = 2.
			else:
				self.para.paraindex[4, ipara]   = 1.
		else:
			self.para.paraindex[4, ipara]   = 0.
		return
	
	def mod2para(self,verbose=False):
		"""
		convert model to parameter arrays for perturbation
		"""
		for i in range(self.para.npara):
			ig      = int(self.para.paraindex[4, i])
			# velocity coefficient 
			if int(self.para.paraindex[0, i]) == 0:
				ip  = int(self.para.paraindex[5, i])
				val = self.cvel[ip][ig]
			# total thickness of the group
			elif int(self.para.paraindex[0, i]) == 1:
				val = self.thickness[ig]
			# vp/vs ratio
			elif int(self.para.paraindex[0, i]) == -1:
				val = self.vpvs[ig]
			else:
				print('Unexpected value in paraindex!')
			self.para.paraval[i] = val
			#-------------------------------------------
			# defining parameter space for perturbation
			#-------------------------------------------
			if not self.para.isspace:
				step        = self.para.paraindex[3, i]
				if int(self.para.paraindex[1, i]) == 1:
					valmin  = val - self.para.paraindex[2, i]
					valmax  = val + self.para.paraindex[2, i]
				else:
					valmin  = val - val*self.para.paraindex[2, i]/100.
					valmax  = val + val*self.para.paraindex[2, i]/100.
				###
				# if self.para.paraindex[0, i] == 1 and i == 12:
				#     valmin  = 0.
				#     valmax  = 5.
				###
				valmin      = max (0.,valmin) # large than 0.
				valmax      = max (valmin + 0.0001, valmax)
				if verbose:
					print("{}, {}, {}, {}, {}".format(i,self.para.paraindex[4, i],self.para.paraindex[0, i],valmin,valmax))
				# if (int(self.para.paraindex[0, i]) == 0 and i == 0 \
				# 	and int(self.para.paraindex[5, i]) == 0): # if it is the upper sedi:
				# 	valmin              = max (0.2, valmin)
				# 	valmax              = max (0.5, valmax) 
				self.para.space[0, i]   = valmin
				self.para.space[1, i]   = valmax
				self.para.space[2, i]   = step
		self.para.isspace               = True
		return
	
	def para2mod(self):
		"""
		Convert paratemers (for perturbation) to model parameters
		"""
		for i in range(self.para.npara):
			val = self.para.paraval[i]
			ig  = int(self.para.paraindex[4, i])
			# velocity coeficient for splines
			if int(self.para.paraindex[0, i]) == 0:
				ip                  = int(self.para.paraindex[5, i])
				self.cvel[ip][ig]   = val
			# total thickness of the group
			elif int(self.para.paraindex[0, i]) == 1:
				self.thickness[ig]  = val
			# vp/vs ratio
			elif int(self.para.paraindex[0, i]) == -1:
				self.vpvs[ig]       = val
			else:
				print('Unexpected value in paraindex!')
		self.thickness[-1]          = 100. - self.thickness[:-1].sum()
		return
	
	def isgood(self, m0, m1, g0, g1, gn0, gn1):
		"""
		check the model is good or not
		==========================================================================
		::: input   :::
		m0, m1  - index of group for monotonic change checking
		g0, g1  - index of group for gradient change checking
		gn0, gn1 - index for negetive gradient checking
		==========================================================================
		"""
		# velocity constrast, contraint (5) in 4.2 of Shen et al., 2012
		for i in range (self.nmod-1):
			if self.vs[0, i+1] < self.vs[-1, i]:
				return False
		if m1 >= self.nmod:
			m1  = self.nmod -1
		if m0 < 0:
			m0  = 0
		if g1 >= self.nmod:
			g1  = self.nmod -1
		if g0 < 0:
			g0  = 0
		# monotonic change
		# velocity constrast, contraint (3) and (4) in 4.2 of Shen et al., 2012
		if m0 <= m1:
			for j in range(m0, m1+1):
				vs0     = self.vs[:self.nlay[j]-1, j]
				vs1     = self.vs[1:self.nlay[j], j]
				if np.any(np.greater(vs0, vs1)):
					return False
		# gradient check
		if g0<=g1:
			for j in range(g0, g1+1):
				if self.vs[0, j] > self.vs[1, j]:
					return False
		if self.age < 180:
			try:
				mant_T_diff = self.mant_temps[1:] - self.mant_temps[:-1]
			except AttributeError:
				self.mant_PT()
				mant_T_diff = self.mant_temps[1:] - self.mant_temps[:-1]
			if mant_T_diff.min() < 0.: # temperature must increase with depth in the oceanic mantle
				return False
			# for j in range(gn0,gn1+1): # negative velocity gradient below moho for ocean (gn0 & gn1 should both be 3)
			# 	if self.vs[0,j] < self.vs[1,j]:
			# 		return False
			T_intp = np.interp([18.75,37.49,56.24], self.mant_deps-self.hArr[:self.nlay[0], 0].sum(),self.mant_temps)# linear interpolate the mantle tempreture
			Tref1 = 1262.2-34.6*self.age-0.66*self.age**2+0.0061*self.age**3 # reference temperature from the statistic for layer1 in Wei's model
			Tref2 = 1337.6+3.32*self.age-1.1*self.age**2-0.023*self.age**3 # layer2
			Tref3 = 1347.7-1.*self.age+0.3*self.age**2-0.03*self.age**3 # layer3
			if abs(T_intp[0] - Tref1) > 150.: # the temperature shouldn't be too off
				return False
			elif abs(T_intp[1] - Tref2) > 100.:
				return False
			elif abs(T_intp[2] - Tref3) > 50.:
				return False
			if self.vs[self.nlay[self.nmod-1]-1,self.nmod-1] > 4.5 or self.vs[self.nlay[self.nmod-1]-1,self.nmod-1] < 4.0: # bottom of mantle, 100km
				return False
		# temperature must increase as depth in the mantle
		# mant = Mantle_temp_vel.OceanMant()
		# mant_vs = self.vs[:self.nlay[-1],-1]
		# mant_z  = self.thickness[:-1].sum() + self.hArr[:self.nlay[-1],-1].cumsum()
		# mant_temp = mant.inv_temp(mant_z,mant_vs,damp=0.3,N=50)
		# diff_max_be4 = [aa-mant_temp[:i].max() for i,aa in enumerate(mant_temp) if i>0] # N-1 array of the difference between every element and max value before it
		# if np.asarray(diff_max_be4,dtype=float).min() < -10.:
		#     return False
		return True
	
	def get_vmodel_old(self):
		"""
		get velocity models
		==========================================================================
		::: output :::
		hArr, vs, vp, rho, qs, qp
		==========================================================================
		"""
		nlay    = self.nlay.sum()
		hArr    = np.array([], dtype = np.float64)
		vs      = np.array([], dtype = np.float64)
		vp      = np.array([], dtype = np.float64)
		rho     = np.array([], dtype = np.float64)
		qs      = np.array([], dtype = np.float64)
		qp      = np.array([], dtype = np.float64)
		depth   = np.array([], dtype = np.float64)
		for i in range(self.nmod):
			hArr    = np.append(hArr, self.hArr[:self.nlay[i], i])
			depth   = np.append(depth, (self.hArr[:self.nlay[i], i]).cumsum())
			if self.mtype[i] == 5:
				vs      = np.append(vs,  0.)
				vp      = np.append(vp,  self.cvel[0][i])
				rho     = np.append(rho, 1.02)
				qs      = np.append(qs,  10000.)
				qp      = np.append(qp,  57822.)
			elif (i == 0 and self.mtype[i] != 5) or (i == 1 and self.mtype[0] == 5):
				vs      = np.append(vs,  self.vs[:self.nlay[i], i])
				vp      = np.append(vp,  self.vs[:self.nlay[i], i]*self.vpvs[i])
				rho     = np.append(rho, 0.541 + 0.3601*self.vs[:self.nlay[i], i]*self.vpvs[i])
				qs      = np.append(qs,  80.*np.ones(self.nlay[i], dtype=np.float64))
				qp      = np.append(qp,  160.*np.ones(self.nlay[i], dtype=np.float64))
			else:
				vs      = np.append(vs,  self.vs[:self.nlay[i], i])
				vp      = np.append(vp,  self.vs[:self.nlay[i], i]*self.vpvs[i])
				rho     = np.append(rho, 0.541 + 0.3601*self.vs[:self.nlay[i], i]*self.vpvs[i])
				qs      = np.append(qs,  600.*np.ones(self.nlay[i], dtype=np.float64))
				qp      = np.append(qp,  1400.*np.ones(self.nlay[i], dtype=np.float64))
		rho[vp > 7.5]   = 3.35
		return hArr, vs, vp, rho, qs, qp, nlay
	
	def get_vmodel_lili(self):
		""" lili's old version...
		get velocity models, slightly faster than get_vmodel_old
		==========================================================================
		::: output :::
		hArr, vs, vp, rho, qs, qp
		==========================================================================
		"""
		nlay    = self.nlay.sum()
		hArr    = np.zeros(nlay, dtype = np.float64)
		vs      = np.zeros(nlay, dtype = np.float64)
		vp      = np.zeros(nlay, dtype = np.float64)
		rho     = np.zeros(nlay, dtype = np.float64)
		qs      = np.zeros(nlay, dtype = np.float64)
		qp      = np.zeros(nlay, dtype = np.float64)
		depth   = np.zeros(nlay, dtype = np.float64)
		for i in range(self.nmod):
			if i == 0:
				hArr[:self.nlay[0]]                             = self.hArr[:self.nlay[0], 0]
			elif i < self.nmod - 1:
				hArr[self.nlay[:i].sum():self.nlay[:i+1].sum()] = self.hArr[:self.nlay[i], i]
			else:
				hArr[self.nlay[:i].sum():]                      = self.hArr[:self.nlay[i], i]
			if self.mtype[i] == 5 and i == 0: # water layer
				vs[0]                   = 0.
				vp[0]                   = self.cvel[0][i]
				rho[0]                  = 1.02
				qs[0]                   = 10000.
				qp[0]                   = 57822.
			elif (i == 0 and self.mtype[i] != 5): # sediment layer
				vs[:self.nlay[0]]       = self.vs[:self.nlay[i], i]
				vp[:self.nlay[0]]       = self.vs[:self.nlay[i], i]*self.vpvs[i]
				rho[:self.nlay[0]]      = 0.541 + 0.3601*self.vs[:self.nlay[i], i]*self.vpvs[i]
				qs[:self.nlay[0]]       = 80.*np.ones(self.nlay[i], dtype=np.float64)
				qp[:self.nlay[0]]       = 160.*np.ones(self.nlay[i], dtype=np.float64)
			elif (i == 1 and self.mtype[0] == 5) and self.nmod > 2: # sediment layer
				vs[self.nlay[:i].sum():self.nlay[:i+1].sum()]   = self.vs[:self.nlay[i], i]
				vp[self.nlay[:i].sum():self.nlay[:i+1].sum()]   = self.vs[:self.nlay[i], i]*self.vpvs[i]
				rho[self.nlay[:i].sum():self.nlay[:i+1].sum()]  = 0.541 + 0.3601*self.vs[:self.nlay[i], i]*self.vpvs[i]
				qs[self.nlay[:i].sum():self.nlay[:i+1].sum()]   = 80.*np.ones(self.nlay[i], dtype=np.float64)
				qp[self.nlay[:i].sum():self.nlay[:i+1].sum()]   = 160.*np.ones(self.nlay[i], dtype=np.float64)
			elif (i == 1 and self.mtype[0] == 5) and self.nmod == 2: # crust layer
				vs[self.nlay[:i].sum():]    = self.vs[:self.nlay[i], i]
				vp[self.nlay[:i].sum():]    = self.vs[:self.nlay[i], i]*self.vpvs[i]
				rho[self.nlay[:i].sum():]   = 0.541 + 0.3601*self.vs[:self.nlay[i], i]*self.vpvs[i]
				qs[self.nlay[:i].sum():]    = 80.*np.ones(self.nlay[i], dtype=np.float64)
				qp[self.nlay[:i].sum():]    = 160.*np.ones(self.nlay[i], dtype=np.float64)
			elif i < self.nmod - 1: # crust layer
				vs[self.nlay[:i].sum():self.nlay[:i+1].sum()]   = self.vs[:self.nlay[i], i]
				vp[self.nlay[:i].sum():self.nlay[:i+1].sum()]   = self.vs[:self.nlay[i], i]*self.vpvs[i]
				rho[self.nlay[:i].sum():self.nlay[:i+1].sum()]  = 0.541 + 0.3601*self.vs[:self.nlay[i], i]*self.vpvs[i]
				qs[self.nlay[:i].sum():self.nlay[:i+1].sum()]   = 600.*np.ones(self.nlay[i], dtype=np.float64)
				qp[self.nlay[:i].sum():self.nlay[:i+1].sum()]   = 1400.*np.ones(self.nlay[i], dtype=np.float64)
			else: # mantle layer
				vs[self.nlay[:i].sum():]    = self.vs[:self.nlay[i], i]
				vp[self.nlay[:i].sum():]    = self.vs[:self.nlay[i], i]*self.vpvs[i]
				rho[self.nlay[:i].sum():]   = 0.541 + 0.3601*self.vs[:self.nlay[i], i]*self.vpvs[i]
				qs[self.nlay[:i].sum():]    = 600.*np.ones(self.nlay[i], dtype=np.float64)
				qp[self.nlay[:i].sum():]    = 1400.*np.ones(self.nlay[i], dtype=np.float64)
		depth               = hArr.cumsum()
		rho[vp > 7.5]       = 3.35
		return hArr, vs, vp, rho, qs, qp, nlay
	
	def get_vmodel(self):
		"""
		get velocity models, modified Q values
		==========================================================================
		::: output :::
		hArr, vs, vp, rho, qs, qp
		==========================================================================
		"""
		nlay    = self.nlay.sum()
		hArr    = np.zeros(nlay, dtype = np.float64)
		vs      = np.zeros(nlay, dtype = np.float64)
		vp      = np.zeros(nlay, dtype = np.float64)
		rho     = np.zeros(nlay, dtype = np.float64)
		qs      = np.zeros(nlay, dtype = np.float64)
		qp      = np.zeros(nlay, dtype = np.float64)
		depth   = np.zeros(nlay, dtype = np.float64)
		for i in range(self.nmod):
			if i == 0:
				hArr[:self.nlay[0]]                             = self.hArr[:self.nlay[0], 0]
			elif i < self.nmod - 1:
				hArr[self.nlay[:i].sum():self.nlay[:i+1].sum()] = self.hArr[:self.nlay[i], i]
			else:
				hArr[self.nlay[:i].sum():]                      = self.hArr[:self.nlay[i], i]
			if self.mtype[i] == 5 and i == 0: # water layer
				vs[0]                   = 0.
				vp[0]                   = self.cvel[0][i]
				rho[0]                  = 1.027 # density of ocean water
				qs[0]                   = 10000.
				qp[0]                   = 57823.
			elif (i == 0 and self.mtype[i] != 5): # sediment layer
				vs[:self.nlay[0]]       = self.vs[:self.nlay[i], i]
				#vp[:self.nlay[0]]       = self.vs[:self.nlay[i], i]*self.vpvs[i]
				vp[:self.nlay[0]]       = self.vs[:self.nlay[i], i]*1.23 + 1.28 # cite [Vp/Vs and Poisson's ratios in marine sediments and rocks, Hamilton 1979]
				rho[:self.nlay[0]]      = 0.541 + 0.3601*vp[:self.nlay[0]]
				qs[:self.nlay[0]]       = 80.*np.ones(self.nlay[i], dtype=np.float64)                
				qp[:self.nlay[0]]       = 1./(4./3.*(self.vpvs[i])**(-2) / qs[:self.nlay[0]] + (1-4./3.*(self.vpvs[i])**(-2))/57823.)
			elif (i == 1 and self.mtype[0] == 5) and self.nmod > 2: # sediment layer
				vs[self.nlay[:i].sum():self.nlay[:i+1].sum()]   = self.vs[:self.nlay[i], i]
				#vp[self.nlay[:i].sum():self.nlay[:i+1].sum()]   = self.vs[:self.nlay[i], i]*self.vpvs[i]
				vp[self.nlay[:i].sum():self.nlay[:i+1].sum()]       = self.vs[:self.nlay[i], i]*1.23 + 1.28 # cite [Vp/Vs and Poisson's ratios in marine sediments and rocks, Hamilton 1979]
				rho[self.nlay[:i].sum():self.nlay[:i+1].sum()]      = 0.541 + 0.3601*vp[:self.nlay[0]]
				qs[self.nlay[:i].sum():self.nlay[:i+1].sum()]   = 80.*np.ones(self.nlay[i], dtype=np.float64)
				qp[self.nlay[:i].sum():self.nlay[:i+1].sum()]   = 1./(4./3.*(self.vpvs[i])**(-2) / qs[self.nlay[:i].sum():self.nlay[:i+1].sum()]\
																	  + (1-4./3.*(self.vpvs[i])**(-2))/57823.)
			elif (i == 1 and self.mtype[0] == 5) and self.nmod == 2: # sediment layer
				vs[self.nlay[:i].sum():]    = self.vs[:self.nlay[i], i]
				#vp[self.nlay[:i].sum():]    = self.vs[:self.nlay[i], i]*self.vpvs[i]
				vp[self.nlay[:i].sum():]       = self.vs[:self.nlay[i], i]*1.23 + 1.28 # cite [Vp/Vs and Poisson's ratios in marine sediments and rocks, Hamilton 1979]
				rho[self.nlay[:i].sum():]      = 0.541 + 0.3601*vp[:self.nlay[0]]
				qs[self.nlay[:i].sum():]    = 80.*np.ones(self.nlay[i], dtype=np.float64)
				qp[self.nlay[:i].sum():]    = 1./(4./3.*(self.vpvs[i])**(-2) / qs[self.nlay[:i].sum():] + (1-4./3.*(self.vpvs[i])**(-2))/57823.)
			elif i < self.nmod - 1: # crust layer
				vs[self.nlay[:i].sum():self.nlay[:i+1].sum()]   = self.vs[:self.nlay[i], i]
				vp[self.nlay[:i].sum():self.nlay[:i+1].sum()]   = self.vs[:self.nlay[i], i]*self.vpvs[i]
				rho[self.nlay[:i].sum():self.nlay[:i+1].sum()]  = 0.541 + 0.3601*self.vs[:self.nlay[i], i]*self.vpvs[i]
				qs[self.nlay[:i].sum():self.nlay[:i+1].sum()]   = 600.*np.ones(self.nlay[i], dtype=np.float64)
				qp[self.nlay[:i].sum():self.nlay[:i+1].sum()]   = 1./(4./3.*(self.vpvs[i])**(-2) / qs[self.nlay[:i].sum():self.nlay[:i+1].sum()]\
																	  + (1-4./3.*(self.vpvs[i])**(-2))/57823.)
			else: # mantle layer
				vs[self.nlay[:i].sum():]    = self.vs[:self.nlay[i], i]
				vp[self.nlay[:i].sum():]    = self.vs[:self.nlay[i], i]*self.vpvs[i]
				rho[self.nlay[:i].sum():]   = 0.541 + 0.3601*self.vs[:self.nlay[i], i]*self.vpvs[i]
				qs[self.nlay[:i].sum():], qp[self.nlay[:i].sum():] = self.get_mantle_Q(hArr.cumsum()[self.nlay[:i].sum():], self.vpvs[i])
		depth               = hArr.cumsum()
		rho[vp > 7.5]       = 3.35
		return hArr, vs, vp, rho, qs, qp, nlay
		
	def get_mantle_Q(self,deps,vpvs):
		""" get Q value for mantle layer, follw Eq(4) from Ye (2013)
		Calculate Q value for 20 sec period, Q doesn't change a lot with period
		"""
		Tm = 1325. # deep mantle temperature in celsius
		A = 18. # A value in Ye 2013 eq(4) 
		temps = 1315.*erf(500*deps/np.sqrt(self.age*365*24*3600))+273.15 # depth dependent mantle temperature in Kelvin
		qs = A * (np.pi/10.)**0.1 * np.exp(0.1*(2.5e5+3.2*9.8*deps*10)/(8.314472*temps))
		qp = 1./(4./3.*vpvs**(-2) / qs + (1-4./3.*vpvs**(-2))/57823.)
		return qs, qp
		
	def plot_Q(self): # simple function for visually check if the Q model makes sense
		import matplotlib.pyplot as plt
		self.update()
		hArr,_,_,_,qs,qp,_ = self.get_vmodel()
		deps = hArr.cumsum()
		plt.plot(qp, deps)
		plt.gca().invert_yaxis()
		plt.show()
		return
	
	
	
