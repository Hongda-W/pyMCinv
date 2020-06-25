###############################################################################################
"""
Calculate the seismic velocity model from a temperature model for the oceanic mantle.
Compositional model for the oceanic upper mantle: 75% Olivine, 21% Orthopyroxne,
				3.5% Clinopyroxene, 0.5% Spinel with an Iron-to-Magnesium ration of 10%.
															[Ritzwoller et al. 2004 EPSL]
"""
###############################################################################################
import numpy as np

class OceanMant(object):
	""" Class for modelling temperature and velocity of oceanic upper mantle
	"""
	def __init__(self, N=4, lambdas=np.array([0.75,0.21,0.035,0.005]),namelist=['Olivine','Orthopyroxnene','Clinopyroxene','Spinel'],X=0.1):
		self.N_mines = N # number of different types of minerals that compose the oceanic upper mantle
		self.mine_list = []
		# array the stores the necessary coefficients for different Olivine, Orthopyroxne, Clinopyroxene and Spinel
		# reference Geos et al. 2000 JGR.  table A1.
		self.paras = np.array([[3.222e3, 82.e9,  129.e9, 1.182e3, -30.e9, 0.,     -14.e6, -16.e6, 1.4, 4.2, 0.201e-4,  0.139e-7,   0.1627e-2,  -0.338],
			                   [3.198e3, 81.e9,  111.e9, 0.804e3, -29.e9, -10.e9, -11.e6, -12.e6, 2.,  6.,  0.3871e-4, 0.0446e-7,  0.0343e-2,  -1.7278],
			                   [3.28e3,  67.e9,  105.e9, 0.377e3, -6.e9,  13.e9,  -10.e6, -13.e6, 1.7, 6.2, 0.3206e-4, 0.0811e-7,  0.1347e-2,  -1.8167],
			                   [3.578e3, 108.e9, 198.e9, 0.702e3, -24.e9, 12.e9,  -12.e6, -28.e6, 0.8, 5.7, 0.6969e-4, -0.0108e-7, -3.0799e-2, 5.0395]])
		for i in range(N):
			rho_0,miu_0,K_0,drhodX,dmiudX,dKdX,dmiudT,dKdT,dmiudP,dKdP,alpha_0,alpha_1,alpha_2,alpha_3 = self.paras[i,:]
			mine = MantMineral(rho_0,miu_0,K_0,drhodX,dmiudX,dKdX,dmiudT,dKdT,dmiudP
							   ,dKdP,alpha_0,alpha_1,alpha_2,alpha_3,name=namelist[i])
			self.mine_list.append(mine)
		self.lambdas = lambdas # percentage for different minerals
		self.X = X # iron content
		self.a = 0.1
		self.H = 2.5e5
		self.A = 30
		self.V = 1.e-5
		return
	
	def get_mean_paras(self,P,T):
		if not self.lambdas.sum() == 1:
			raise ValueError("lamdas for all the {} minerals should sum to 1.".format(self.N_mines))
		self.mean_rho = 0.
		mean_miu1 = mean_miu2 = mean_K1 = mean_K2 = 0.
		for i in range(self.N_mines):
			self.mean_rho += self.mine_list[i].get_rho(P,T,self.X) * self.lambdas[i]
			mean_miu1 += self.mine_list[i].get_miu(P,T,self.X) * self.lambdas[i]
			mean_miu2 += self.lambdas[i] / self.mine_list[i].get_miu(P,T,self.X)
			mean_K1   += self.mine_list[i].get_K(P,T,self.X) * self.lambdas[i]
			mean_K2   += self.lambdas[i] / self.mine_list[i].get_K(P,T,self.X)
		self.mean_miu_voigt = mean_miu1
		self.mean_miu_reuss = 1./mean_miu2
		self.mean_K_voigt = mean_K1
		self.mean_K_reuss = 1./mean_K2
		self.mean_miu = (mean_miu1 + 1./mean_miu2) / 2.
		self.mean_K   = (mean_K1 + 1./mean_K2) / 2.
		return self.mean_rho, self.mean_miu, self.mean_K
	
	def get_vs(self,P,T):
		mean_rho, mean_miu, mean_K = self.get_mean_paras(P,T)
		self.vs = np.sqrt(mean_miu/mean_rho) / 1.e3 # convert to km/s
		return self.vs
	
	def get_vp(self):
		return np.sqrt(4./3.*self.mean_miu/self.mean_rho + self.mean_K) / 1.e3

	def get_Qs(self,P,T,omega=np.pi*2/10.): # [Goes and Govers, 2000, JGR]
		return self.A * (omega ** self.a) * np.exp(self.a*(self.H+P*self.V)/T/8.31446)
		
	def get_Vanel(self,P,T):
		self.Qs = self.get_Qs(P=P,T=T,omega=np.pi*2)
		vs = self.get_vs(P,T)
		self.Vanel = vs * (1.-1./(2.*self.Qs*np.tan(np.pi/2.*self.a)))
		return self.Vanel
	
	def inv_temp(self,Z,vs,N=50,T0=1273.15,damp=0.3):
		""" Invert for temperature from the veloicty model
		Parameters:   Z   --  depth in km
					  vs  -- shear velocity distribution
					  N   -- number of iterations for the inverion
					  T0  -- starting temperature
		"""
		P = Z*3.3e3*9.8*1.e3 # Pressure crude estimates
		temp = np.ones(vs.shape) * T0
		damps = damp * np.ones(vs.shape)
		for i in range(N):
			v_syn,Qs = self.get_Vanel(P,temp)
			dvdT_anel = 1./Qs*self.a*self.H/(2.*8.31446*temp**2*np.tan(np.pi*self.a/2.))
			dvdT_anh  = ((paras[:,6]*self.lambdas).sum() + (paras[:,6]*self.lambdas/(paras[:,1]**2)).sum() / (self.mean_miu_reuss **2)
				- ((v_syn*1.e3)**2)*((paras[:,10]*self.lambdas).sum()+(paras[:,11]*self.lambdas).sum()*temp+(paras[:,12]*self.lambdas).sum()/temp
					+(paras[:,13]*self.lambdas).sum()/temp**2)) / (2*self.mean_rho*v_syn*1.e3)
			dvdT = dvdT_anel + dvdT_anh
			temp += damps * (vs-v_syn) * 1.e3 / dvdT
		return temp
	
class MantMineral(object):
	""" Class for dealing oceanic minerals
	"""
	def __init__(self,rho_0,miu_0,K_0,drhodX,dmiudX,dKdX,dmiudT,dKdT,dmiudP,dKdP,alpha_0,alpha_1,alpha_2,alpha_3,name="Olivine"):
		self.T_0     = 273.15 # temperature on the Earth's surface
		self.P_0     = 0. # pressure on the Earth's surface
		self.name    = name
		self.rho_0   = rho_0
		self.miu_0   = miu_0
		self.K_0     = K_0
		self.drhodX  = drhodX
		self.dmiudX  = dmiudX
		self.dKdX    = dKdX
		self.dmiudT  = dmiudT
		self.dKdT    = dKdT
		self.dmiudP  = dmiudP
		self.dKdP    = dKdP
		self.alpha_0 = alpha_0
		self.alpha_1 = alpha_1
		self.alpha_2 = alpha_2
		self.alpha_3 = alpha_3
		return
	
	def get_alpha(self,T):
		return self.alpha_0 + self.alpha_1*(T-self.T_0) + self.alpha_2 / (T-self.T_0) + self.alpha_3 / ((T-self.T_0)**2)
	
	def get_rho_0X(self, X):
		return self.rho_0 + X*self.drhodX
			
	def get_miu(self, P, T, X):
		return self.miu_0 + (T-self.T_0)*self.dmiudT + (P-self.P_0)*self.dmiudP + X*self.dmiudX
	
	def get_K(self, P, T, X):
		return self.K_0 + (T-self.T_0)*self.dKdT + (P-self.P_0)*self.dKdP + X*self.dKdX	

	def get_rho(self, P, T, X):
		alpha = self.get_alpha(T)
		rho_0X = self.get_rho_0X(X)
		K = self.get_K(P=P, T=T, X=X)
		return rho_0X * (1 - alpha*(T-self.T_0) + (P-self.P_0)/K )
