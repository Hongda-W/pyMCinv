# -*- coding: utf-8 -*-
"""
Module for handling parameterization of the model

:Copyright:
    Author: Lili Feng
    Graduate Research Assistant
    CIEI, Department of Physics, University of Colorado Boulder
    email: lili.feng@colorado.edu
"""
import numpy as np
import numba
import math
import random
from scipy.optimize import lsq_linear
import sys
from scipy.signal import argrelmax, argrelmin
from scipy.interpolate import interp1d

sys.path.append('/work3/wang/code_bkup/pyMCinv/temp_2_vel')
import Mantle_temp_vel


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
                                        0   - velocity coefficient for splines
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
        self.npara = 0
        self.maxind = 6
        self.isspace = False
        return

    def init_arr(self, npara):
        """
        initialize the arrays
        """
        self.npara = npara
        self.paraval = np.zeros(self.npara, dtype=np.float64)
        self.paraindex = np.zeros((self.maxind, self.npara), dtype=np.float64)
        self.space = np.zeros((3, self.npara), dtype=np.float64)
        return

    def readparatxt(self, infname):
        """
        read txt perturbation parameter file
        ==========================================================================
        ::: input :::
        infname - input file name
        ==========================================================================
        """
        npara = 0
        i = 0
        for l1 in open(infname, "r"):
            npara += 1
        print("Number of parameters for perturbation: %d " % npara)
        self.init_arr(npara)
        with open(infname, 'r') as fid:
            for line in fid.readlines():
                temp = np.array(line.split(), dtype=np.float64)
                ne = temp.size
                for j in range(ne):
                    self.paraindex[j, i] = temp[j]
                i += 1
        # print "read para over!"
        return

    def write_paraval_txt(self, outfname):
        np.savetxt(outfname, self.paraval, fmt='%g')
        return

    def read_paraval_txt(self, infname):
        self.paraval = np.loadtxt(infname, dtype=np.float64)
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
                tval = random.random()
                self.paraval[i] = tval * (self.space[1, i] - self.space[0, i]) + self.space[0, i]
        elif ptype == 1:
            for i in range(self.npara):
                oldval = self.paraval[i]
                step = self.space[2, i]
                run = True
                j = 0
                while (run and j < 10000):
                    newval = random.gauss(oldval, step)
                    if (newval >= self.space[0, i] and newval <= self.space[1, i]):
                        run = False
                    j += 1
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
    # --------------------------------
    # defining the knot vector
    # --------------------------------
    m = nBs - 1 + degBs
    t = np.zeros(m + 1, dtype=np.float64)
    for i in range(degBs):
        t[i] = zmin_Bs + i * (zmax_Bs - zmin_Bs) / 10000.
    for i in range(degBs, m + 1 - degBs):
        n_temp = m + 1 - degBs - degBs + 1
        if disfacBs != 1:
            temp = (zmax_Bs - zmin_Bs) * (disfacBs - 1) / (math.pow(disfacBs, n_temp) - 1)
        else:
            temp = (zmax_Bs - zmin_Bs) / n_temp
        t[i] = temp * math.pow(disfacBs, (i - degBs)) + zmin_Bs
    for i in range(m + 1 - degBs, m + 1):
        t[i] = zmax_Bs - (zmax_Bs - zmin_Bs) / 10000. * (m - i)
    # depth array
    step = (zmax_Bs - zmin_Bs) / (npts - 1)
    depth = np.zeros(npts, dtype=np.float64)
    for i in range(npts):
        depth[i] = np.float64(i) * np.float64(step) + np.float64(zmin_Bs)
    # arrays for storing B spline basis
    obasis = np.zeros((np.int64(m), np.int64(npts)), dtype=np.float64)
    nbasis = np.zeros((np.int64(m), np.int64(npts)), dtype=np.float64)
    # --------------------------------
    # computing B spline basis
    # --------------------------------
    for i in range(m):
        for j in range(npts):
            if (depth[j] >= t[i] and depth[j] < t[i + 1]):
                obasis[i][j] = 1
            else:
                obasis[i][j] = 0
    for pp in range(1, degBs):
        for i in range(m - pp):
            for j in range(npts):
                nbasis[i][j] = (depth[j] - t[i]) / (t[i + pp] - t[i]) * obasis[i][j] + \
                               (t[i + pp + 1] - depth[j]) / (t[i + pp + 1] - t[i + 1]) * obasis[i + 1][j]
        for i in range(m - pp):
            for j in range(npts):
                obasis[i][j] = nbasis[i][j]
    nbasis[0][0] = 1
    nbasis[nBs - 1][npts - 1] = 1
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
        self.nmod = 0
        self.maxlay = 100
        self.maxspl = 20
        self.para = para1d()
        return

    def init_arr(self, nmod, age):
        """
        initialization of arrays
        """
        self.nmod = nmod
        self.age = age
        # arrays of size nmod
        self.numbp = np.zeros(self.nmod, dtype=np.int64)
        self.mtype = np.zeros(self.nmod, dtype=np.int64)
        self.thickness = np.zeros(self.nmod, dtype=np.float64)
        self.nlay = np.ones(self.nmod, dtype=np.int64) * 20
        self.vpvs = np.ones(self.nmod, dtype=np.float64) * 1.75
        self.isspl = np.zeros(self.nmod, dtype=np.int64)
        # arrays of size maxspl, nmod
        self.cvel = np.zeros((self.maxspl, self.nmod), dtype=np.float64)
        # arrays of size maxlay, nmod
        self.ratio = np.zeros((self.maxlay, self.nmod), dtype=np.float64)
        self.vs = np.zeros((self.maxlay, self.nmod), dtype=np.float64)
        self.hArr = np.zeros((self.maxlay, self.nmod), dtype=np.float64)
        self.zArr = np.zeros((self.maxlay, self.nmod), dtype=np.float64)
        # arrays of size maxspl, maxlay, nmod
        self.spl = np.zeros((self.maxspl, self.maxlay, self.nmod), dtype=np.float64)
        return

    def readmodtxt(self, infname):
        """
        Read model parameterization from a txt file
        first line: oceanic age, continental location if > 180
        column 1: id
        column 2: flag  - layer(1)/B-splines(2/3)/gradient layer(4)/water(5)
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
        nmod = 0
        for l1 in open(infname, "r"):
            nmod += 1
        print("Number of model parameter groups: %d " % (nmod - 1))
        # step 1
        for num, l1 in enumerate(open(infname, "r")):
            if num == 0:  # the first line gives age
                self.init_arr(nmod - 1, float(l1.split('#', 1)[0].rstrip()))
                continue
            l1 = l1.rstrip()
            l2 = l1.split()
            iid = int(l2[0])
            flag = int(l2[1])
            thickness = float(l2[2])
            tnp = int(l2[3])  # number of parameters
            # step 2
            self.mtype[iid] = flag
            self.thickness[iid] = thickness
            self.numbp[iid] = tnp
            if (int(l2[1]) == 5):  # water layer
                if (tnp != 1):
                    print('Water layer! Only one value for Vp')
                    return False
            if (int(l2[1]) == 4):
                if (tnp != 2):
                    print('Error: only two values needed for gradient type, and one value for vpvs')
                    print(tnp)
                    return False
            if ((int(l2[1]) == 1 and len(l2) != 4 + 2 * tnp + 1) or (int(l2[1]) == 2 and len(
                    l2) != 4 + tnp + 1)):  # tnp parameters (+ tnp ratio for layered model) + 1 vpvs parameter
                print(l2)
                print('wrong input !!!')
                return False
            nr = 0
            # step 3
            for i in range(tnp):
                self.cvel[i, iid] = float(l2[4 + i])
                if (int(l2[1]) == 1):  # type 1 layer
                    self.ratio[nr, iid] = float(l2[4 + tnp + i])
                    nr += 1
            # step 4
            self.vpvs[iid] = (float(l2[-1])) - 0.
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
            self.nlay[i] = 60
        elif self.thickness[i] < 10:
            self.nlay[i] = 5
        elif self.thickness[i] < 20:
            self.nlay[i] = 10
        else:
            self.nlay[i] = 50
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
        nBs = self.numbp[i]
        if nBs < 4:
            degBs = 3
        else:
            degBs = 4
        zmin_Bs = 0.
        zmax_Bs = self.thickness[i]
        disfacBs = 2.
        npts = self.nlay[i]
        nbasis = bspl_basis(nBs, degBs, zmin_Bs, zmax_Bs, disfacBs, npts)
        m = nBs - 1 + degBs
        if m > self.maxspl:
            raise ValueError('number of splines is too large, change default maxspl!')
        self.spl[:nBs, :npts, i] = nbasis[:nBs, :]
        self.isspl[i] = True
        return True

    def update(self):
        """
        Update model (vs and hArr arrays), from the thickness, cvel
        """
        for i in range(self.nmod):
            if self.nlay[i] > self.maxlay:
                print('number of layers is too large, need change default maxlay!')
                return False
            # layered model
            """
            if self.mtype[i] == 1:
                self.nlay[i]                = self.numbp[i]
                #self.nlay[i]                = 5
                if self.nlay[i] == 1: # only one layer:
                    self.ratio[0, i] = 1.
                self.hArr[:self.nlay[i], i] = self.ratio[:self.nlay[i], i] * self.thickness[i]
                #self.hArr[:self.nlay[i], i] = self.thickness[i]/5
                self.vs[:self.nlay[i], i]   = self.cvel[:self.nlay[i], i]
            """
            if self.mtype[i] == 1:
                # extra 7 layers for the top 700m of oceanic sediment
                self.nlay[i] = self.numbp[i]+7
                self.hArr[:7, i] = 0.1
                self.hArr[7:self.nlay[i], i] = self.ratio[:self.numbp[i], i]-7 * self.thickness[i]
                self.vs[7:self.nlay[i], i] = self.cvel[:self.numbp[i], i]
            # B spline model
            elif self.mtype[i] == 2:
                self.isspl[i] = False
                self.bspline(i)
                # # if self.isspl[i] != 1:
                # #     self.bspline(i)
                self.vs[:self.nlay[i], i] = np.dot((self.spl[:self.numbp[i], :self.nlay[i], i]).T,
                                                   self.cvel[:self.numbp[i], i])
                self.hArr[:self.nlay[i], i] = self.thickness[i] / self.nlay[i]
                # # for ilay in range(self.nlay[i]):
                # #     tvalue 	= 0.
                # #     for ibs in xrange(self.numbp[i]):
                # #         tvalue = tvalue + self.spl[ibs, ilay, i] * self.cvel[ibs, i]
                # #     self.vs[ilay, i]    = tvalue
                # #     self.hArr[ilay, i]  = self.thickness[i]/self.nlay[i]
            # gradient layer
            elif self.mtype[i] == 4:
                nlay = 20
                if self.thickness[i] >= 20.:
                    nlay = 30
                if self.thickness[i] > 10. and self.thickness[i] < 20.:
                    nlay = int(self.thickness[i] / .3)
                if self.thickness[i] > 2. and self.thickness[i] <= 10.:
                    nlay = int(self.thickness[i] / 0.2)
                if self.thickness[i] < 0.5:
                    nlay = 5
                dh = self.thickness[i] / float(nlay)
                dcvel = (self.cvel[1, i] - self.cvel[0, i]) / (nlay - 1.)
                self.vs[:nlay, i] = self.cvel[0, i] + dcvel * np.arange(nlay, dtype=np.float64)
                self.hArr[:nlay, i] = dh
                self.nlay[i] = nlay
            # water layer
            elif self.mtype[i] == 5:
                nlay = 1
                self.vs[0, i] = 0.
                self.hArr[0, i] = self.thickness[i]
                self.nlay[i] = 1
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
                self.nlay[i] = self.numbp[i]+7
                self.hArr[:self.nlay[i], i] = self.ratio[:self.nlay[i], i] * self.thickness[i]
            # B spline model
            elif self.mtype[i] == 2:
                self.isspl[i] = False
                self.bspline(i)
                self.hArr[:self.nlay[i], i] = self.thickness[i] / self.nlay[i]
            # gradient layer
            elif self.mtype[i] == 4:
                nlay = 4
                if self.thickness[i] >= 20.:
                    nlay = 20
                if self.thickness[i] > 10. and self.thickness[i] < 20.:
                    nlay = int(self.thickness[i] / 1.)
                if self.thickness[i] > 2. and self.thickness[i] <= 10.:
                    nlay = int(self.thickness[i] / 0.5)
                if self.thickness[i] < 0.5:
                    nlay = 3
                dh = self.thickness[i] / float(nlay)
                self.hArr[:nlay, i] = dh
                self.nlay[i] = nlay
            # water layer
            elif self.mtype[i] == 5:
                nlay = 1
                self.hArr[0, i] = self.thickness[i]
                self.nlay[i] = 1
        for i_mod in range(self.nmod):
            # depth below Earth's surface
            self.zArr[:self.nlay[i_mod], i_mod] = self.hArr[:self.nlay[i_mod], i_mod].cumsum() + self.hArr[:,
                                                                                                 :i_mod].flatten().sum()
        return True

    def parameterize_input(self, zarr, vsarr, mohodepth, seddepth, maxdepth=50.):
        """
        paramerization of a refernce input model
        ===============================================================================
        ::: input :::
        zarr, vsarr - input depth/vs array, must be the same size (unit - km, km/s)
        mohodepth   - input moho depth (unit - km)
        seddepth    - input sediment depth (unit - km)
        maxdepth    - maximum depth for the 1-D profile (default - 200 km)
        ::: output :::
        self.thickness
        self.numbp      - [2, 4, 5]
        self.mtype      - [4, 2, 2]
        self.vpvs       - [2., 1.75, 1.75]
        self.spl
        self.cvel       - determined from input vs profile
        ===============================================================================
        """
        if zarr.size != vsarr.size:
            raise ValueError('Inconsistent input 1-D profile depth and vs arrays!')
        self.init_arr(3)
        self.thickness[:] = np.array([seddepth, mohodepth - seddepth, maxdepth - mohodepth])
        self.numbp[:] = np.array([2, 4, 5])
        self.mtype[:] = np.array([4, 2, 2])
        self.vpvs[:] = np.array([2., 1.75, 1.75])  # vpvs for the ocean sediment is way off.
        self.update_depth()
        hArr = np.append(self.hArr[:self.nlay[0], 0], self.hArr[:self.nlay[1], 1])
        hArr = np.append(hArr, self.hArr[:self.nlay[2], 2])
        zinterp = hArr.cumsum()
        ind_max = np.where(zarr >= maxdepth)[0][0]
        zarr = zarr[:(ind_max + 1)]
        vsarr = vsarr[:(ind_max + 1)]
        if vsarr[0] > vsarr[1]:
            vs_temp = vsarr[0]
            vsarr[0] = vsarr[1]
            vsarr[1] = vs_temp
        ind_crust = np.where(zarr >= mohodepth)[0][0]
        vs_crust = vsarr[:(ind_crust + 1)]
        if not np.all(vs_crust[1:] >= vs_crust[:-1]):
            print('WARNING: sort the input vs array to make it monotonically increases with depth in the crust')
            vs_crust = np.sort(vs_crust)
            vsarr = np.append(vs_crust, vsarr[(ind_crust + 1):])
        # interpolation
        vsinterp = np.interp(zinterp, zarr, vsarr)
        # ------------------------------------
        # determine self.cvel arrays
        # ------------------------------------
        # sediments
        self.cvel[0, 0] = vsinterp[0]
        self.cvel[1, 0] = vsinterp[self.nlay[0] - 1]
        # # # crust
        # # self.cvel[0, 1]     = vsinterp[self.nlay[0]]
        # # spl                 = (self.spl[:self.numbp[1], :self.nlay[1], 1]).T
        # # ind_max2            = spl[:, 1].argmax()
        # # ind_max3            = spl[:, 2].argmax()
        # # self.cvel[3, 1]     = vsinterp[self.nlay[0]+self.nlay[1] - 1]
        # # self.cvel[1, 1]     = self.cvel[0, 1] + (self.cvel[3, 1] - self.cvel[0, 1])*ind_max2/self.nlay[1]
        # # self.cvel[2, 1]     = self.cvel[0, 1] + (self.cvel[3, 1] - self.cvel[0, 1])*ind_max3/self.nlay[1]
        # # # mantle
        # # self.cvel[0, 2]     = vsinterp[self.nlay[0]+self.nlay[1]]
        # # self.cvel[4, 2]     = vsinterp[-1]
        # ---------------------------------
        # inversion with lsq_linear
        # ---------------------------------
        # crust
        A = (self.spl[:self.numbp[1], :self.nlay[1], 1]).T
        b = vsinterp[self.nlay[0]:(self.nlay[0] + self.nlay[1])]
        vs0 = max(vsinterp[self.nlay[0]], 3.0)
        vs1 = min(vsinterp[self.nlay[0] + self.nlay[1] - 1], 4.2)
        x = lsq_linear(A, b, bounds=(vs0, vs1)).x
        self.cvel[:4, 1] = x[:]
        # mantle
        A = (self.spl[:self.numbp[2], :self.nlay[2], 2]).T
        b = vsinterp[(self.nlay[0] + self.nlay[1]):]
        vs0 = max(vsinterp[(self.nlay[0] + self.nlay[1]):].min(), 4.0)
        vs1 = min(vsinterp[(self.nlay[0] + self.nlay[1]):].max(), vsarr.max())
        x = lsq_linear(A, b, bounds=(vs0, vs1)).x
        self.cvel[:5, 2] = x[:]
        # # # inversion with numpy
        # # # crust
        # # A                   = (self.spl[:self.numbp[1], :self.nlay[1], 1]).T
        # # b                   = vsinterp[self.nlay[0]:(self.nlay[0]+self.nlay[1])]
        # # x                   = np.linalg.lstsq(A, b)[0]
        # # self.cvel[:4, 1]    = x[:]
        # # # mantle
        # # A                   = (self.spl[:self.numbp[2], :self.nlay[2], 2]).T
        # # b                   = vsinterp[(self.nlay[0]+self.nlay[1]):]
        # # x                   = np.linalg.lstsq(A, b)[0]
        # # self.cvel[:5, 2]    = x[:]
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
            # self.get_paraind_ocean()
            self.get_paraind_ocean2()
            return
        numbp_sum = self.numbp.sum()
        npara = numbp_sum + self.nmod - 1
        # water layer, added May 15, 2018
        if self.mtype[0] == 5:
            npara -= 2
        #
        self.para.init_arr(npara)
        ipara = 0
        for i in range(self.nmod):
            # water layer, added May 15, 2018
            if self.mtype[i] == 5:
                continue
            # --------------------------------
            for j in range(self.numbp[i]):
                self.para.paraindex[0, ipara] = 0
                if i == 0 or (i == 1 and self.mtype[0] == 5):  # water layer, added May 15, 2018
                    # sediment, cvel space is +- 0.3 km/s, different from Shen et al. 2012
                    self.para.paraindex[1, ipara] = 1
                    self.para.paraindex[2, ipara] = 1.
                elif (i == 1 and self.mtype[0] == 4) or (i == 2 and self.mtype[0] == 5):  # crustal layer
                    self.para.paraindex[1, ipara] = 1.
                    self.para.paraindex[2, ipara] = 0.5
                else:  # mantle layer
                    self.para.paraindex[1, ipara] = 1.
                    self.para.paraindex[2, ipara] = 0.3
                # 0.05 km/s
                self.para.paraindex[3, ipara] = self.para.paraindex[2, ipara] / 10.
                self.para.paraindex[4, ipara] = i
                self.para.paraindex[5, ipara] = j
                ipara += 1
        if self.nmod >= 3:
            # sediment thickness
            self.para.paraindex[0, ipara] = 1
            self.para.paraindex[1, ipara] = 1.
            self.para.paraindex[2, ipara] = 3.
            self.para.paraindex[3, ipara] = 0.2
            if self.mtype[0] == 5:  # if there's water layer, added May 15, 2018
                self.para.paraindex[4, ipara] = 1
            else:
                self.para.paraindex[4, ipara] = 0
            ipara += 1
        # crustal thickness
        self.para.paraindex[0, ipara] = 1
        self.para.paraindex[1, ipara] = 1
        self.para.paraindex[2, ipara] = 10.
        self.para.paraindex[3, ipara] = 1.5
        if self.nmod >= 3:
            if self.mtype[0] == 5:  # water layer, added May 15, 2018
                self.para.paraindex[4, ipara] = 2.
            else:
                self.para.paraindex[4, ipara] = 1.
        else:
            self.para.paraindex[4, ipara] = 0.
        return

    def get_paraind_ocean(self):
        """ Parameterize the priori information for oceanic settings
        """
        numbp_sum = self.numbp.sum()
        npara = numbp_sum + self.nmod - 1
        # water layer, added May 15, 2018
        if self.mtype[0] == 5:
            npara -= 2
        #
        self.para.init_arr(npara)
        ipara = 0
        for i in range(self.nmod):
            # water layer, added May 15, 2018
            if self.mtype[i] == 5:
                continue
            # --------------------------------
            for j in range(self.numbp[i]):
                self.para.paraindex[0, ipara] = 0
                if i == 0 or (i == 1 and self.mtype[0] == 5):  # water layer, added May 15, 2018
                    # sediment, cvel space is +- 2 km/s, different from Shen et al. 2012
                    self.para.paraindex[1, ipara] = 1
                    self.para.paraindex[2, ipara] = 0.5
                    self.para.paraindex[3, ipara] = 0.05
                elif (i == 1 and self.mtype[0] == 4) or (i == 2 and self.mtype[0] == 5):  # crustal layer
                    self.para.paraindex[1, ipara] = 1
                    self.para.paraindex[2, ipara] = 0.
                    self.para.paraindex[3, ipara] = 0.
                else:  # mantle layer
                    self.para.paraindex[1, ipara] = -1
                    if j == 0:
                        self.para.paraindex[2, ipara] = 10.  # first B-spline coefficient can change 10%
                    elif j == 1:
                        self.para.paraindex[2, ipara] = 10.  # second B-spline coefficient can change 10%
                    elif j == 2:
                        self.para.paraindex[2, ipara] = 5.  # third B-spline coefficient can change 5%
                    else:
                        self.para.paraindex[2, ipara] = 2.
                    # if j == self.numbp[i]-1: # smaller change at bottom of mantle
                    #     self.para.paraindex[2, ipara] = 2.
                    # else:
                    #     self.para.paraindex[2, ipara] = 4.
                # 0.05 km/s
                self.para.paraindex[3, ipara] = 0.05
                self.para.paraindex[4, ipara] = i
                self.para.paraindex[5, ipara] = j
                ipara += 1
        if self.nmod >= 3:
            # sediment thickness
            self.para.paraindex[0, ipara] = 1
            self.para.paraindex[1, ipara] = 1
            self.para.paraindex[2, ipara] = 2.
            self.para.paraindex[3, ipara] = 0.2
            if self.mtype[0] == 5:  # water layer, added May 15, 2018
                self.para.paraindex[4, ipara] = 1
            else:
                self.para.paraindex[4, ipara] = 0
            ipara += 1
        # crustal thickness
        self.para.paraindex[0, ipara] = 1
        self.para.paraindex[1, ipara] = 1
        self.para.paraindex[2, ipara] = 0.
        self.para.paraindex[3, ipara] = 0.
        if self.nmod >= 3:
            if self.mtype[0] == 5:  # water layer, added May 15, 2018
                self.para.paraindex[4, ipara] = 2.
            else:
                self.para.paraindex[4, ipara] = 1.
        else:
            self.para.paraindex[4, ipara] = 0.
        return

    def get_paraind_ocean2(self):
        """ Parameterize the priori information for oceanic settings
        Try something new, allowing crust to change.
        """
        numbp_sum = self.numbp.sum()
        npara = numbp_sum + self.nmod - 1
        # water layer, added May 15, 2018
        if self.mtype[0] == 5:
            npara -= 1
        #
        self.para.init_arr(npara)
        ipara = 0
        for i in range(self.nmod):
            # water layer, added May 15, 2018
            if self.mtype[i] == 5:
                continue
            # --------------------------------
            for j in range(self.numbp[i]):
                self.para.paraindex[0, ipara] = 0
                if i == 0 or (i == 1 and self.mtype[0] == 5):  # water layer, added May 15, 2018
                    # sediment, cvel space is +- 0.5 km/s, different from Shen et al. 2012
                    self.para.paraindex[1, ipara] = 1
                    self.para.paraindex[2, ipara] = 1.
                    self.para.paraindex[3, ipara] = 0.1
                elif (i == 1 and self.mtype[0] == 4) or (i == 2 and self.mtype[0] == 5):  # crustal layer
                    self.para.paraindex[1, ipara] = 1
                    if j == 1:  # second parameter for the crust
                        self.para.paraindex[2, ipara] = 0.5
                    else:
                        self.para.paraindex[2, ipara] = 0.5
                    self.para.paraindex[3, ipara] = self.para.paraindex[2, ipara] / 10
                else:  # mantle layer
                    self.para.paraindex[1, ipara] = 1
                    self.para.paraindex[2, ipara] = 0.2
                    # if j == self.numbp[i]-1: # smaller change at bottom of mantle
                    #     self.para.paraindex[2, ipara] = 2.
                    # else:
                    #     self.para.paraindex[2, ipara] = 4.
                    self.para.paraindex[3, ipara] = 0.02
                self.para.paraindex[4, ipara] = i
                self.para.paraindex[5, ipara] = j
                ipara += 1
        if self.mtype[0] == 5:
            # water thickness
            self.para.paraindex[0, ipara] = 1
            self.para.paraindex[1, ipara] = 1
            self.para.paraindex[2, ipara] = 1.
            self.para.paraindex[3, ipara] = 0.1
            self.para.paraindex[4, ipara] = 0
            ipara += 1
        if self.nmod >= 3:
            # sediment thickness
            self.para.paraindex[0, ipara] = 1
            self.para.paraindex[1, ipara] = 1
            self.para.paraindex[2, ipara] = 3.
            self.para.paraindex[3, ipara] = 0.1
            if self.mtype[0] == 5:  # water layer, added May 15, 2018
                self.para.paraindex[4, ipara] = 1
            else:
                self.para.paraindex[4, ipara] = 0
            ipara += 1
        # crustal thickness
        self.para.paraindex[0, ipara] = 1
        self.para.paraindex[1, ipara] = 1
        self.para.paraindex[2, ipara] = 5.
        self.para.paraindex[3, ipara] = 0.5
        if self.nmod >= 3:
            if self.mtype[0] == 5:  # water layer, added May 15, 2018
                self.para.paraindex[4, ipara] = 2.
            else:
                self.para.paraindex[4, ipara] = 1.
        else:
            self.para.paraindex[4, ipara] = 0.
        return

    def mod2para(self, verbose=False):
        """
        convert model to parameter arrays for perturbation
        """
        for i in range(self.para.npara):
            ig = int(self.para.paraindex[4, i])
            # velocity coefficient
            if int(self.para.paraindex[0, i]) == 0:
                ip = int(self.para.paraindex[5, i])
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
            # -------------------------------------------
            # defining parameter space for perturbation
            # -------------------------------------------
            if not self.para.isspace:
                step = self.para.paraindex[3, i]
                if int(self.para.paraindex[1, i]) == 1:
                    valmin = val - self.para.paraindex[2, i]
                    valmax = val + self.para.paraindex[2, i]
                else:
                    valmin = val - val * self.para.paraindex[2, i] / 100.
                    valmax = val + val * self.para.paraindex[2, i] / 100.
                ###
                # if self.para.paraindex[0, i] == 1 and i == 12:
                #     valmin  = 0.
                #     valmax  = 5.
                ###
                valmin = max(0., valmin)
                valmax = max(valmin + 0.0001, valmax)
                if verbose:
                    print("{}, {}, {}, {}, {}".format(i, self.para.paraindex[4, i], self.para.paraindex[0, i], valmin,
                                                      valmax))
                # if (int(self.para.paraindex[0, i]) == 0 and i == 0 \
                #     and int(self.para.paraindex[5, i]) == 0): # if it is the upper sedi:
                #     valmin              = max (0.2, valmin)
                #     valmax              = max (0.5, valmax)
                self.para.space[0, i] = valmin
                self.para.space[1, i] = valmax
                self.para.space[2, i] = step
        self.para.isspace = True
        return

    def para2mod(self):
        """
        Convert paratemers (for perturbation) to model parameters
        """
        for i in range(self.para.npara):
            val = self.para.paraval[i]
            ig = int(self.para.paraindex[4, i])
            # velocity coeficient for splines
            if int(self.para.paraindex[0, i]) == 0:
                ip = int(self.para.paraindex[5, i])
                self.cvel[ip][ig] = val
            # total thickness of the group
            elif int(self.para.paraindex[0, i]) == 1:
                self.thickness[ig] = val
            # vp/vs ratio
            elif int(self.para.paraindex[0, i]) == -1:
                self.vpvs[ig] = val
            else:
                print('Unexpected value in paraindex!')
        self.thickness[-1] = 100. - self.thickness[:-1].sum()
        return

    # def isgood_cvel(self):
    #     """ check if the coefficients make physical sense
    #     """
    #     if self.age < 180: # negative velocity gradient below moho for ocean (gn0 & gn1 should both be 3)
    #         mant_cvels = self.cvel[:self.numbp[-1],-1]
    #         if mant_cvels[0] - mant_cvels[1] < 0.05: # negative velocity gradient below moho for ocean
    #             return False
    #         try:
    #             ind = np.where((mant_cvels[1:] - mant_cvels[:-1])>0.02)[0][0]
    #             if (mant_cvels[ind+2:] - mant_cvels[ind+2] < -0.05).any():
    #                 return False
    #         except:
    #             # print("1.pass. {}".format(mant_cvels))
    #             return True
    #     return True

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
        # velocity contrast, constraint (5) in 4.2 of Shen et al., 2012
        for i in range(self.nmod - 1):
            if self.vs[0, i + 1] < self.vs[self.nlay[i] - 1, i]:
                return False
        return True
        if m1 >= self.nmod:
            m1 = self.nmod - 1
        if m0 < 0:
            m0 = 0
        if g1 >= self.nmod:
            g1 = self.nmod - 1
        if g0 < 0:
            g0 = 0
        # monotonic change
        # velocity constrast, contraint (3) and (4) in 4.2 of Shen et al., 2012
        if m0 <= m1:
            for j in range(m0, m1 + 1):
                vs0 = self.vs[:self.nlay[j] - 1, j]
                vs1 = self.vs[1:self.nlay[j], j]
                if np.any(np.greater(vs0, vs1)):
                    return False
        # gradient check
        if g0 <= g1:
            for j in range(g0, g1 + 1):
                if self.vs[0, j] > self.vs[1, j]:
                    return False
        # if self.age < 180:
        #     crust_cvels = self.cvel[:self.numbp[-2],-2] # B spline coefficients for the crust
        #     if not np.any(np.diff(crust_cvels) < 0 ): # velocity in the crust must increase with depth
        #         return False
        #     mant_cvels = self.cvel[:self.numbp[-1],-1]
        #     if self.vs[self.nlay[self.nmod-1]-1,self.nmod-1] > 4.4 or self.vs[self.nlay[self.nmod-1]-1,self.nmod-1] < 4.2: # bottom of mantle, 100km
        #         return False
        #     if mant_cvels[0] < mant_cvels[1]: # negative velocity gradient below moho for ocean
        #         return False
        # vs_mantle       = self.vs[:self.nlay[self.nmod-1], self.nmod-1]
        # local_indmax    = argrelmax(vs_mantle)[0]
        # local_indmin    = argrelmin(vs_mantle)[0]
        # if local_indmin.size > 0 and local_indmax.size > 0:
        #     if local_indmin.size == local_indmax.size:
        #         vmin    = vs_mantle[local_indmin]
        #         vmax    = vs_mantle[local_indmax]
        #     else:
        #         Ndiff   = local_indmax.size - local_indmin.size
        #         if Ndiff > 0:
        #             vmin    = vs_mantle[local_indmin]
        #             vmax    = vs_mantle[local_indmax[:-Ndiff]]
        #         else:
        #             vmin    = vs_mantle[local_indmin[:Ndiff]]
        #             vmax    = vs_mantle[local_indmax]
        #     if (vmax-vmin).max() > 0.1*vs_mantle.mean():
        #         return False
        # if (vs_mantle.max() - vs_mantle.min()) > 0.15*vs_mantle.mean():
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
        nlay = self.nlay.sum()
        hArr = np.array([], dtype=np.float64)
        vs = np.array([], dtype=np.float64)
        vp = np.array([], dtype=np.float64)
        rho = np.array([], dtype=np.float64)
        qs = np.array([], dtype=np.float64)
        qp = np.array([], dtype=np.float64)
        depth = np.array([], dtype=np.float64)
        for i in range(self.nmod):
            hArr = np.append(hArr, self.hArr[:self.nlay[i], i])
            depth = np.append(depth, (self.hArr[:self.nlay[i], i]).cumsum())
            if self.mtype[i] == 5:
                vs = np.append(vs, 0.)
                vp = np.append(vp, self.cvel[0][i])
                rho = np.append(rho, 1.02)
                qs = np.append(qs, 10000.)
                qp = np.append(qp, 57822.)
            elif (i == 0 and self.mtype[i] != 5) or (i == 1 and self.mtype[0] == 5):
                vs = np.append(vs, self.vs[:self.nlay[i], i])
                vp = np.append(vp, self.vs[:self.nlay[i], i] * self.vpvs[i])
                rho = np.append(rho, 0.541 + 0.3601 * self.vs[:self.nlay[i], i] * self.vpvs[i])
                qs = np.append(qs, 80. * np.ones(self.nlay[i], dtype=np.float64))
                qp = np.append(qp, 160. * np.ones(self.nlay[i], dtype=np.float64))
            else:
                vs = np.append(vs, self.vs[:self.nlay[i], i])
                vp = np.append(vp, self.vs[:self.nlay[i], i] * self.vpvs[i])
                rho = np.append(rho, 0.541 + 0.3601 * self.vs[:self.nlay[i], i] * self.vpvs[i])
                qs = np.append(qs, 600. * np.ones(self.nlay[i], dtype=np.float64))
                qp = np.append(qp, 1400. * np.ones(self.nlay[i], dtype=np.float64))
        rho[vp > 7.5] = 3.35
        return hArr, vs, vp, rho, qs, qp, nlay

    def get_vmodel_lili(self):
        """ lili's old version...
        get velocity models, slightly faster than get_vmodel_old
        ==========================================================================
        ::: output :::
        hArr, vs, vp, rho, qs, qp
        ==========================================================================
        """
        nlay = self.nlay.sum()
        hArr = np.zeros(nlay, dtype=np.float64)
        vs = np.zeros(nlay, dtype=np.float64)
        vp = np.zeros(nlay, dtype=np.float64)
        rho = np.zeros(nlay, dtype=np.float64)
        qs = np.zeros(nlay, dtype=np.float64)
        qp = np.zeros(nlay, dtype=np.float64)
        depth = np.zeros(nlay, dtype=np.float64)
        for i in range(self.nmod):
            if i == 0:
                hArr[:self.nlay[0]] = self.hArr[:self.nlay[0], 0]
            elif i < self.nmod - 1:
                hArr[self.nlay[:i].sum():self.nlay[:i + 1].sum()] = self.hArr[:self.nlay[i], i]
            else:
                hArr[self.nlay[:i].sum():] = self.hArr[:self.nlay[i], i]
            if self.mtype[i] == 5 and i == 0:  # water layer
                vs[0] = 0.
                vp[0] = self.cvel[0][i]
                rho[0] = 1.02
                qs[0] = 10000.
                qp[0] = 57822.
            elif (i == 0 and self.mtype[i] != 5):  # sediment layer
                vs[:self.nlay[0]] = self.vs[:self.nlay[i], i]
                vp[:self.nlay[0]] = self.vs[:self.nlay[i], i] * self.vpvs[i]
                rho[:self.nlay[0]] = 0.541 + 0.3601 * self.vs[:self.nlay[i], i] * self.vpvs[i]
                qs[:self.nlay[0]] = 80. * np.ones(self.nlay[i], dtype=np.float64)
                qp[:self.nlay[0]] = 160. * np.ones(self.nlay[i], dtype=np.float64)
            elif (i == 1 and self.mtype[0] == 5) and self.nmod > 2:  # sediment layer
                vs[self.nlay[:i].sum():self.nlay[:i + 1].sum()] = self.vs[:self.nlay[i], i]
                vp[self.nlay[:i].sum():self.nlay[:i + 1].sum()] = self.vs[:self.nlay[i], i] * self.vpvs[i]
                rho[self.nlay[:i].sum():self.nlay[:i + 1].sum()] = 0.541 + 0.3601 * self.vs[:self.nlay[i], i] * \
                                                                   self.vpvs[i]
                qs[self.nlay[:i].sum():self.nlay[:i + 1].sum()] = 80. * np.ones(self.nlay[i], dtype=np.float64)
                qp[self.nlay[:i].sum():self.nlay[:i + 1].sum()] = 160. * np.ones(self.nlay[i], dtype=np.float64)
            elif (i == 1 and self.mtype[0] == 5) and self.nmod == 2:  # crust layer
                vs[self.nlay[:i].sum():] = self.vs[:self.nlay[i], i]
                vp[self.nlay[:i].sum():] = self.vs[:self.nlay[i], i] * self.vpvs[i]
                rho[self.nlay[:i].sum():] = 0.541 + 0.3601 * self.vs[:self.nlay[i], i] * self.vpvs[i]
                qs[self.nlay[:i].sum():] = 80. * np.ones(self.nlay[i], dtype=np.float64)
                qp[self.nlay[:i].sum():] = 160. * np.ones(self.nlay[i], dtype=np.float64)
            elif i < self.nmod - 1:  # crust layer
                vs[self.nlay[:i].sum():self.nlay[:i + 1].sum()] = self.vs[:self.nlay[i], i]
                vp[self.nlay[:i].sum():self.nlay[:i + 1].sum()] = self.vs[:self.nlay[i], i] * self.vpvs[i]
                rho[self.nlay[:i].sum():self.nlay[:i + 1].sum()] = 0.541 + 0.3601 * self.vs[:self.nlay[i], i] * \
                                                                   self.vpvs[i]
                qs[self.nlay[:i].sum():self.nlay[:i + 1].sum()] = 600. * np.ones(self.nlay[i], dtype=np.float64)
                qp[self.nlay[:i].sum():self.nlay[:i + 1].sum()] = 1400. * np.ones(self.nlay[i], dtype=np.float64)
            else:  # mantle layer
                vs[self.nlay[:i].sum():] = self.vs[:self.nlay[i], i]
                vp[self.nlay[:i].sum():] = self.vs[:self.nlay[i], i] * self.vpvs[i]
                rho[self.nlay[:i].sum():] = 0.541 + 0.3601 * self.vs[:self.nlay[i], i] * self.vpvs[i]
                qs[self.nlay[:i].sum():] = 600. * np.ones(self.nlay[i], dtype=np.float64)
                qp[self.nlay[:i].sum():] = 1400. * np.ones(self.nlay[i], dtype=np.float64)
        depth = hArr.cumsum()
        vs, vp = self.fixdeepmodel(hArr, vs, vp)
        rho[vp > 7.5] = 3.35
        return hArr, vs, vp, rho, qs, qp, nlay

    def get_vmodel(self):
        """
        get velocity models, modified Q values
        ==========================================================================
        ::: output :::
        hArr, vs, vp, rho, qs, qp
        ==========================================================================
        """
        if self.age > 180:
            return self.get_vmodel_lili()
        nlay = self.nlay.sum()
        hArr = np.zeros(nlay, dtype=np.float64)
        vs = np.zeros(nlay, dtype=np.float64)
        vp = np.zeros(nlay, dtype=np.float64)
        rho = np.zeros(nlay, dtype=np.float64)
        qs = np.zeros(nlay, dtype=np.float64)
        qp = np.zeros(nlay, dtype=np.float64)
        depth = np.zeros(nlay, dtype=np.float64)
        for i in range(self.nmod):
            if i == 0:
                hArr[:self.nlay[0]] = self.hArr[:self.nlay[0], 0]
            elif i < self.nmod - 1:
                hArr[self.nlay[:i].sum():self.nlay[:i + 1].sum()] = self.hArr[:self.nlay[i], i]
            else:
                hArr[self.nlay[:i].sum():] = self.hArr[:self.nlay[i], i]
            if self.mtype[i] == 5 and i == 0:  # water layer
                vs[0] = 0.
                vp[0] = self.cvel[0][i]
                rho[0] = 1.027  # density of ocean water
                qs[0] = 10000.
                qp[0] = 57823.
            elif (i == 0 and self.mtype[i] != 5):  # sediment layer
                # we don't use vpvs ratio to calculate vp for ocean sediments. !!!
                vs[:self.nlay[0]] = self.vs[:self.nlay[i], i]
                # vp[:self.nlay[0]]       = self.vs[:self.nlay[i], i]*self.vpvs[i]
                vp[:self.nlay[0]] = self.vs[:self.nlay[i],
                                    i] * 1.23 + 1.28  # cite [Vp/Vs and Poisson's ratios in marine sediments and rocks, Hamilton 1979]
                rho[:self.nlay[0]] = 0.541 + 0.3601 * vp[:self.nlay[0]]
                qs[:self.nlay[0]] = 80. * np.ones(self.nlay[i], dtype=np.float64)
                qp[:self.nlay[0]] = 1. / (4. / 3. * (self.vpvs[i]) ** (-2) / qs[:self.nlay[0]] + (
                            1 - 4. / 3. * (self.vpvs[i]) ** (-2)) / 57823.)
            elif (i == 1 and self.mtype[0] == 5) and self.nmod > 2:  # sediment layer
                vs[self.nlay[:i].sum():self.nlay[:i + 1].sum()] = self.vs[:self.nlay[i], i]
                # vp[self.nlay[:i].sum():self.nlay[:i+1].sum()]   = self.vs[:self.nlay[i], i]*self.vpvs[i]
                vp[self.nlay[:i].sum():self.nlay[:i + 1].sum()] = self.vs[:self.nlay[i],
                                                                  i] * 1.23 + 1.28  # cite [Vp/Vs and Poisson's ratios in marine sediments and rocks, Hamilton 1979]
                rho[self.nlay[:i].sum():self.nlay[:i + 1].sum()] = 0.541 + 0.3601 * vp[:self.nlay[0]]
                qs[self.nlay[:i].sum():self.nlay[:i + 1].sum()] = 80. * np.ones(self.nlay[i], dtype=np.float64)
                qp[self.nlay[:i].sum():self.nlay[:i + 1].sum()] = 1. / (
                            4. / 3. * (self.vpvs[i]) ** (-2) / qs[self.nlay[:i].sum():self.nlay[:i + 1].sum()] \
                            + (1 - 4. / 3. * (self.vpvs[i]) ** (-2)) / 57823.)
            elif (i == 1 and self.mtype[0] == 5) and self.nmod == 2:  # sediment layer
                vs[self.nlay[:i].sum():] = self.vs[:self.nlay[i], i]
                # vp[self.nlay[:i].sum():]    = self.vs[:self.nlay[i], i]*self.vpvs[i]
                vp[self.nlay[:i].sum():] = self.vs[:self.nlay[i],
                                           i] * 1.23 + 1.28  # cite [Vp/Vs and Poisson's ratios in marine sediments and rocks, Hamilton 1979]
                rho[:self.nlay[0]] = 0.541 + 0.3601 * vp[:self.nlay[0]]
                qs[self.nlay[:i].sum():] = 80. * np.ones(self.nlay[i], dtype=np.float64)
                qp[self.nlay[:i].sum():] = 1. / (4. / 3. * (self.vpvs[i]) ** (-2) / qs[self.nlay[:i].sum():] + (
                            1 - 4. / 3. * (self.vpvs[i]) ** (-2)) / 57823.)
            elif i < self.nmod - 1:  # crust layer
                vs[self.nlay[:i].sum():self.nlay[:i + 1].sum()] = self.vs[:self.nlay[i], i]
                vp[self.nlay[:i].sum():self.nlay[:i + 1].sum()] = self.vs[:self.nlay[i], i] * self.vpvs[i]
                rho[self.nlay[:i].sum():self.nlay[:i + 1].sum()] = 0.541 + 0.3601 * self.vs[:self.nlay[i], i] * \
                                                                   self.vpvs[i]
                qs[self.nlay[:i].sum():self.nlay[:i + 1].sum()] = 350. * np.ones(self.nlay[i], dtype=np.float64)
                qp[self.nlay[:i].sum():self.nlay[:i + 1].sum()] = 1. / (
                            4. / 3. * (self.vpvs[i]) ** (-2) / qs[self.nlay[:i].sum():self.nlay[:i + 1].sum()] \
                            + (1 - 4. / 3. * (self.vpvs[i]) ** (-2)) / 57823.)
            else:  # mantle layer
                vs[self.nlay[:i].sum():] = self.vs[:self.nlay[i], i]
                vp[self.nlay[:i].sum():] = self.vs[:self.nlay[i], i] * self.vpvs[i]
                rho[self.nlay[:i].sum():] = 0.541 + 0.3601 * self.vs[:self.nlay[i], i] * self.vpvs[i]
                qs[self.nlay[:i].sum():], qp[self.nlay[:i].sum():] = self.get_mantle_Q(
                    hArr.cumsum()[self.nlay[:i].sum():], self.vpvs[i])
        depth = hArr.cumsum()
        vs, vp, rho, qs, qp = self.fixtopsed(vs, vp, rho, qs, qp)
        vs, vp, rho, qs, qp = self.fixdeepmodel(hArr, vs, vp, rho, qs, qp)
        rho[vp > 7.5] = 3.35
        return hArr, vs, vp, rho, qs, qp, nlay

    @staticmethod
    def fixdeepmodel(hArr, vs, vp, rho, qs, qp, mindep=10):
        deepModel = np.loadtxt("/work3/wang/code_bkup/pyMCinv/JeffModel/178.6_-38.8_JeffersonModel.txt")
        depth = hArr.cumsum()
        ind = depth > mindep
        vs_func = interp1d(deepModel[:, 2], deepModel[:, 4], fill_value="extrapolate")
        vp_func = interp1d(deepModel[:, 2], deepModel[:, 3], fill_value="extrapolate")
        vs[ind] = vs_func(depth[ind])
        vp[ind] = vp_func(depth[ind])
        rho[ind] = 0.541 + 0.3601 * vp[ind]
        qs[ind] = 350. * np.ones(ind.sum(), dtype=np.float64)
        qp[ind] = 1. / (4. / 3. * (vp[ind] / vs[ind]) ** (-2) / qs[ind] + (1 - 4. / 3. * (vp[ind] / vs[ind]) ** (-2)) / 57823.)
        return vs, vp, rho, qs, qp

    def fixtopsed(self, vs, vp, rho, qs, qp):
        i = 1
        z = self.hArr[:self.nlay[i], i].cumsum()
        ind = z <= 0.7
        vs[self.nlay[:i].sum():self.nlay[:i + 1].sum()][ind] = 0.4 + 0.6 / 0.7 * z[ind]
        vp[self.nlay[:i].sum():self.nlay[:i + 1].sum()][ind] = 1.6 + 0.8 / 0.7 * z[ind]
        rho[self.nlay[:i].sum():self.nlay[:i + 1].sum()][ind] = 1.6 + 0.4 / 0.7 * z[ind]
        qs[self.nlay[:i].sum():self.nlay[:i + 1].sum()][ind] = 80. * np.ones(z[ind].size, dtype=np.float64)
        vpvs = vp[self.nlay[:i].sum():self.nlay[:i + 1].sum()][ind] / vs[self.nlay[:i].sum():self.nlay[:i + 1].sum()][ind]
        qp[self.nlay[:i].sum():self.nlay[:i + 1].sum()] = 1. / (
                4. / 3. * (vpvs) ** (-2) / qs[self.nlay[:i].sum():self.nlay[:i + 1].sum()]
                + (1 - 4. / 3. * (vpvs) ** (-2)) / 57823.)
        return vs, vp, rho, qs, qp

    def get_mantle_Q(self, deps, vpvs):
        """ get Q value for mantle layer, follw Eq(4) from Ye (2013)
        Calculate Q value for 20 sec period, Q doesn't change a lot with period
        """
        from scipy.special import erf
        Tm = 1315.  # deep mantle temperature in celsius
        A = 30.  # A value in Ye 2013 eq(4)
        temps = 1315. * erf(
            500 * deps / np.sqrt(self.age * 365 * 24 * 3600)) + 273.15  # depth dependent mantle temperature in Kelvin
        qs = A * (np.pi / 10) ** 0.1 * np.exp(0.1 * (2.5e5 + 3.2 * 9.8 * deps * 10) / (8.314472 * temps))
        qp = 1. / (4. / 3. * vpvs ** (-2) / qs + (1 - 4. / 3. * vpvs ** (-2)) / 57823.)
        return qs, qp

    def plot_Q(self):  # simple function for visually check if the Q model makes sense
        import matplotlib.pyplot as plt
        self.update()
        hArr, _, _, _, qs, qp, _ = self.get_vmodel()
        deps = hArr.cumsum()
        plt.plot(qp, deps)
        plt.gca().invert_yaxis()
        plt.show()
        return
