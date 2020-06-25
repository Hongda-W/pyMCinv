# -*- coding: utf-8 -*-
"""
Module for results and postprocessing of MC inversion

:Copyright:
    Author: Lili Feng
    Graduate Research Assistant
    CIEI, Department of Physics, University of Colorado Boulder
    email: lili.feng@colorado.edu
"""
import vmodel, modparam, data, vprofile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib
import sys
sys.path.append('/work3/wang/code_bkup/pyMCinv/temp_2_vel')
import Mantle_temp_vel


def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    # s = str(100. * y)
    s = '%g'%(100. *y)
    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'
    
class postvpr(object):
    """
    An object for post data processing of 1D velocity profile inversion
    =====================================================================================================================
    ::: parameters :::
    : --- arrays --- :
    invdata         - data arrays storing inversion results
    disppre_ph/gr   - predicted phase/group dispersion
    rfpre           - object storing 1D model
    ind_acc         - index array indicating accepted models
    ind_rej         - index array indicating rejected models
    ind_thresh      - index array indicating models that pass the misfit criterion
    misfit          - misfit array
    : --- values --- :
    numbrun         - number of total runs
    numbacc         - number of accepted models
    numbrej         - number of rejected models
    npara           - number of parameters for inversion
    min_misfit      - minimum misfit value
    ind_min         - index of the minimum misfit
    thresh          - threshhold value for selecting  the finalized model (misfit < min_misfit + thresh)
    : --- object --- :
    data            - data object storing obsevred data
    avg_model       - average model object
    min_model       - minimum misfit model object
    init_model      - inital model object
    real_model      - real model object, used for synthetic test only
    temp_model      - temporary model object, used for analysis of the full assemble of the finally accepted models
    vprfwrd         - vprofile1d object for forward modelling of the average model
    =====================================================================================================================
    """
    def __init__(self, factor=1, thresh=0.2, age=9999., waterdepth=-1., vpwater=1.475):
        self.age        = age
        self.data       = data.data1d()
        self.thresh     = thresh
        self.factor     = factor
        self.avg_model  = vmodel.model1d()
        self.min_model  = vmodel.model1d()
        self.init_model = vmodel.model1d()
        self.real_model = vmodel.model1d()
        self.temp_model = vmodel.model1d()
        self.vprfwrd    = vprofile.vprofile1d()
        self.waterdepth = waterdepth
        self.vpwater    = vpwater
        # self.numbp      = np.array([1, 2, 4, 2]) # thermal parameterization
        # self.mtype      = np.array([5, 4, 2, 6]) # thermal parameterization
        self.numbp      = np.array([1, 2, 4, 5]) # seismic parameterization
        self.mtype      = np.array([5, 4, 2, 2])
        self.vpvs       = np.array([0, 2., 1.76, 1.76])
        self.maxdepth   = 50.
        self.code       = ''
        return
    
    # def read_inv_data(self, infname, verbose=True):
    #     """
    #     read inversion results from an input compressed npz file
    #     """
    #     inarr           = np.load(infname)
    #     self.invdata    = inarr['arr_0']
    #     self.disppre_ph = inarr['arr_1']
    #     self.disppre_gr = inarr['arr_2']
    #     self.rfpre      = inarr['arr_3']
    #     #
    #     self.numbrun    = self.invdata.shape[0]
    #     self.npara      = self.invdata.shape[1] - 9
    #     self.ind_acc    = self.invdata[:, 0] == 1.
    #     self.ind_rej    = self.invdata[:, 0] == -1.
    #     self.misfit     = self.invdata[:, self.npara+3]
    #     self.min_misfit = self.misfit[self.ind_acc + self.ind_rej].min()
    #     self.ind_min    = np.where(self.misfit == self.min_misfit)[0][0]
    #     self.ind_thresh = np.where(self.ind_acc*(self.misfit<= self.min_misfit+ self.thresh))[0]
    #     self.numbacc    = np.where(self.ind_acc)[0].size
    #     self.numbrej    = np.where(self.ind_rej)[0].size
    #     if verbose:
    #         print('Number of runs = '+str(self.numbrun))
    #         print('Number of accepted models = '+str(self.numbacc))
    #         print('Number of rejected models = '+str(self.numbrej))
    #         print('Number of invalid models = '+str(self.numbrun - self.numbacc - self.numbrej))
    #         print('minimum misfit = '+str(self.min_misfit))
    #     return
    
     
    def read_inv_data(self, infname, verbose=True, thresh_misfit=None, Nmax=None, Nmin=None):
        """
        read inversion results from an input compressed npz file
        """
        inarr           = np.load(infname)
        self.invdata    = inarr['arr_0']
        self.disppre_ph = inarr['arr_1']
        self.disppre_gr = inarr['arr_2']
        self.rfpre      = inarr['arr_3']
        # 
        self.numbrun    = self.invdata.shape[0]
        self.npara      = self.invdata.shape[1] - 9
        self.ind_acc    = self.invdata[:, 0] == 1.
        self.ind_rej    = self.invdata[:, 0] == -1.
        self.misfit     = self.invdata[:, self.npara+3]
        self.min_misfit = self.misfit[self.ind_acc + self.ind_rej].min()
        self.ind_min    = np.where(self.misfit == self.min_misfit)[0][0]
        self.get_thresh_model(thresh_misfit=thresh_misfit, Nmax=Nmax, Nmin=Nmin)
        self.numbacc    = np.where(self.ind_acc)[0].size
        self.numbrej    = np.where(self.ind_rej)[0].size
        if verbose:
            print('Number of runs = '+str(self.numbrun))
            print('Number of accepted models = '+str(self.numbacc))
            print('Number of rejected models = '+str(self.numbrej))
            print( 'Number of invalid models = '+str(self.numbrun - self.numbacc - self.numbrej))
            print('Number of finally accepted models = '+str(self.ind_thresh.size))
            print('minimum misfit = '+str(self.min_misfit))
        return
    
    def get_vmodel(self):
        """
        get the minimum misfit and average model from the inversion data array
        """
        min_paraval         = self.invdata[self.ind_min, 2:(self.npara+2)]
        if self.waterdepth <= 0.:
            self.min_model.get_para_model(paraval=min_paraval, nmod=self.numbp.size, \
                numbp=self.numbp, mtype = self.mtype, vpvs = self.vpvs, maxdepth=self.maxdepth)
        else:
            self.min_model.get_para_model(paraval=min_paraval, waterdepth=self.waterdepth, vpwater=self.vpwater, age=self.age, nmod=self.numbp.size, \
                numbp=self.numbp, mtype = self.mtype, vpvs = self.vpvs, maxdepth=self.maxdepth)
        self.min_model.isomod.mod2para()
        self.min_model.get_grid_mod()
        avg_paraval         = (self.invdata[self.ind_thresh, 2:(self.npara+2)]).mean(axis=0)
        if self.waterdepth <= 0.:
            self.avg_model.get_para_model(paraval=avg_paraval, nmod=self.numbp.size, \
                numbp=self.numbp, mtype = self.mtype, vpvs = self.vpvs, maxdepth=self.maxdepth)
        else:
            self.avg_model.get_para_model(paraval=avg_paraval, waterdepth=self.waterdepth, vpwater=self.vpwater, age=self.age, nmod=self.numbp.size, \
                numbp=self.numbp, mtype = self.mtype, vpvs = self.vpvs, maxdepth=self.maxdepth)
        self.vprfwrd.model  = self.avg_model
        self.avg_model.isomod.mod2para()
        self.avg_model.get_grid_mod()
        return
        
    def read_data(self, infname):
        """
        read observed data from an input npz file
        """
        inarr           = np.load(infname)
        index           = inarr['arr_0']
        if index[0] == 1 and index[1] == 0 and index[2] == 0:
            indata      = np.append(inarr['arr_1'], inarr['arr_2'])
            indata      = np.append(indata, inarr['arr_3'])
            indata      = indata.reshape(3, -1)
            self.data.dispR.get_disp(indata=indata, dtype='ph')
        if index[0] == 0 and index[1] == 1 and index[2] == 0:
            indata      = np.append(inarr['arr_1'], inarr['arr_2'])
            indata      = np.append(indata, inarr['arr_3'])
            indata      = indata.reshape(3, -1)
            self.data.dispR.get_disp(indata=indata, dtype='gr')
        if index[0] == 1 and index[1] == 1 and index[2] == 1:
            indata      = np.append(inarr['arr_1'], inarr['arr_2'])
            indata      = np.append(indata, inarr['arr_3'])
            indata      = indata.reshape(3, -1)
            self.data.dispR.get_disp(indata=indata, dtype='ph')
            indata      = np.append(inarr['arr_4'], inarr['arr_5'])
            indata      = np.append(indata, inarr['arr_6'])
            indata      = indata.reshape(3, -1)
            self.data.dispR.get_disp(indata=indata, dtype='gr')
            indata      = np.append(inarr['arr_7'], inarr['arr_8'])
            indata      = np.append(indata, inarr['arr_9'])
            indata      = indata.reshape(3, -1)
            self.data.rfr.get_rf(indata=indata)
        if index[0] == 1 and index[1] == 1 and index[2] == 0:
            indata      = np.append(inarr['arr_1'], inarr['arr_2'])
            indata      = np.append(indata, inarr['arr_3'])
            indata      = indata.reshape(3, -1)
            self.data.dispR.get_disp(indata=indata, dtype='ph')
            indata      = np.append(inarr['arr_4'], inarr['arr_5'])
            indata      = np.append(indata, inarr['arr_6'])
            indata      = indata.reshape(3, -1)
            self.data.dispR.get_disp(indata=indata, dtype='gr')
        if index[0] == 0 and index[1] == 1 and index[2] == 1:
            indata      = np.append(inarr['arr_1'], inarr['arr_2'])
            indata      = np.append(indata, inarr['arr_3'])
            indata      = indata.reshape(3, -1)
            self.data.dispR.get_disp(indata=indata, dtype='gr')
            indata      = np.append(inarr['arr_4'], inarr['arr_5'])
            indata      = np.append(indata, inarr['arr_6'])
            indata      = indata.reshape(3, -1)
            self.data.rfr.get_rf(indata=indata)
        if index[0] == 1 and index[1] == 0 and index[2] == 1:
            indata      = np.append(inarr['arr_1'], inarr['arr_2'])
            indata      = np.append(indata, inarr['arr_3'])
            indata      = indata.reshape(3, -1)
            self.data.dispR.get_disp(indata=indata, dtype='ph')
            indata      = np.append(inarr['arr_4'], inarr['arr_5'])
            indata      = np.append(indata, inarr['arr_6'])
            indata      = indata.reshape(3, -1)
            self.data.rfr.get_rf(indata=indata)
        return
    
    def get_thresh_model(self, thresh_misfit=None, Nmax=None, Nmin=None):
        """
        get the index for the finalized accepted model
        ===================================================================================
        ::: input :::
        factor  - factor to determine the threshhold value for selectingthe finalized model
        thresh  - threshhold value for selecting the finalized model
                misfit < min_misfit*factor + thresh
        ===================================================================================
        """
        if thresh_misfit is None:
            thresh_val  = self.min_misfit*self.factor+ self.thresh
        else:
            thresh_val  = thresh_misfit
        ind_thresh = self.ind_acc*(self.misfit<= thresh_val)
        if Nmax is not None:
            Nacc                = np.where(self.ind_acc)[0].size
            if Nmax > Nacc:
                print("WARNING: Nmax is reset from "+str(Nmax)+" to "+str(Nacc))
                Nmax            = Nacc
            temp_ind            = np.where(ind_thresh)[0]
            while (temp_ind.size > Nmax):
                thresh_val      -= 0.01
                ind_thresh      = self.ind_acc*(self.misfit<= thresh_val)
                temp_ind        = np.where(ind_thresh)[0]
        if Nmin is not None:
            temp_ind            = np.where(ind_thresh)[0]
            while (temp_ind.size < Nmin):
                thresh_val      += 0.01
                ind_thresh      = self.ind_acc*(self.misfit<= thresh_val)
                temp_ind        = np.where(ind_thresh)[0]
        self.thresh_val         = thresh_val
        ind_thresh_temp         = ind_thresh.copy()
        self.ind_thresh = np.where(ind_thresh)[0]
        return
    
    def get_period(self):
        """
        get period array for forward modelling
        """
        if self.data.dispR.npper>0:
            self.vprfwrd.TRpiso = self.data.dispR.pper.copy()
        else:
            self.vprfwrd.TRpiso = np.array([])
        if self.data.dispR.ngper>0:
            self.vprfwrd.TRgiso = self.data.dispR.gper.copy()
        else:
            self.vprfwrd.TRgiso = np.array([])
        if self.data.dispL.npper>0:
            self.vprfwrd.TLpiso = self.data.dispL.pper.copy()
        else:
            self.vprfwrd.TLpiso = np.array([])
        if self.data.dispL.ngper>0:
            self.vprfwrd.TLgiso = self.data.dispL.gper.copy()
        else:
            self.vprfwrd.TLgiso = np.array([])
        self.vprfwrd.TRiso = np.unique(np.append(self.vprfwrd.TRgiso, self.vprfwrd.TRpiso))
        self.vprfwrd.TLiso = np.unique(np.append(self.vprfwrd.TLgiso, self.vprfwrd.TLpiso))
        return
    
    def run_avg_fwrd(self):
        """
        run and store receiver functions and surface wave dispersion for the average model
        """
        self.get_period()
        self.get_vmodel()
        self.vprfwrd.npts   = self.rfpre.shape[1]
        self.vprfwrd.update_mod(mtype = 'iso')
        self.vprfwrd.get_vmodel(mtype = 'iso')
        self.vprfwrd.compute_fsurf()
        self.vprfwrd.compute_rftheo()
        return
    
    def plot_rf(self, title='Receiver function', obsrf=True, minrf=True, avgrf=True, assemrf=True, showfig=True):
        """
        plot receiver functions
        ==============================================================================================
        ::: input :::
        title   - title for the figure
        obsrf   - plot observed receiver function or not
        minrf   - plot minimum misfit receiver function or not
        avgrf   - plot the receiver function corresponding to the average of accepted models or not 
        assemrf - plot the receiver functions corresponding to the assemble of accepted models or not 
        ==============================================================================================
        """
        plt.figure()
        ax  = plt.subplot()
        if assemrf:
            for i in self.ind_thresh:
                rf_temp = self.rfpre[i, :]
                plt.plot(self.data.rfr.to, rf_temp, '-',color='grey',  alpha=0.01, lw=3)
        if obsrf:
            plt.errorbar(self.data.rfr.to, self.data.rfr.rfo, yerr=self.data.rfr.stdrfo, color='b', label='observed')
        if minrf:
            rf_min      = self.rfpre[self.ind_min, :]
            plt.plot(self.data.rfr.to, rf_min, 'y--', lw=3, label='min model')
        if avgrf:
            self.vprfwrd.npts   = self.rfpre.shape[1]
            self.run_avg_fwrd()
            plt.plot(self.data.rfr.to, self.vprfwrd.data.rfr.rfp, 'r--', lw=3, label='avg model')
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        plt.xlabel('time (sec)', fontsize=30)
        plt.ylabel('amplitude', fontsize=30)
        plt.title(title, fontsize=30)
        plt.legend(loc=0, fontsize=20)
        if showfig:
            plt.show()
        return
    
    def plot_disp(self, title='Dispersion curves', obsdisp=True, mindisp=True, avgdisp=True, assemdisp=True,\
                  disptype='ph', alpha=0.05, showfig=True):
        """
        plot phase/group dispersion curves
        =================================================================================================
        ::: input :::
        title       - title for the figure
        obsdisp     - plot observed disersion curve or not
        mindisp     - plot minimum misfit dispersion curve or not
        avgdisp     - plot the dispersion curve corresponding to the average of accepted models or not 
        assemdisp   - plot the dispersion curves corresponding to the assemble of accepted models or not
        =================================================================================================
        """
        plt.figure()
        ax  = plt.subplot()
        if assemdisp:
            for i in self.ind_thresh:
                if disptype == 'ph':
                    disp_temp   = self.disppre_ph[i, :]
                    plt.plot(self.data.dispR.pper, disp_temp, '-',color='grey',  alpha=alpha, lw=1)
                elif disptype == 'gr':
                    disp_temp   = self.disppre_gr[i, :]
                    plt.plot(self.data.dispR.gper, disp_temp, '-',color='grey',  alpha=alpha, lw=1)
                else:
                    disp_temp   = self.disppre_gr[i, :]
                    plt.plot(self.data.dispR.gper, disp_temp, '-',color='grey',  alpha=alpha, lw=1)
                    disp_temp   = self.disppre_ph[i, :]
                    plt.plot(self.data.dispR.pper, disp_temp, '-',color='grey',  alpha=alpha, lw=1)
        if obsdisp:
            if disptype == 'ph':
                plt.errorbar(self.data.dispR.pper, self.data.dispR.pvelo, yerr=self.data.dispR.stdpvelo,color='b', lw=1, label='observed')
            elif disptype == 'gr':
                plt.errorbar(self.data.dispR.gper, self.data.dispR.gvelo, yerr=self.data.dispR.stdgvelo,color='b', lw=2, label='observed')
            else:
                plt.errorbar(self.data.dispR.pper, self.data.dispR.pvelo, yerr=self.data.dispR.stdpvelo,color='b', lw=1, label='observed phase')
                plt.errorbar(self.data.dispR.gper, self.data.dispR.gvelo, yerr=self.data.dispR.stdgvelo,color='k', lw=1, label='observed group')
        if mindisp:
            if disptype == 'ph':
                disp_min    = self.disppre_ph[self.ind_min, :]
                plt.plot(self.data.dispR.pper, disp_min, 'r-', lw=1, ms=10, label='min model')
            elif disptype == 'gr':
                disp_min    = self.disppre_gr[self.ind_min, :]
                plt.plot(self.data.dispR.gper, disp_min, 'r-', lw=1, ms=10, label='min model')
            else:
                disp_min    = self.disppre_ph[self.ind_min, :]
                plt.plot(self.data.dispR.pper, disp_min, 'r-', lw=1, ms=10, label='min model phase')
                disp_min    = self.disppre_gr[self.ind_min, :]
                plt.plot(self.data.dispR.gper, disp_min, 'm-', lw=1, ms=10, label='min model group')
        if avgdisp:
            self.run_avg_fwrd()
            if disptype == 'ph':
                disp_avg    = self.vprfwrd.data.dispR.pvelp
                plt.plot(self.data.dispR.pper, disp_avg, 'y-', lw=1, ms=10, label='avg model')
            elif disptype == 'gr':
                disp_avg    = self.vprfwrd.data.dispR.gvelp
                plt.plot(self.data.dispR.gper, disp_avg, 'r-', lw=2, ms=10, label='avg model')
                print(self.data.dispR.gper)
                print(disp_avg)
            else:
                disp_avg    = self.vprfwrd.data.dispR.pvelp
                plt.plot(self.data.dispR.pper, disp_avg, 'y-', lw=1, ms=10, label='avg model phase')
                disp_avg    = self.vprfwrd.data.dispR.gvelp
                plt.plot(self.data.dispR.gper, disp_avg, 'g-', lw=1, ms=10, label='avg model group')
        ax.minorticks_on()
        from matplotlib.ticker import AutoMinorLocator
        minorLocator = AutoMinorLocator(5)
        # ax.xaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.grid(b=True,which='major', color='orange',alpha=0.5,linestyle='-',linewidth=0.5)
        ax.grid(b=True,which='minor', color='orange',alpha=0.5,linestyle='--',linewidth=0.5)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        plt.xlabel('Period (sec)', fontsize=20)
        label_type  = {'ph': 'Phase', 'gr': 'Group'}
        if disptype == 'ph' or disptype == 'gr':
            plt.ylabel(label_type[disptype]+' velocity (km/s)', fontsize=20)
        else:
            plt.ylabel('Velocity (km/s)', fontsize=20)
        plt.title(title+' '+self.code, fontsize=20)
        # plt.legend(loc=0, fontsize=20)
        if showfig:
            plt.show()
        return
    
    def plot_profile(self, title='Vs profile', minvpr=True, avgvpr=True, assemvpr=True, realvpr=False, showfig=True):
        """
        plot vs profiles
        =================================================================================================
        ::: input :::
        title       - title for the figure
        minvpr      - plot minimum misfit vs profile or not
        avgvpr      - plot the the average of accepted models or not 
        assemvpr    - plot the assemble of accepted models or not
        realvpr     - plot the real models or not, used for synthetic test only
        =================================================================================================
        """
        plt.figure()
        ax  = plt.subplot()
        if assemvpr:
            for i in self.ind_thresh:
                paraval     = self.invdata[i, 2:(self.npara+2)]
                if self.waterdepth <= 0.:
                    self.temp_model.get_para_model(paraval=paraval, nmod=self.numbp.size, \
                numbp=self.numbp, mtype = self.mtype, vpvs = self.vpvs, maxdepth=self.maxdepth)
                else:
                    self.temp_model.get_para_model(paraval=paraval, waterdepth=self.waterdepth, vpwater=self.vpwater, nmod=self.numbp.size, \
                numbp=self.numbp, mtype = self.mtype, vpvs = self.vpvs, maxdepth=self.maxdepth)
                plt.plot(self.temp_model.VsvArr, self.temp_model.zArr, '-',color='grey',  alpha=0.01, lw=3)               
        if minvpr:
            plt.plot(self.min_model.VsvArr, self.min_model.zArr, 'r-', lw=3, label='min model')
        if avgvpr:
            plt.plot(self.avg_model.VsvArr, self.avg_model.zArr, 'r-', lw=3, label='avg model')
        if realvpr:
            plt.plot(self.real_model.VsvArr, self.real_model.zArr, 'g-', lw=3, label='real model')
        if minvpr and avgvpr:
            plt.legend(loc=0, fontsize=12)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.minorticks_on()
        from matplotlib.ticker import AutoMinorLocator
        minorLocator = AutoMinorLocator(5)
        # ax.xaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.grid(b=True,which='major', color='orange',alpha=0.5,linestyle='-',linewidth=0.5)
        ax.grid(b=True,which='minor', color='orange',alpha=0.5,linestyle='--',linewidth=0.5)
        ax.set_ylim(ymin=0., ymax=50)
        ax.set_xlim(xmin=0.)
        ax.set_xlabel('Vs (km/s)', fontsize=16)
        ax.set_ylabel('Depth (km)', fontsize=16)
        plt.title(title, fontsize=16)
        plt.gca().invert_yaxis()
        if showfig:
            np.save('2020outVs.npy',self.avg_model.VsvArr)
            np.save('2020outZs.npy',self.avg_model.zArr)
            plt.show()
        return
    
    def plot_temp(self, title='Temperature profile', minvpr=True, avgvpr=True, assemvpr=True, realvpr=False, showfig=True):
        """
        plot temperature profiles
        =================================================================================================
        ::: input :::
        title       - title for the figure
        minvpr      - plot minimum misfit vs profile or not
        avgvpr      - plot the the average of accepted models or not 
        assemvpr    - plot the assemble of accepted models or not
        realvpr     - plot the real models or not, used for synthetic test only
        =================================================================================================
        """
        plt.figure()
        ax  = plt.subplot()
        if assemvpr:
            for i in self.ind_thresh:
                paraval     = self.invdata[i, 2:(self.npara+2)]
                if self.waterdepth <= 0.:
                    self.temp_model.get_para_model(paraval=paraval, nmod=self.numbp.size, \
                                                   numbp=self.numbp, mtype = self.mtype, vpvs = self.vpvs, maxdepth=self.maxdepth)
                else:
                    self.temp_model.get_para_model(paraval=paraval, waterdepth=self.waterdepth, vpwater=self.vpwater, nmod=self.numbp.size, \
                numbp=self.numbp, mtype = self.mtype, vpvs = self.vpvs, maxdepth=self.maxdepth)
                temp_model_T = self.temp_model.isomod.mant_temps
                # temp_model_T = self.temp_model.isomod.vs_noQ
                # temp_model_T = self.temp_model.isomod.Qs
                temp_model_deps = self.temp_model.isomod.mant_deps
                # P_model_temp = self.temp_model.isomod.mant_press
                plt.plot(temp_model_T, temp_model_deps, '-',color='grey',  alpha=0.01, lw=3)               
        if minvpr:
            min_model_T = self.min_model.isomod.mant_temps
            # min_model_T = self.min_model.isomod.vs_noQ
            # min_model_T = self.min_model.isomod.Qs
            min_model_deps = self.min_model.isomod.mant_deps
            plt.plot(min_model_T, min_model_deps, 'r-', lw=3, label='min model')
        if avgvpr:
            avg_model_T = self.avg_model.isomod.mant_temps
            # avg_model_T = self.avg_model.isomod.vs_noQ
            # avg_model_T = self.avg_model.isomod.Qs
            avg_model_deps = self.avg_model.isomod.mant_deps
            plt.plot(avg_model_T, avg_model_deps, 'y-', lw=3, label='avg model')
        if realvpr:
            real_model_T = self.real_model.isomod.mant_temps
            # real_model_T = self.real_model.isomod.vs_noQ
            # real_model_T = self.real_model.isomod.Qs
            real_model_deps = self.real_model.isomod.mant_deps
            plt.plot(real_model_T, real_model_deps, 'g-', lw=3, label='real model')
        plt.legend(loc=0, fontsize=12)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.minorticks_on()
        from matplotlib.ticker import AutoMinorLocator
        minorLocator = AutoMinorLocator(5)
        plt.gca().invert_yaxis()
        # ax.xaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.grid(b=True,which='major', color='orange',alpha=0.5,linestyle='-',linewidth=0.5)
        ax.grid(b=True,which='minor', color='orange',alpha=0.5,linestyle='--',linewidth=0.5)
        # plt.xlabel('Temperature (C)', fontsize=16)
        ax.set_xlabel('Temperature (Celsius)', fontsize=16)
        ax.set_ylabel('Depth (km)', fontsize=16)
        ax.set_ylim(ymin=0., ymax=100)
        plt.title(title, fontsize=16)
        plt.gca().invert_yaxis()
        if showfig:        
            # plt.xlim([2.5, 4.])
            # plt.xlabel('Velocity(km/s)', fontsize=30)
            plt.show()
        return
    
    def plot_hist(self, pindex=0, bins=50, title='', xlabel='', priori=False, p_name=None, minline=True, avgline=True, showfig=True):
        """
        Plot a histogram of one specified model parameter
        =================================================================================================
        ::: input :::
        pindex  - parameter index in the paraval array
        bins    - integer or sequence or ‘auto’, optional
                    If an integer is given, bins + 1 bin edges are calculated and returned,
                        consistent with numpy.histogram().
                    If bins is a sequence, gives bin edges, including left edge of first bin and
                        right edge of last bin. In this case, bins is returned unmodified.
        title   - title for the figure
        xlabel  - x axis label for the figure
        priori  - boolean, show the priori distribution or not
        =================================================================================================
        """
        plt.figure()
        ax      = plt.subplot()
        if pindex == -1:
            paraval = (self.invdata[self.ind_thresh, 2:(self.npara+2)])[:, pindex] + (self.invdata[self.ind_thresh, 2:(self.npara+2)])[:, -2]
        else:
            paraval = (self.invdata[self.ind_thresh, 2:(self.npara+2)])[:, pindex]
        if priori:
            p_inarr = np.load(p_name)
            p_invdata = p_inarr["arr_0"]
            if pindex == -1:
                paraval_pri = (p_invdata[:, 2:(self.npara+2)])[:, pindex] + (p_invdata[:, 2:(self.npara+2)])[:, -2]
            else:
                paraval_pri = (p_invdata[:, 2:(self.npara+2)])[:, pindex]
            weights_pri = np.ones_like(paraval_pri)/float(paraval_pri.size)
            _,bins,_ = plt.hist(paraval_pri, bins=bins, weights=weights_pri, alpha=0.5, color='white',edgecolor='k')
        weights = np.ones_like(paraval)/float(paraval.size)
        # weights_pri = np.ones_like(paraval_pri)/float(paraval_pri.size)
        # plt.hist(paraval_pri, bins=bins, weights=weights_pri, alpha=0.5, color='white',edgecolor='k')
        plt.hist(paraval, bins=bins, weights=weights, alpha=0.5, color='blue')
        formatter = FuncFormatter(to_percent)
        # Set the formatter
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.xlabel(xlabel, fontsize=18)
        plt.ylabel('Percentage', fontsize=18)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        plt.title(title, fontsize=20)
        min_paraval     = self.invdata[self.ind_min, 2:(self.npara+2)]
        avg_paraval     = (self.invdata[self.ind_thresh, 2:(self.npara+2)]).mean(axis=0)
        if pindex == -1:
            if minline:
                plt.axvline(x=min_paraval[pindex]+min_paraval[-2], c='r', linestyle='-.', label='min misfit value')
            if avgline:
                plt.axvline(x=avg_paraval[pindex]+avg_paraval[-2], c='y', label='average value')
        else:
            if minline:
                plt.axvline(x=min_paraval[pindex], c='r', linestyle='-.', label='min misfit value')
            if avgline:
                plt.axvline(x=avg_paraval[pindex], c='r', lw=3, label='average value')
        if minline and avgline:
            plt.legend(loc=0, fontsize=15)
        if showfig:
            plt.show()
        return
    
    def plot_hist_two_group(self, x1min, x1max, x2min, x2max, ind_s, ind_p, bins1=50, bins2=50,  title='', xlabel='', showfig=True):
        """
        Plot a histogram of one specified model parameter
        =================================================================================================
        ::: input :::
        pindex  - parameter index in the paraval array
        bins    - integer or sequence or ‘auto’, optional
                    If an integer is given, bins + 1 bin edges are calculated and returned,
                        consistent with numpy.histogram().
                    If bins is a sequence, gives bin edges, including left edge of first bin and
                        right edge of last bin. In this case, bins is returned unmodified.
        title   - title for the figure
        xlabel  - x axis label for the figure
        =================================================================================================
        """
        ax      = plt.subplot()
        paraval0= (self.invdata[self.ind_thresh, 2:(self.npara+2)])[:, ind_p]
        index1  = np.where((self.invdata[self.ind_thresh, 2:(self.npara+2)][:, ind_s] >= x1min)\
                    * (self.invdata[self.ind_thresh, 2:(self.npara+2)][:, ind_s] <= x1max))[0]
        paraval1= (self.invdata[self.ind_thresh, 2:(self.npara+2)])[index1, ind_p]
        weights1= np.ones_like(paraval1)/float(paraval0.size)
        index2  = np.where((self.invdata[self.ind_thresh, 2:(self.npara+2)][:, ind_s] >= x2min)\
                    * (self.invdata[self.ind_thresh, 2:(self.npara+2)][:, ind_s] <= x2max))[0]
        paraval2= (self.invdata[self.ind_thresh, 2:(self.npara+2)])[index2, ind_p]
        weights2= np.ones_like(paraval2)/float(paraval0.size)
        plt.hist(paraval1, bins=bins1, weights=weights1, alpha=0.5, color='r')
        plt.hist(paraval2, bins=bins2, weights=weights2, alpha=0.5, color='b')
        formatter = FuncFormatter(to_percent)
        # Set the formatter
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.xlabel(xlabel, fontsize=30)
        plt.ylabel('Percentage', fontsize=30)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        plt.title(title, fontsize=35)
        min_paraval     = self.invdata[self.ind_min, 2:(self.npara+2)]
        avg_paraval     = (self.invdata[self.ind_thresh, 2:(self.npara+2)]).mean(axis=0)
        plt.axvline(x=min_paraval[pindex], c='k', linestyle='-.', label='min misfit value')
        plt.axvline(x=avg_paraval[pindex], c='b', label='average value')
        plt.legend(loc=0, fontsize=15)
        if showfig:
            plt.show()
        return
    
    def plot_hist_three_group(self, x1min, x1max, x2min, x2max, x3min, x3max, ind_s, ind_p, bins1=50, bins2=50, bins3=50, title='', xlabel='', showfig=True):
        """
        Plot a histogram of one specified model parameter
        =================================================================================================
        ::: input :::
        pindex  - parameter index in the paraval array
        bins    - integer or sequence or ‘auto’, optional
                    If an integer is given, bins + 1 bin edges are calculated and returned,
                        consistent with numpy.histogram().
                    If bins is a sequence, gives bin edges, including left edge of first bin and
                        right edge of last bin. In this case, bins is returned unmodified.
        title   - title for the figure
        xlabel  - x axis label for the figure
        =================================================================================================
        """
        ax      = plt.subplot()
        paraval0= (self.invdata[self.ind_thresh, 2:(self.npara+2)])[:, ind_p]
        index1  = np.where((self.invdata[self.ind_thresh, 2:(self.npara+2)][:, ind_s] >= x1min)\
                    * (self.invdata[self.ind_thresh, 2:(self.npara+2)][:, ind_s] <= x1max))[0]
        paraval1= (self.invdata[self.ind_thresh, 2:(self.npara+2)])[index1, ind_p]
        weights1= np.ones_like(paraval1)/float(paraval0.size)
        index2  = np.where((self.invdata[self.ind_thresh, 2:(self.npara+2)][:, ind_s] >= x2min)\
                    * (self.invdata[self.ind_thresh, 2:(self.npara+2)][:, ind_s] <= x2max))[0]
        paraval2= (self.invdata[self.ind_thresh, 2:(self.npara+2)])[index2, ind_p]
        weights2= np.ones_like(paraval2)/float(paraval0.size)
        index3  = np.where((self.invdata[self.ind_thresh, 2:(self.npara+2)][:, ind_s] >= x3min)\
                    * (self.invdata[self.ind_thresh, 2:(self.npara+2)][:, ind_s] <= x3max))[0]
        paraval3= (self.invdata[self.ind_thresh, 2:(self.npara+2)])[index3, ind_p]
        weights3= np.ones_like(paraval3)/float(paraval0.size)
        plt.hist(paraval1, bins=bins1, weights=weights1, alpha=0.5, color='r')
        plt.hist(paraval2, bins=bins2, weights=weights2, alpha=0.5, color='b')
        plt.hist(paraval3, bins=bins3, weights=weights3, alpha=0.5, color='g')
        formatter = FuncFormatter(to_percent)
        # Set the formatter
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.xlabel(xlabel, fontsize=30)
        plt.ylabel('Percentage', fontsize=30)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        plt.title(title, fontsize=35)
        min_paraval     = self.invdata[self.ind_min, 2:(self.npara+2)]
        # avg_paraval     = (self.invdata[self.ind_thresh, 2:(self.npara+2)]).mean(axis=0)
        plt.axvline(x=min_paraval[ind_p], c='k', linestyle='-.', label='min misfit value')
        # plt.axvline(x=avg_paraval[pindex], c='b', label='average value')
        plt.legend(loc=0, fontsize=15)
        if showfig:
            plt.show()
        return
    
    def plot_hist_age(self,bins=50,minvpr=True,avgvpr=True,assemvpr=True,realvpr=False,showfig=True):
        """ Plot cooling age distribution
        """
        plt.figure()
        ax1  = plt.subplot()
        plt.figure()
        ax2  = plt.subplot()
        c_age = np.array([])
        pot_T = np.array([])
        if assemvpr:
            for i in self.ind_thresh:
                paraval     = self.invdata[i, 2:(self.npara+2)]
                if self.waterdepth <= 0.:
                    self.temp_model.get_para_model(paraval=paraval, nmod=self.numbp.size, \
                                                   numbp=self.numbp, mtype = self.mtype, vpvs = self.vpvs, maxdepth=self.maxdepth)
                else:
                    self.temp_model.get_para_model(paraval=paraval, waterdepth=self.waterdepth, vpwater=self.vpwater, nmod=self.numbp.size, \
                numbp=self.numbp, mtype = self.mtype, vpvs = self.vpvs, maxdepth=self.maxdepth)
                c_age = np.append(c_age,self.temp_model.isomod.cvel[0,-1])
                pot_T = np.append(pot_T,self.temp_model.isomod.cvel[1,-1])
        if minvpr:
            c_age_min = self.min_model.isomod.cvel[0,-1]
            pot_T_min = self.min_model.isomod.cvel[1,-1]
            ax1.axvline(x=c_age_min, c='r', linestyle='-.', label='min misfit value')
            ax2.axvline(x=pot_T_min, c='r', linestyle='-.', label='min misfit value')
        if avgvpr:
            c_age_avg = self.avg_model.isomod.cvel[0,-1]
            pot_T_avg = self.avg_model.isomod.cvel[1,-1]
            ax1.axvline(x=c_age_avg, c='y', label='average value')
            ax2.axvline(x=pot_T_avg, c='y', label='average value')
        if realvpr:
            c_age_real = self.real_model.isomod.cvel[0,-1]
            pot_T_real = self.real_model.isomod.cvel[1,-1]
        weights1 = np.ones_like(c_age)/float(c_age.size)
        weights2 = np.ones_like(pot_T)/float(pot_T.size)
        ax1.hist(c_age,bins=bins,alpha=0.5, color='b', weights=weights1,density=False)
        ax2.hist(pot_T,bins=bins,alpha=0.5, color='b', weights=weights2,density=True)
        ax1.legend(loc=0, fontsize=10)
        ax2.legend(loc=0, fontsize=10)
        ax1.set_title("Cooling age distribution")
        ax1.set_xlabel("Cooling age (Ma)")
        ax2.set_title("Potential temperature distribution")
        ax2.set_xlabel("Temperature (Celsius)")
        if showfig:
            plt.show()
        return
    
    
    
    
    
    
