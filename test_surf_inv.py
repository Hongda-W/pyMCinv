import ocean_surf_dbase

dset = ocean_surf_dbase.invhdf5('test_surf_dbase.h5')
dset.read_raytomo_dbase('/work3/wang/code_bkup/AgeJdF/ray_tomo_All_ph_ani.h5', runid="qc_run_0", dtype='ph', un_ph=None, create_header=True)
dset.read_raytomo_dbase('/work3/wang/code_bkup/AgeJdF/ray_tomo_All_gr_ani.h5', runid='qc_run_0', dtype='gr', un_ph=None, create_header=False, ungrfactor=2)
#dset.read_sediment_thickness(sed_file="/work3/wang/code_bkup/ToolKit/Models/SedThick/sedthick_world_v2.xyz")
#dset.read_age_JdF(age_file="/work3/wang/code_bkup/ToolKit/Models/Age_Ocean_Crust/JdF/age_JdF_model.xyz")
#dset.read_etopo(infname='/work3/wang/code_bkup/ToolKit/Models/ETOPO1/ETOPO1_Ice_g_gmt4.grd')
#dset.mc_inv_iso()
