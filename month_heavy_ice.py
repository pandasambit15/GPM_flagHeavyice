import multiprocessing as mp
import glob
import h5py
import numpy as np 
import time
from matplotlib.dates import date2num
import datetime
import xarray as xr
from scipy.interpolate import interp1d

def parse_dtime(subsection):
    """
    
    parses the GPM time info to make datetimes
    
    Copyright: Randy J. Chase & The University of Illinois @ Urbana-Champaign 2018
    
    """
    year = subsection['ScanTime']['Year']
    month = subsection['ScanTime']['Month']
    day = subsection['ScanTime']['DayOfMonth']
    hour = subsection['ScanTime']['Hour']
    minute = subsection['ScanTime']['Minute']
    second = subsection['ScanTime']['Second']
    mil_second = subsection['ScanTime']['MilliSecond']
    
    dtime = np.zeros(year.shape[0],dtype=object)
    for i in np.arange(0,year.shape[0]):
        dtime[i] = datetime.datetime(year[i],month[i],day[i],hour[i],minute[i],second[i],mil_second[i])
    
    return dtime
    
def Model_to_GPM(T_model,H_model,H_GPM):
    """
    
    Interpolates Model data to GPM range bins
    
    Copyright: Randy J. Chase & The University of Illinois @ Urbana-Champaign 2018
    
    """
    
    ind = np.isnan(T_model)
    H = H_model[~ind]/1000.
    T = T_model[~ind]
    f = interp1d(H, T, kind='linear',bounds_error=False)
    T_GPM = f(H_GPM)
    
    return T_GPM 
    
def get_merra(time,heavy_dict):
        """

        Co-locates MERRA-2 data to the GPM profiles

        Copyright: Randy J. Chase & The University of Illinois @ Urbana-Champaign 2018

        """
        h = np.load('/data/gpm/a/randyjc2/GPM_datafiles/HeavyIce/h_data.npy')

        destime = time[0]
        month = destime.month
        day = destime.day

        if month < 10:
            month = '0'+ str(month)
        else:
            month = str(month)

        if day <10:
            day = '0' + str(day)
        else:
            day = str(day)

        ds_url = '/data/gpm/a/randyjc2/MERRA/NEW/'+ str(destime.year) + '/' + 'MERRA2_400.inst6_3d_ana_Np.2014' + month + day+ '.nc4'

        ###load file
        merra = xr.open_dataset(ds_url)
        ###

        merra_s = {}
        listokeys = ['T', 'U', 'V', 'QV','lev']
        for j in listokeys:
            ###loop through all data points where flag is met in the orbital
            T_GPM = np.zeros(heavy_dict['ku'].shape)
            h_gpm = np.zeros(heavy_dict['ku'].shape)
            time_array = np.array([],dtype="S10")
            for i in np.arange(0,heavy_dict['ku'].shape[0]):
                
                deslat = heavy_dict['lats'][i]
                deslon = heavy_dict['lons'][i]
                destime = time[i]
                cross_ind = heavy_dict['cross_ind'][i]

                sounding = merra.sel(lon=deslon,lat=deslat,time=destime,method='nearest')

                H = sounding['H'].values
                T = sounding[j].values
                time_merra = sounding['time'].values
                
                #get gpm z according to the h_datafile and the cross index
                T_GPM[i,:] = Model_to_GPM(T,H,h[:,cross_ind])
                h_gpm[i,:] = h[:,cross_ind]
                time_array = np.append(time_array,str(time_merra))

            heavy_dict[j] = T_GPM
            heavy_dict['H'] = h_gpm
            heavy_dict['time_merra'] = time_array
                
        del merra,T_GPM,time_array,h_gpm,h

        return heavy_dict
    
def make_norm_mats(xx,yy,month):
    x = [1,2]
    it = -1 
    for i in x:
        it = it + 1
        xedge = np.arange(-180,181,i)
        yedge = np.arange(-90,91,i)
        norm_mat = np.zeros([4,len(xedge)-1,len(yedge)-1])
        handle = np.histogram2d(xx,yy,bins=[xedge,yedge])
        
        if month == 12 or month == 1 or month == 2:
            norm_mat[0,:,:] = norm_mat[0,:,:] + handle[0]
        elif month == 3 or month ==4 or month==5:
            norm_mat[1,:,:] = norm_mat[1,:,:] + handle[0]
        elif month ==6 or month ==7 or month ==8:
            norm_mat[2,:,:] = norm_mat[2,:,:] + handle[0]
        else:
            norm_mat[3,:,:] = norm_mat[3,:,:] + handle[0]
             
        if it == 0:
            onebyone = norm_mat
        else:
            twobytwo = norm_mat
            
    return onebyone,twobytwo

def get_heavy_ice(filename):
    
    """
    
    Reads GPM V05 DPR files, retrieves GPM data (including co-located MERRA2 data) where the flagHeavyice is satisfied...
    
    flag is the sum of several different conditions:
    
    16: DPR flag == True, see Iguchi et al. 2018 
    
    Copyright: Randy J. Chase & The University of Illinois @ Urbana-Champaign 2018
    
    """
    
    #load file into memory 
    data = h5py.File(filename,"r")
    
    #try the first bit of code. If no MS data, then it will exit
    try:
        
        #2D arrays dim: along_track,cross_track
        flags = data['MS']['CSF']['flagHeavyIcePrecip'][:]
        lons = data['MS']['Longitude'][:]
        lats = data['MS']['Latitude'][:]
        lons_all = np.ravel(data['MS']['Longitude'][:])
        lats_all = np.ravel(data['MS']['Latitude'][:])
        
        

        #1D dim:along_track
        dtimes = parse_dtime(data['MS'])
        #make dtimes 2D dim:along_track,cross_track
        dtimes_2D = np.zeros(flags.shape)
        dtimes_2D_dtime = np.zeros(flags.shape,dtype=object)
        for i in np.arange(0,flags.shape[0]):
            dtimes_2D[i,:] = date2num(dtimes[i]) #convert dtimes to seconds since 1970-1-1 for easy saving 
            dtimes_2D_dtime[i,:] = dtimes[i] #datetimes to grab the right MERRA file 
            
        #get profiles of Z (only on inner swath of Ku) dim: along_track,cross_track,height
        ku_prof = data['NS']['PRE']['zFactorMeasured'][:,12:37,:]
        ka_prof = data['MS']['PRE']['zFactorMeasured'][:]

        #get clutter free bin estimate dim: along_track,cross_track 
        clu_bo_ku = data['NS']['PRE']['binClutterFreeBottom'][:]
        clu_bo_ka = data['MS']['PRE']['binClutterFreeBottom'][:]
        
        #get pathintegrated attenuation correction dim: along_track,cross_track 
        pathAtten = data['MS']['SRT']['pathAtten'][:] #in dB from Surface Reference Tech. 
        reliabFlag = data['MS']['SRT']['reliabFlag'][:] # see docs for meaning. essentially > 1 is OK

        #the flag has to be at least 16 to get the DPR to be satisfied. If > than there are other conditions met
        ind2d = np.where(flags >= 16)
        
        #if not satisfied anywhere exit
        if len(np.ravel(ind2d)) == 0:
            data_r = {}
            data_r['lons'] = np.nan
            data_r['lats'] = np.nan
            data_r['flags'] = np.nan
            data_r['time'] = np.nan
            data_r['ku'] = np.nan
            data_r['ka'] = np.nan
            data_r['clutter_ku'] = np.nan
            data_r['clutter_ka'] = np.nan 
            data_r['cross_ind'] = np.nan
            data_r['SLP'] = np.nan
            data_r['PS'] = np.nan
            data_r['H'] = np.nan
            data_r['T'] = np.nan
            data_r['U'] = np.nan
            data_r['V'] = np.nan
            data_r['QV'] = np.nan
            data_r['lev'] = np.nan
            data_r['pathAtten'] = np.nan
            data_r['reliabFlag'] = np.nan
            data_r['time_merra'] = np.nan
            onedate = dtimes[0]
            xx,yy = make_norm_mats(lons_all,lats_all,onedate.month)
            data_r['1x1'] = xx
            data_r['2x2'] = yy
            
            del data,flags,lons,lats,ku_prof,ka_prof,clu_bo_ku,clu_bo_ka,dtimes
            
            return data_r
        
        #extract 2d arrays
        lon_r = lons[ind2d]
        lat_r = lats[ind2d]
        flags_r = flags[ind2d]
        dtime_r = dtimes_2D[ind2d]
        dtime_r2 = dtimes_2D_dtime[ind2d]
        clu_bo_ku_r = clu_bo_ku[ind2d]
        clu_bo_ka_r = clu_bo_ka[ind2d] 
        pathAtten_r = pathAtten[ind2d]
        reliabFlag_r = reliabFlag[ind2d]

        #extract 3d arrays
        ku_prof_r = np.zeros([len(ind2d[0]),ku_prof.shape[2]])
        ka_prof_r = np.zeros([len(ind2d[0]),ka_prof.shape[2]])
        it = -1
        for i in np.arange(0,len(ind2d[0])):
            ku_prof_r[i,:] = ku_prof[ind2d[0][i],ind2d[1][i],:]
            ka_prof_r[i,:] = ka_prof[ind2d[0][i],ind2d[1][i],:]


        data_r = {}
        data_r['lons'] = lon_r
        data_r['lats'] = lat_r
        data_r['flags'] = flags_r 
        data_r['time'] = dtime_r
        data_r['ku'] = ku_prof_r
        data_r['ka'] = ka_prof_r
        data_r['clutter_ku'] = clu_bo_ku_r
        data_r['clutter_ka'] = clu_bo_ka_r 
        data_r['cross_ind'] = ind2d[1]
        data_r['pathAtten'] = pathAtten_r
        data_r['reliabFlag'] = reliabFlag_r
        onedate = dtimes[0]
        xx,yy = make_norm_mats(lons_all,lats_all,onedate.month)
        data_r['1x1'] = xx
        data_r['2x2'] = yy
        
        del data,flags,lons,lats,ku_prof,ka_prof,clu_bo_ku,clu_bo_ka,lon_r,lat_r,flags_r,dtimes,dtime_r,ku_prof_r, ka_prof_r, clu_bo_ku_r,clu_bo_ka_r
        
        data_r = get_merra(dtime_r2,data_r)
        
        del dtime_r2
        
 #this is in case the MS data doesnt exist
    except:
    
        data_r = {}
        data_r['lons'] = np.nan
        data_r['lats'] = np.nan
        data_r['flags'] = np.nan
        data_r['time'] = np.nan
        data_r['ku'] = np.nan
        data_r['ka'] = np.nan
        data_r['clutter_ku'] = np.nan
        data_r['clutter_ka'] = np.nan 
        data_r['cross_ind'] = np.nan
        data_r['SLP'] = np.nan
        data_r['PS'] = np.nan
        data_r['H'] = np.nan
        data_r['T'] = np.nan
        data_r['U'] = np.nan
        data_r['V'] = np.nan
        data_r['QV'] = np.nan
        data_r['lev'] = np.nan
        data_r['pathAtten'] = np.nan
        data_r['reliabFlag'] = np.nan
        data_r['time_merra'] = np.nan
        xedge = np.arange(-180,181,1)
        yedge = np.arange(-90,91,1)
        data_r['1x1'] = np.zeros([4,len(xedge)-1,len(yedge)-1])
        xedge = np.arange(-180,181,2)
        yedge = np.arange(-90,91,2)
        data_r['2x2'] = np.zeros([4,len(xedge)-1,len(yedge)-1])
        
        del data 
        
    
    return data_r

def make_hdf(d,savestr):
    
    myfile = h5py.File(savestr,'w')
    myfile.create_group('data')
    data=myfile['data']
    flag1 = 0
    it = 0
    onebyone = d[it]['1x1']
    twobytwo = d[it]['2x2']
    #need to make sure the first entry isnt a nan...
    while flag1 == 0:
        ku_mat = d[it]['ku']
        try:
            dummy = ku_mat.shape
            ka_mat = d[it]['ka']
            H_mat = d[it]['H']
            T_mat = d[it]['T']
            QV_mat = d[it]['QV']
            U_mat = d[it]['U']
            V_mat = d[it]['V']
            P_mat = d[it]['lev']
            
            #1d
            lons = d[it]['lons']
            lats = d[it]['lats']
            clut_ku = np.asarray(d[it]['clutter_ku'],dtype='i4')
            clut_ka = np.asarray(d[it]['clutter_ka'],dtype='i4')
            pathAtten = d[it]['pathAtten']
            reliabFlag = np.asarray(d[it]['reliabFlag'],dtype='i4')
            cross_ind = np.asarray(d[it]['cross_ind'],dtype='i4')
            time = d[it]['time']
            time_merra = d[it]['time_merra']
            n_prof = ku_mat.shape[it]
            
            if it > 0:
                for j in np.arange(0,4):
                    onebyone[j,:,:] = onebyone[j,:,:] + d[it]['1x1'][j,:,:]
                    twobytwo[j,:,:] = twobytwo[j,:,:] + d[it]['2x2'][j,:,:]
            
            
            flag1 = 1
            break
        except:
            it = it + 1


    for i in d[it+1:]:
        try:
            ku = i['ku']
            addme = ku.shape[0]
            n_prof = n_prof + addme 

            #2d arrays
            ku_mat = np.concatenate([ku_mat,i['ku']])
            ka_mat = np.concatenate([ka_mat,i['ka']])
            H_mat = np.concatenate([H_mat,i['H']])
            T_mat = np.concatenate([T_mat,i['T']])
            QV_mat = np.concatenate([QV_mat,i['QV']])
            U_mat = np.concatenate([U_mat,i['U']])
            V_mat = np.concatenate([V_mat,i['V']])
            P_mat = np.concatenate([P_mat,i['lev']])

            #1d arrays
            lons = np.append(lons,i['lons'])
            lats = np.append(lats,i['lats'])
            clut_ku = np.append(clut_ku, np.asarray(i['clutter_ku'],dtype='i4'))
            clut_ka = np.append(clut_ka, np.asarray(i['clutter_ka'],dtype='i4'))
            pathAtten = np.append(pathAtten,i['pathAtten'])
            reliabFlag = np.append(reliabFlag,np.asarray(i['reliabFlag'],dtype='i4'))
            cross_ind = np.append(cross_ind,np.asarray(i['cross_ind'],dtype='i4'))
            time = np.append(time,i['time'])
            time_merra = np.append(time_merra,i['time_merra'])
            
            for j in np.arange(0,4):
                onebyone[j,:,:] = onebyone[j,:,:] + i['1x1'][j,:,:]
                twobytwo[j,:,:] = twobytwo[j,:,:] + i['2x2'][j,:,:]

        except:
            for j in np.arange(0,4):
                onebyone[j,:,:] = onebyone[j,:,:] + i['1x1'][j,:,:]
                twobytwo[j,:,:] = twobytwo[j,:,:] + i['2x2'][j,:,:]


    #2d arrays        
    data.create_dataset('ku',(ku_mat.shape[0],ku_mat.shape[1]),dtype='f8',data=ku_mat)
    data.create_dataset('ka',(ku_mat.shape[0],ku_mat.shape[1]),dtype='f8',data=ka_mat)
    data.create_dataset('H',(ku_mat.shape[0],ku_mat.shape[1]),dtype='f8',data=H_mat)
    data.create_dataset('T',(ku_mat.shape[0],ku_mat.shape[1]),dtype='f8',data=T_mat)
    data.create_dataset('QV',(ku_mat.shape[0],ku_mat.shape[1]),dtype='f8',data=QV_mat)
    data.create_dataset('U',(ku_mat.shape[0],ku_mat.shape[1]),dtype='f8',data=U_mat)
    data.create_dataset('V',(ku_mat.shape[0],ku_mat.shape[1]),dtype='f8',data=V_mat)
    data.create_dataset('P',(ku_mat.shape[0],ku_mat.shape[1]),dtype='f8',data=P_mat)

    #1d arrays
    data.create_dataset('lons',(n_prof,),dtype='f8',data=lons)
    data.create_dataset('lats',(n_prof,),dtype='f8',data=lats)
    data.create_dataset('clut_ku',(n_prof,),dtype='i4',data=clut_ku)
    data.create_dataset('clut_ka',(n_prof,),dtype='i4',data=clut_ka)
    data.create_dataset('pathAtten',(n_prof,),dtype='f8',data=pathAtten)
    data.create_dataset('reliabFlag',(n_prof,),dtype='i4',data=reliabFlag)
    data.create_dataset('cross_ind',(n_prof,),dtype='i4',data=cross_ind)
    data.create_dataset('time_GPM',(n_prof,),dtype='f8',data=time)
    data.create_dataset('1x1',(onebyone.shape[0],onebyone.shape[1],onebyone.shape[2]),dtype='i4',data=onebyone)
    data.create_dataset('2x2',(twobytwo.shape[0],twobytwo.shape[1],twobytwo.shape[2]),dtype='i4',data=twobytwo)
    time_merra = np.asarray(time_merra,dtype="S10")
    data.create_dataset('time_merra',(n_prof,),dtype='S10',data=time_merra)
    
    myfile.close()
    
    return


##########################
#Main Loop over GPM files START 
##########################

#define which year you wish to process
year = 2014
year = str(year)

#define which months
month = np.arange(4,5)


#loop over all months designated 
for i in month:
    
    #pad month for filenames and saving if needed 
    if i < 10:
        ii = '0' + str(i)
    else:
        ii = str(i)
        
    #get all files 
    files = glob.glob('/data/gpm/a/snesbitt/V05/arthurhou.pps.eosdis.nasa.gov/gpmdata/'+ year +'/'+ ii + '/*/radar/2A.GPM.DPR.V7*')
    
    #sort them in order so save file is in time order 
    files.sort()
    
    #for timing purposes
    start = time.time()
    #get workers
    pool = mp.Pool(processes=28, maxtasksperchild=1)
    #process in parallel
    d = pool.map(get_heavy_ice,files)
    #send workers home
    pool.close()
    #see how long it took
    end = time.time()
    print('extracting gpm takes: ', end - start)
    #lets save as monthly files
    savestr = '/data/gpm/a/randyjc2/HeavyIceFiles/'+ year + '/flagheavyice_' + ii + '.hdf5'
    
    #save as hdf file to save alittle space
    make_hdf(d,savestr)
    
##########################
#Main Loop over GPM files END
##########################
