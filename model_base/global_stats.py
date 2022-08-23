import sys
import math
import time

from model_base.model_base import *
from osgeo import gdal
import numpy as np

class GlobalStats:
    def __init__(self):
        self.sum = 0.0
        self.count = 0
        self.minimum = 0.0
        self.maximum = 0.0
        self.median = 0.0
        self.mean = 0.0
        self.stddev = 0.0

glblhistogram_dtype = np.uint16
glblmdn = {}
glblmin = {}
glblmax = {}
glblsum = {}
glblcount = {}
glblmean = {}
glblsqsum = {}
glblstats = {}

def clear_global_stats():
    global glblmdn
    global glblmin
    global glblmax
    global glblsum
    global glblcount
    global glblmean
    global glblsqsum
    global glblstats
    glblmdn = {}
    glblmin = {}
    glblmax = {}
    glblsum = {}
    glblcount = {}
    glblmean = {}
    glblsqsum = {}
    glblstats = {}

def is_mpi_enabled():
    """
    Only return True if MPI size is > 1
    """
    try:
        from mpi4py import MPI
        mpi_world = MPI.COMM_WORLD
        mpi_size = mpi_world.size
        return (mpi_size > 1)
    except ModuleNotFoundError:
        return False
    except ImportError:
        return False

def is_mpi_master():
    """
    Only return False if this is a worker
    """
    try:
        from mpi4py import MPI
        mpi_world = MPI.COMM_WORLD
        mpi_size = mpi_world.size
        if (mpi_size > 1):
            return (mpi_world.rank == 0)
        else:
            return True
    except ModuleNotFoundError:
        return True
    except ImportError:
        return True

def get_min_median_max(values, ignore_vals = None):
    if (ignore_vals is not None):
        for ignore_val in ignore_vals:
            #Ignore 0
            try:
                del values[ignore_val]
            except KeyError:
                pass

    val_len = len(values)
    #print('get_min_median_max input len: ', val_len)
    if (val_len <= 0):
        return (0.0, 0.0, 0.0)
    if (val_len == 1):
        return (list(values.keys())[0], list(values.keys())[0], list(values.keys())[0])

    from collections import OrderedDict
    sorted_dict = OrderedDict(sorted(values.items()))
    #print(sorted_dict)
    #Determine the number of all of the counts, how many pixels total?
    val = 0
    if (len(sorted_dict.values()) > 0):
        sum_of_values = 0
        for val in sorted_dict.values():
            sum_of_values += val
        half = sum_of_values / 2

        #Finds the index of the middle of the dataset
        sum_var = 0
        index = 0
        for val in sorted_dict.values():
            if half-sum_var > 0:
                sum_var += val
                index += 1
            else:
                break

        #index = (list(sorted_dict.values()).index(val))
        #Returns the median based off some characteristics of the dataset
        upperhalf_count = sum(list(sorted_dict.values())[index:])
        lowerhalf_count = sum(list(sorted_dict.values())[:index])
        if upperhalf_count != lowerhalf_count:
            if upperhalf_count > lowerhalf_count:
                median = list(sorted_dict.keys())[index]
            else:
                median = list(sorted_dict.keys())[index-1]
        else:
            median = (list(sorted_dict.keys())[index-1] + list(sorted_dict.keys())[index]) / 2
        minimum = list(sorted_dict.keys())[0]
        maximum = list(sorted_dict.keys())[-1]
        #print('val:',val,'sum_of_values:',sum_of_values,'half:',half,'sum_var:',sum_var,'index:',index,'lh_cnt:',lowerhalf_count,'uh_cnt:',upperhalf_count,'med:',median,'min:',minimum,'max:',maximum,'hist:',sorted_dict.items())
    else:
        #print('missing median')
        median = 0.0
        minimum = 0.0
        maximum = 0.0

    #print(median)
    return (minimum, median, maximum)

def global_stats_stddev(in_data, no_data):
    global glblmean
    global glblsqsum
    #block_shape = in_data['in(1)'].shape
    #bands = len(in_data)
    bands = 0
    for key,val in in_data.items():
        if key.startswith('in'):
            bands += 1
    for i in range(bands):
        band_num = i+1
        band_key = 'in('+str(band_num)+')'
        in_1 = in_data[band_key]
        diff = in_1 - glblmean[i]
        if 'aoi(1)' in in_data:
            aoi_mask = (in_data['aoi(1)'] == 0)
            diff[aoi_mask] = 0.0
            del aoi_mask
        if 'mask(1)' in in_data:
            aoi_mask = (in_data['mask(1)'] == 1)
            diff[aoi_mask] = 0.0
            del aoi_mask
        mask = (in_1 == 0) | (in_1 == no_data[band_key])
        diff[mask] = 0.0
        tot_sum = np.sum((diff * diff), dtype=np.float64)
        try:
            glblsqsum[i] += tot_sum
        except KeyError:
            glblsqsum[i] = tot_sum
        del diff
        del in_1

    out_data = {}
    return out_data

def global_stats_first_pass_with_median(in_data, no_data):
    global glblhistogram_dtype
    global glblmdn
    global glblmin
    global glblmax
    global glblsum
    global glblcount
    #block_shape = in_data['in(1)'].shape
    #bands = len(in_data)
    bands = 0
    for key,val in in_data.items():
        if key.startswith('in'):
            bands += 1
    for i in range(bands):
        band_num = i+1
        band_key = 'in('+str(band_num)+')'
        in_1 = in_data[band_key]
        if 'aoi(1)' in in_data:
            aoi_mask = (in_data['aoi(1)'] == 0)
            in_1[aoi_mask] = 0
            del aoi_mask
        if 'mask(1)' in in_data:
            aoi_mask = (in_data['mask(1)'] == 1)
            in_1[aoi_mask] = 0.0
            del aoi_mask
        mask = (in_1 == no_data[band_key])
        in_1[mask] = 0

        glblmin[i] = 0.0
        glblmax[i] = 0.0

        #Calculate sum
        try:
            glblsum[i] += np.sum(in_1, dtype=np.float64)
        except KeyError:
            glblsum[i] = np.sum(in_1, dtype=np.float64)

        #Calculate non-zero count
        try:
            glblcount[i] += np.count_nonzero(in_1)
        except KeyError:
            glblcount[i] = np.count_nonzero(in_1)

        #Calculate binning of unique values for median
        vals,cnts = np.unique(in_1.astype(glblhistogram_dtype), return_counts=True)
        for val_i in range(vals.shape[0]):
            try:
                if glblmdn[i] is None:
                    glblmdn[i] = {}
            except KeyError:
                glblmdn[i] = {}
            try:
                glblmdn[i][vals[val_i]] += cnts[val_i]
            except KeyError:
                glblmdn[i][vals[val_i]] = cnts[val_i]
        del in_1

    out_data = {}
    return out_data

def global_stats_first_pass_no_median(in_data, no_data):
    global glblmdn
    global glblmin
    global glblmax
    global glblsum
    global glblcount
    #block_shape = in_data['in(1)'].shape
    #bands = len(in_data)
    bands = 0
    for key,val in in_data.items():
        if key.startswith('in'):
            bands += 1
    for i in range(bands):
        band_num = i+1
        band_key = 'in('+str(band_num)+')'
        in_1 = in_data[band_key]
        if 'aoi(1)' in in_data:
            aoi_mask = (in_data['aoi(1)'] == 0)
            in_1[aoi_mask] = 0
            del aoi_mask
        if 'mask(1)' in in_data:
            aoi_mask = (in_data['mask(1)'] == 1)
            in_1[aoi_mask] = 0.0
            del aoi_mask
        mask = (in_1 == no_data[band_key])
        in_1[mask] = 0

        glblmdn[i] = {}

        #Calculate sum
        try:
            glblsum[i] += np.sum(in_1, dtype=np.float64)
        except KeyError:
            glblsum[i] = np.sum(in_1, dtype=np.float64)

        #Calculate non-zero count
        try:
            glblcount[i] += np.count_nonzero(in_1)
        except KeyError:
            glblcount[i] = np.count_nonzero(in_1)

        #Calculate min/max without median histogram
        in_min = np.min(in_1)
        try:
            if (in_min < glblmin[i]):
                glblmin[i] = in_min
        except KeyError:
            glblmin[i] = in_min

        in_max = np.max(in_1)
        try:
            if (in_max > glblmax[i]):
                glblmax[i] = in_max
        except KeyError:
            glblmax[i] = in_max

        del in_1

    out_data = {}
    return out_data

def get_global_stats(input_file, input_name = 'in', aoi=None, mask=None, stats = ['sum','count','min','median','max','mean','stddev'], histogram_dtype=np.uint16, force_blocksize=None):
    global glblhistogram_dtype
    global glblmdn
    global glblmin
    global glblmax
    global glblsum
    global glblcount
    global glblmean
    global glblsqsum
    global glblstats

    global_start_time = time.perf_counter()

    clear_global_stats()

    input_files = {
            'in': input_file,
            }

    if aoi is not None:
        input_files['aoi'] = aoi

    if mask is not None:
        input_files['mask'] = mask

    output_files = {}
    output_datatypes = {}
    output_band_counts = {}
    output_drivers = {}
    output_options = {}

    output_parameters = (output_files, output_datatypes, output_band_counts, output_drivers, output_options)

    glblhistogram_dtype = histogram_dtype

    if ('median' in stats):
        main(input_files, output_parameters, func=global_stats_first_pass_with_median, force_blocksize=force_blocksize, args=None)
    else:
        main(input_files, output_parameters, func=global_stats_first_pass_no_median, force_blocksize=force_blocksize, args=None)

    ###START OF MPI CODE
    if is_mpi_enabled():
        #workers must send stats to master, master will collect all stats and merge them
        from mpi4py import MPI
        mpi_world = MPI.COMM_WORLD
        mpi_size = mpi_world.size
        mpi_world.barrier()
        if is_mpi_master() is False:
            #We are a worker, send all global stats to master
            mpi_world.send((glblmdn, glblsum, glblcount, glblmin, glblmax), dest=0)
        else:
            print('Receiving global stats from workers...')
            #We are the master, we must receive the global stats from each worker
            for n_recvd in range(1,mpi_size): #does not include master
                status = MPI.Status()
                (worker_glblmdn, worker_glblsum, worker_glblcount, worker_glblmin, worker_glblmax) = mpi_world.recv(source = MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                requesting_rank = status.Get_source()
                #print('Received global stats from',requesting_rank,'...')
                #merge worker global stats with ours
                bands = len(worker_glblcount)
                for i in range(bands):
                    if n_recvd == 1:
                        #If this is the first worker then we must initialize these with stats from that worker
                        glblmdn[i] = worker_glblmdn[i]
                        glblsum[i] = worker_glblsum[i]
                        glblcount[i] = worker_glblcount[i]
                        glblmin[i] = worker_glblmin[i]
                        glblmax[i] = worker_glblmax[i]
                    else:
                        #merge median dictionaries together
                        for key,val in worker_glblmdn[i].items():
                            try:
                                glblmdn[i][key] += worker_glblmdn[i][key]
                            except KeyError:
                                glblmdn[i][key] = val
                        glblsum[i] += worker_glblsum[i]
                        glblcount[i] += worker_glblcount[i]
                        if (worker_glblmin[i] < glblmin[i]):
                            glblmin[i] = worker_glblmin[i]
                        if (worker_glblmax[i] > glblmax[i]):
                            glblmax[i] = worker_glblmax[i]

        mpi_world.barrier()
    ###END OF MPI CODE

    #This will only return False if we are MPI enabled and we are a worker, we only want to do this on masters and single-process runs
    if is_mpi_master() is True:
        print('sum=',glblsum)
        print('count=',glblcount)
        bands = len(glblcount)
        for i in range(bands):
            glblstats[i] = GlobalStats()
            if ('median' in stats):
                (minimum,median,maximum) = get_min_median_max(glblmdn[i], ignore_vals = [0])
                glblstats[i].minimum = minimum
                glblstats[i].maximum = maximum
                glblstats[i].median = median
            else:
                try:
                    glblstats[i].minimum = glblmin[i]
                except KeyError:
                    glblstats[i].minimum = 0.0
                try:
                    glblstats[i].maximum = glblmax[i]
                except KeyError:
                    glblstats[i].maximum = 0.0
                glblstats[i].median = 0.0

            try:
                band_sum = glblsum[i]
                glblstats[i].sum += band_sum
            except KeyError:
                print('band',str(i+1),'missing glblsum')
                band_sum = 0.0

            try:
                band_count = glblcount[i]
                glblstats[i].count += band_count
            except KeyError:
                print('band',str(i+1),'missing glblcount')
                band_count = 0

            glblmean[i] = (band_sum / band_count) if band_count != 0 else 0.0
            glblstats[i].mean = glblmean[i]
            #print('i',str(i),':','glblmean:',glblmean[i])
            print('Band',str(i+1),': ','MIN:',glblstats[i].minimum,'MAX:',glblstats[i].maximum,'MEDIAN:',glblstats[i].median,'MEAN:',glblstats[i].mean)

    if ('stddev' in stats):
        ###START OF MPI CODE
        if is_mpi_enabled():
            from mpi4py import MPI
            mpi_world = MPI.COMM_WORLD
            mpi_size = mpi_world.size
            mpi_world.barrier()
            #master only needs to send glblmean[i] to all workers
            if is_mpi_master() is False:
                #We are a worker, receive global mean from master
                glblmean = mpi_world.recv(source=0)
            else:
                print('Sending global mean to workers...')
                #We are the master, send global mean to all workers
                for n_recvd in range(1,mpi_size): #does not include master
                    mpi_world.send(glblmean, dest=n_recvd)
                    #print('Sent global mean to',n_recvd,'...')
        ###END OF MPI CODE

        main(input_files, output_parameters, func=global_stats_stddev, force_blocksize=force_blocksize, args=None)

        ###START OF MPI CODE
        if is_mpi_enabled():
            from mpi4py import MPI
            mpi_world = MPI.COMM_WORLD
            mpi_size = mpi_world.size
            mpi_world.barrier()
            #workers send glblsqsum to master
            if is_mpi_master() is False:
                #We are a worker, send all global stats to master
                mpi_world.send(glblsqsum, dest=0)
            else:
                print('Receiving global sq. sum from workers...')
                #We are the master, we must receive the global stats from each worker
                for n_recvd in range(1,mpi_size): #does not include master
                    status = MPI.Status()
                    worker_glblsqsum = mpi_world.recv(source = MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                    requesting_rank = status.Get_source()
                    #print('Received global sq. sum from',requesting_rank,'...')
                    if (n_recvd == 1):
                        glblsqsum = worker_glblsqsum
                    else:
                        bands = len(glblcount)
                        for i in range(bands):
                            glblsqsum[i] += worker_glblsqsum[i]
        ###END OF MPI CODE

        if is_mpi_master() is True:
            for i in range(bands):
                glblstats[i].stddev = math.sqrt(glblsqsum[i] / (glblcount[i] - 1)) if glblcount[i] > 1 else 0.0
    else:
        glblstats[i].stddev = 0.0

    if is_mpi_master() is True:
        output_global_stats = {}
        for i in range(bands):
            output_global_stats[input_name+'('+str(i+1)+')'] = glblstats[i]
            print('Band ',str(i+1),': ','MIN:',glblstats[i].minimum,'MAX:',glblstats[i].maximum,'MEDIAN:',glblstats[i].median,'MEAN:',glblstats[i].mean,'STD DEV:',glblstats[i].stddev)

    ###START OF MPI CODE
    if is_mpi_enabled():
        from mpi4py import MPI
        mpi_world = MPI.COMM_WORLD
        mpi_size = mpi_world.size
        mpi_world.barrier()
        #master send output_global_stats to each worker
        if is_mpi_master() is False:
            #We are a worker, receive output_global_stats from master
            output_global_stats = mpi_world.recv(source=0)
        else:
            print('Sending output_global_stats to workers...')
            #We are the master, send output_global_stats to all workers
            for n_recvd in range(1,mpi_size): #does not include master
                mpi_world.send(output_global_stats, dest=n_recvd)
                #print('Sent output_global_stats to',n_recvd,'...')
    ###END OF MPI CODE

    global_stop_time = time.perf_counter()
    global_total_time = global_stop_time - global_start_time
    if is_mpi_master():
        print('Global Stats total time:',global_total_time,'secs',timedelta(seconds=global_total_time))

    return output_global_stats
