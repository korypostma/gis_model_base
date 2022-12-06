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
glblhistogram = {}
glblstats = {}

def clear_global_stats():
    global glblhistogram
    global glblstats
    glblhistogram = {}
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

def calculate_global_stats(values, ignore_vals = None):
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
        this_global_stats = GlobalStats()
        return this_global_stats
    if (val_len == 1):
        this_global_stats = GlobalStats()
        val = list(values.keys())[0]
        this_global_stats.sum = val
        this_global_stats.count = 1
        this_global_stats.minimum = val
        this_global_stats.maximum = val
        this_global_stats.median = val
        this_global_stats.mean = val
        this_global_stats.stddev = 0.0
        return this_global_stats

    from collections import OrderedDict
    sorted_dict = OrderedDict(sorted(values.items()))
    #print(sorted_dict)
    #Determine the number of all of the counts, how many pixels total?
    val = 0
    sum_of_values = 0
    if (len(sorted_dict.values()) > 0):
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

    #stddev
    sqsum = 0.0
    total_count = 0
    total_sum = 0.0
    for key,val in sorted_dict.items():
        total_count += val
        total_sum += key*val

    mean = (total_sum / total_count) if total_count != 0 else 0.0

    for key,val in sorted_dict.items():
        diff = key - mean
        #account for the count of this histogram key by multiplying by val
        sqsum += val * (diff*diff)
    stddev = math.sqrt(sqsum / (total_count - 1)) if total_count > 1 else 0.0

    #print(median)
    this_global_stats = GlobalStats()
    this_global_stats.sum = total_sum
    this_global_stats.count = total_count
    this_global_stats.minimum = minimum
    this_global_stats.maximum = maximum
    this_global_stats.median = median
    this_global_stats.mean = mean
    this_global_stats.stddev = stddev
    return this_global_stats

def global_stats_histogram(in_data, no_data):
    global glblhistogram_dtype
    global glblhistogram
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
        mask = (in_1 == no_data[band_key])
        if 'aoi(1)' in in_data:
            mask = (mask | (in_data['aoi(1)'] == 0))
        if 'mask(1)' in in_data:
            mask = (mask | (in_data['mask(1)'] == 1))

        #Calculate binning of unique values for median
        vals,cnts = np.unique(np.ma.masked_array(in_1, mask=mask).compressed().astype(glblhistogram_dtype), return_counts=True)
        for val_i in range(vals.shape[0]):
            try:
                if glblhistogram[i] is None:
                    glblhistogram[i] = {}
            except KeyError:
                glblhistogram[i] = {}
            try:
                glblhistogram[i][vals[val_i]] += cnts[val_i]
            except KeyError:
                glblhistogram[i][vals[val_i]] = cnts[val_i]
        del in_1

    out_data = {}
    return out_data

def get_histogram(input_file, input_name = 'in', aoi=None, mask=None, stats = ['sum','count','min','median','max','mean','stddev'], ignore_vals = None, histogram_dtype=np.int32, force_blocksize=None):
    global glblhistogram_dtype
    global glblhistogram
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

    main(input_files, output_parameters, func=global_stats_histogram, force_blocksize=force_blocksize, args=None)

    ###START OF MPI CODE
    if is_mpi_enabled():
        #workers must send stats to master, master will collect all stats and merge them
        from mpi4py import MPI
        mpi_world = MPI.COMM_WORLD
        mpi_size = mpi_world.size
        mpi_world.barrier()
        if is_mpi_master() is False:
            #We are a worker, send all global stats to master
            mpi_world.send(glblhistogram, dest=0)
        else:
            print('Receiving global stats from workers...')
            #We are the master, we must receive the global stats from each worker
            for n_recvd in range(1,mpi_size): #does not include master
                status = MPI.Status()
                worker_histogram = mpi_world.recv(source = MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                requesting_rank = status.Get_source()
                #print('Received global stats from',requesting_rank,'...')
                #merge worker global stats with ours
                bands = len(worker_histogram)
                for i in range(bands):
                    if n_recvd == 1:
                        #If this is the first worker then we must initialize these with stats from that worker
                        glblhistogram[i] = worker_histogram[i]
                    else:
                        #merge median dictionaries together
                        for key,val in worker_histogram[i].items():
                            try:
                                glblhistogram[i][key] += worker_histogram[i][key]
                            except KeyError:
                                glblhistogram[i][key] = val

        mpi_world.barrier()
    ###END OF MPI CODE

    #This will only return False if we are MPI enabled and we are a worker, we only want to do this on masters and single-process runs
    if is_mpi_master() is True:
        bands = len(glblhistogram)
        for i in range(bands):
            glblstats[i] = GlobalStats()
            #(minimum,median,maximum) = get_min_median_max(glblhistogram[i], ignore_vals = [0])
            this_global_stats = calculate_global_stats(glblhistogram[i], ignore_vals = ignore_vals)
            glblstats[i].sum = this_global_stats.sum
            glblstats[i].count = this_global_stats.count
            glblstats[i].minimum = this_global_stats.minimum
            glblstats[i].maximum = this_global_stats.maximum
            glblstats[i].median = this_global_stats.median
            glblstats[i].mean = this_global_stats.mean
            glblstats[i].stddev = this_global_stats.stddev

            print('Band',str(i+1),': ','SUM:',glblstats[i].sum,'COUNT:',glblstats[i].count,'MIN:',glblstats[i].minimum,'MAX:',glblstats[i].maximum,'MEDIAN:',glblstats[i].median,'MEAN:',glblstats[i].mean,'STDDEV:',glblstats[i].stddev)

    if is_mpi_master() is True:
        output_global_stats = {}
        bands = len(glblhistogram)
        for i in range(bands):
            output_global_stats[input_name+'('+str(i+1)+')'] = glblstats[i]
            print('Band',str(i+1),': ','SUM:',glblstats[i].sum,'COUNT:',glblstats[i].count,'MIN:',glblstats[i].minimum,'MAX:',glblstats[i].maximum,'MEDIAN:',glblstats[i].median,'MEAN:',glblstats[i].mean,'STDDEV:',glblstats[i].stddev)

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
