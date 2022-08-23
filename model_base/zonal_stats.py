import sys
import math
import time

from model_base.model_base import *
from osgeo import gdal
import numpy as np

class ZonalStatsPerBandStruct:
    def __init__(self):
        self.zonal_counts = {} #ZonalCounts
        self.zonal_stats = {} #ZonalStats

class ZonalCounts:
    def __init__(self):
        self.totalsum = 0.0
        self.count = 0
        self.histogram = {}
        self.sqsum = 0.0

class ZonalStats:
    def __init__(self):
        self.minimum = 0.0
        self.maximum = 0.0
        self.median = 0.0
        self.mean = 0.0
        self.stddev = 0.0
        self.diversity = 0.0
        self.majority = 0.0

#Dictionary of ZonalStatsPerBandStruct objects
glblhistogram_dtype = np.uint16
zonal_bands = {}

def clear_zonal_stats():
    global zonal_bands
    zonal_bands = {}

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

def merge_zonal_bands(in_zonal_bands):
    global zonal_bands

    bands = len(in_zonal_bands)
    for band_i in range(bands):
        band_num = band_i+1

        if band_num not in zonal_bands:
            zonal_bands[band_num] = ZonalStatsPerBandStruct()

        zones = in_zonal_bands[band_num].zonal_counts.keys()

        for zone in zones:
            if zone not in zonal_bands[band_num].zonal_counts:
                zonal_bands[band_num].zonal_counts[zone] = ZonalCounts()

            #Add total sum
            zonal_bands[band_num].zonal_counts[zone].totalsum += in_zonal_bands[band_num].zonal_counts[zone].totalsum
            #Add count
            zonal_bands[band_num].zonal_counts[zone].count += in_zonal_bands[band_num].zonal_counts[zone].count
            #Add histograms
            in_histogram = in_zonal_bands[band_num].zonal_counts[zone].histogram
            for key,val in in_histogram.items():
                if key not in zonal_bands[band_num].zonal_counts[zone].histogram:
                    zonal_bands[band_num].zonal_counts[zone].histogram[key] = val
                else:
                    zonal_bands[band_num].zonal_counts[zone].histogram[key] += val

        del zones

def merge_zonal_bands_sqsum(in_zonal_bands):
    global zonal_bands

    bands = len(in_zonal_bands)
    for band_i in range(bands):
        band_num = band_i+1

        if band_num not in zonal_bands:
            zonal_bands[band_num] = ZonalStatsPerBandStruct()

        zones = in_zonal_bands[band_num].zonal_counts.keys()

        for zone in zones:
            if zone not in zonal_bands[band_num].zonal_counts:
                zonal_bands[band_num].zonal_counts[zone] = ZonalCounts()

            #Add square sum
            zonal_bands[band_num].zonal_counts[zone].sqsum += in_zonal_bands[band_num].zonal_counts[zone].sqsum

        del zones

def zonal_stats_mean(in_data, no_data):
    global zonal_bands

    zones = np.unique(in_data['zone(1)'].astype(np.uintc))

    #subtract one for the zone raster
    bands = len(in_data) - 1
    for band_i in range(bands):
        band_num = band_i+1
        band_key = 'in('+str(band_num)+')'
        in_1 = in_data[band_key].astype(np.uintc)

        if band_num not in zonal_bands:
            zonal_bands[band_num] = ZonalStatsPerBandStruct()

        for zone in zones:
            zone_mask = (in_data['zone(1)'] == zone)
            zone_data = in_1[zone_mask]
            del zone_mask

            if zone not in zonal_bands[band_num].zonal_counts:
                zonal_bands[band_num].zonal_counts[zone] = ZonalCounts()

            #Calculate sum
            zonal_bands[band_num].zonal_counts[zone].totalsum += np.sum(zone_data, dtype=np.uint)
            #Calculate non-zero count
            zonal_bands[band_num].zonal_counts[zone].count += np.count_nonzero(zone_data)

            del zone_data

        del in_1
        #del zonal_counts

    del zones
    out_data = {}
    return out_data

def zonal_stats_stddev(in_data, no_data):
    global zonal_bands
    #block_shape = in_data['in(1)'].shape
    #zones,counts = np.unique(in_data['zone(1)'].astype(np.uintc), return_counts=True)
    zones = np.unique(in_data['zone(1)'].astype(np.uintc))

    #subtract one for the zone raster
    bands = len(in_data) - 1
    for band_i in range(bands):
        band_num = band_i+1
        band_key = 'in('+str(band_num)+')'
        in_1 = in_data[band_key].astype(np.uintc)

        for zone in zones:
            zone_mask = (in_data['zone(1)'] == zone)
            zone_data = in_1[zone_mask]
            del zone_mask

            zonal_count = zonal_bands[band_num].zonal_counts[zone]
            zonal_stat = zonal_bands[band_num].zonal_stats[zone]

            diff = zone_data - zonal_stat.mean
            mask = (zone_data == 0) | (zone_data == no_data[band_key])
            diff[mask] = 0.0
            tot_sum = np.sum((diff * diff), dtype=np.float64)
            try:
                zonal_count.sqsum += tot_sum
            except KeyError:
                zonal_count.sqsum = tot_sum
            del diff
            del zone_data

        del in_1

    out_data = {}
    return out_data

def zonal_stats_first_pass(in_data, no_data):
    global glblhistogram_dtype
    global zonal_bands
    #block_shape = in_data['in(1)'].shape

    #zones,counts = np.unique(in_data['zone(1)'].astype(np.uintc), return_counts=True)
    zones = np.unique(in_data['zone(1)'].astype(np.uintc))

    #print(in_data['zone(1)'])
    #try:
    #    print('1:',zonal_bands[1].zonal_counts[3002].count)
    #    print('2:',zonal_bands[2].zonal_counts[3002].count)
    #    print('3:',zonal_bands[3].zonal_counts[3002].count)
    #    print('4:',zonal_bands[4].zonal_counts[3002].count)
    #    print('5:',zonal_bands[5].zonal_counts[3002].count)
    #    print('6:',zonal_bands[6].zonal_counts[3002].count)
    #except KeyError:
    #    print('no keys')

    #subtract one for the zone raster
    bands = len(in_data) - 1
    for band_i in range(bands):
        band_num = band_i+1
        band_key = 'in('+str(band_num)+')'
        in_1 = in_data[band_key].astype(np.uintc)
        #mask = (in_1 == no_data[band_key])
        #in_1[mask] = 0
        #del mask
        #print(in_1)

        if band_num not in zonal_bands:
            #print('Created new ZonalStatsPerBandStruct for band:',band_num)
            zonal_bands[band_num] = ZonalStatsPerBandStruct()

        for zone in zones:
            zone_mask = (in_data['zone(1)'] == zone)
            zone_data = in_1[zone_mask]
            del zone_mask

            #if (zone == 3002):
            #    print(zone_data)

            if zone not in zonal_bands[band_num].zonal_counts:
                #print('Created new ZonalCounts for zone:',zone)
                zonal_bands[band_num].zonal_counts[zone] = ZonalCounts()
                #zonal_bands[band_num].zonal_counts[zone].totalsum = 0.0
                #zonal_bands[band_num].zonal_counts[zone].count = 0
                #zonal_bands[band_num].zonal_counts[zone].histogram = {}

            #Calculate sum
            zonal_bands[band_num].zonal_counts[zone].totalsum += np.sum(zone_data, dtype=np.uint)
            #Calculate non-zero count
            #if (zone == 3002):
            #    print(band_num,'before_count:',zonal_bands[band_num].zonal_counts[zone].count)
            #    print(band_num,'nonzero_count:',np.count_nonzero(zone_data))
            zonal_bands[band_num].zonal_counts[zone].count += np.count_nonzero(zone_data)
            #if (zone == 3002):
            #    print(band_num,'final_count:',zonal_bands[band_num].zonal_counts[zone].count)

            #Calculate binning of unique values for median
            histogram = zonal_bands[band_num].zonal_counts[zone].histogram
            vals,cnts = np.unique(zone_data.astype(glblhistogram_dtype), return_counts=True)
            for val_i in range(vals.shape[0]):
                val = vals[val_i]
                cnt = cnts[val_i]
                if val not in histogram:
                    histogram[val] = cnt
                else:
                    histogram[val] += cnt
            #print('zone:',zone,' zonal_bands[band_num].zonal_counts.count:',zonal_bands[band_num].zonal_counts[zone].count)
            del vals
            del cnts
            del histogram

            del zone_data

        del in_1
        #del zonal_counts

    del zones
    out_data = {}
    return out_data

def zonal_stats_output(in_data, no_data, stats = None):
    out_data = {}
    global zonal_bands
    block_shape = in_data['zone(1)'].shape
    #zones,counts = np.unique(in_data['zone(1)'].astype(np.uintc), return_counts=True)
    zones = np.unique(in_data['zone(1)'].astype(np.uintc))

    #subtract one for the zone raster
    bands = len(in_data) - 1
    for band_i in range(bands):
        band_num = band_i+1
        band_key = 'in('+str(band_num)+')'
        in_1 = in_data[band_key].astype(np.uintc)
        #stats = ['min','median','max','mean','stddev']
        for stat in stats:
            out_data[str(stat)+'('+str(band_num)+')'] = np.zeros(block_shape, dtype=np.float)

        for zone in zones:
            zone_mask = (in_data['zone(1)'] == zone)
            zone_data = in_1[zone_mask]

            zonal_count = zonal_bands[band_num].zonal_counts[zone]
            zonal_stat = zonal_bands[band_num].zonal_stats[zone]

            #stats = ['min','median','max','mean','stddev']
            if ('min' in stats):
                out_data['min('+str(band_num)+')'][zone_mask] = zonal_stat.minimum
            if ('median' in stats):
                out_data['median('+str(band_num)+')'][zone_mask] = zonal_stat.median
            if ('max' in stats):
                out_data['max('+str(band_num)+')'][zone_mask] = zonal_stat.maximum
            if ('mean' in stats):
                out_data['mean('+str(band_num)+')'][zone_mask] = zonal_stat.mean
            if ('stddev' in stats):
                out_data['stddev('+str(band_num)+')'][zone_mask] = zonal_stat.stddev
            if ('diversity' in stats):
                out_data['diversity('+str(band_num)+')'][zone_mask] = zonal_stat.diversity
            if ('majority' in stats):
                out_data['majority('+str(band_num)+')'][zone_mask] = zonal_stat.majority
            del zone_mask

    return out_data

def get_diversity_majority(values, ignore_vals = None):
    if (ignore_vals is not None):
        for ignore_val in ignore_vals:
            try:
                del values[ignore_val]
            except KeyError:
                pass

    vals = list(values.keys())
    cnts = list(values.values())
    diversity = len(vals)
    majority = vals[np.argmax(cnts)]

    return (diversity,majority)

def get_min_median_max(values, ignore_vals = None):
    val_len = len(values)
    #print('get_min_median_max input len: ', val_len)

    if (ignore_vals is not None):
        for ignore_val in ignore_vals:
            try:
                del values[ignore_val]
            except KeyError:
                pass

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
            median = ((list(sorted_dict.keys())[index-1]).astype(float) + (list(sorted_dict.keys())[index]).astype(float)) / 2
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

def get_zonal_mean(input_file, zone_file, input_name = 'in', output_parameters = (), ignore_vals = [0], force_blocksize=None):
    global zonal_bands

    zonal_start_time = time.perf_counter()

    clear_zonal_stats()

    input_files = {
            'in': input_file,
            'zone': zone_file,
            }

    output_files = {}
    output_datatypes = {}
    output_band_counts = {}
    output_drivers = {}
    output_options = {}

    empty_output_parameters = (output_files, output_datatypes, output_band_counts, output_drivers, output_options)

    main(input_files, empty_output_parameters, func=zonal_stats_mean, force_blocksize=force_blocksize, args=None)

    ###START OF MPI CODE
    if is_mpi_enabled():
        #workers must send stats to master, master will collect all stats and merge them
        from mpi4py import MPI
        mpi_world = MPI.COMM_WORLD
        mpi_size = mpi_world.size
        mpi_world.barrier()
        if is_mpi_master() is False:
            #We are a worker, send all zonal stats to master
            mpi_world.send(zonal_bands, dest=0)
        else:
            print('Receiving zonal bands from workers and merging them...')
            #We are the master, we must receive the zonal stats from each worker
            for n_recvd in range(1,mpi_size): #does not include master
                status = MPI.Status()
                worker_zonal_bands = mpi_world.recv(source = MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                requesting_rank = status.Get_source()
                #print('Received zonal bands from',requesting_rank,'...')
                #merge worker zonal stats with ours
                merge_zonal_bands(worker_zonal_bands)
        mpi_world.barrier()
    ###END OF MPI CODE

    if is_mpi_master():
        for band in zonal_bands.keys():
            zonal_counts = zonal_bands[band].zonal_counts
            for zone in zonal_counts.keys():
                zonal_count = zonal_counts[zone]
                mean = (zonal_count.totalsum / zonal_count.count) if zonal_count.count != 0 else 0.0
                #if (zonal_count.count >= 3):
                #    print('Band ',str(band),' (Zone: ',str(zone),')','SUM:',zonal_count.totalsum,'COUNT:',zonal_count.count,'HIST.keys.count:',len(zonal_count.histogram.keys()),'minimum:',minimum,'median:',median,'mean:',mean,'maximum:',maximum)
                del zonal_count

                zonal_bands[band].zonal_stats[zone] = ZonalStats()
                zonal_bands[band].zonal_stats[zone].mean = mean

            del zonal_counts

    ###START OF MPI CODE
    if is_mpi_enabled():
        #master sends all workers the updated zonal_stats
        from mpi4py import MPI
        mpi_world = MPI.COMM_WORLD
        mpi_size = mpi_world.size
        mpi_world.barrier()
        if is_mpi_master() is False:
            #We are a worker, receive zonal stats from master
            zonal_bands = mpi_world.recv(source=0)
        else:
            print('Sending zonal bands to workers...')
            #We are the master, send all workers the zonal_stats
            for n_recvd in range(1,mpi_size): #does not include master
                mpi_world.send(zonal_bands, dest=n_recvd)
                #print('Sent zonal bands to',n_recvd,'...')
        mpi_world.barrier()
    ###END OF MPI CODE

    output_zonal_stats = {}
    for band in zonal_bands.keys():
        output_zonal_stats[input_name+'('+str(band)+')'] = zonal_bands[band].zonal_stats
        if is_mpi_master():
            print('Band ',str(band),': ','Zonal_Stats Count:',len(zonal_bands[band].zonal_stats.keys()))

    if (len(output_parameters) > 0):
        args = {}
        stats = ['mean']
        args['stats'] = stats
        main(input_files, output_parameters, func=zonal_stats_output, force_blocksize=force_blocksize, args=args)

    zonal_stop_time = time.perf_counter()
    zonal_total_time = zonal_stop_time - zonal_start_time
    if is_mpi_master():
        print('Zonal Stats total time:',zonal_total_time,'secs',timedelta(seconds=zonal_total_time))

    return output_zonal_stats

def get_zonal_stats(input_file, zone_file, input_name = 'in', stats = ['min','median','max','mean','stddev','majority','diversity'], output_parameters = (), histogram_dtype = np.uint16, ignore_vals = [0], force_blocksize=None):
    global glblhistogram_dtype
    global zonal_bands

    zonal_start_time = time.perf_counter()

    clear_zonal_stats()

    input_files = {
            'in': input_file,
            'zone': zone_file,
            }

    output_files = {}
    output_datatypes = {}
    output_band_counts = {}
    output_drivers = {}
    output_options = {}

    empty_output_parameters = (output_files, output_datatypes, output_band_counts, output_drivers, output_options)

    glblhistogram_dtype = histogram_dtype

    main(input_files, empty_output_parameters, func=zonal_stats_first_pass, force_blocksize=force_blocksize, args=None)

    ###START OF MPI CODE
    if is_mpi_enabled():
        #workers must send stats to master, master will collect all stats and merge them
        from mpi4py import MPI
        mpi_world = MPI.COMM_WORLD
        mpi_size = mpi_world.size
        mpi_world.barrier()
        if is_mpi_master() is False:
            #We are a worker, send all zonal stats to master
            mpi_world.send(zonal_bands, dest=0)
        else:
            print('Receiving zonal bands from workers and merging...')
            #We are the master, we must receive the zonal stats from each worker
            for n_recvd in range(1,mpi_size): #does not include master
                status = MPI.Status()
                worker_zonal_bands = mpi_world.recv(source = MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                requesting_rank = status.Get_source()
                #print('Received zonal bands from',requesting_rank,'...')
                #merge worker zonal stats with ours
                merge_zonal_bands(worker_zonal_bands)
        mpi_world.barrier()
    ###END OF MPI CODE

    if is_mpi_master():
        print('Calculating stats...')
        for band in zonal_bands.keys():
            zonal_counts = zonal_bands[band].zonal_counts
            for zone in zonal_counts.keys():
                zonal_count = zonal_counts[zone]
                (minimum,median,maximum) = get_min_median_max(zonal_count.histogram, ignore_vals = ignore_vals)
                (diversity,majority) = get_diversity_majority(zonal_count.histogram, ignore_vals = ignore_vals)
                mean = (zonal_count.totalsum / zonal_count.count) if zonal_count.count != 0 else 0.0
                #if (zonal_count.count >= 3):
                #    print('Band ',str(band),' (Zone: ',str(zone),')','SUM:',zonal_count.totalsum,'COUNT:',zonal_count.count,'HIST.keys.count:',len(zonal_count.histogram.keys()),'minimum:',minimum,'median:',median,'mean:',mean,'maximum:',maximum)
                del zonal_count

                zonal_bands[band].zonal_stats[zone] = ZonalStats()
                zonal_bands[band].zonal_stats[zone].minimum = minimum
                zonal_bands[band].zonal_stats[zone].maximum = maximum
                zonal_bands[band].zonal_stats[zone].median = median
                zonal_bands[band].zonal_stats[zone].mean = mean
                zonal_bands[band].zonal_stats[zone].diversity = diversity
                zonal_bands[band].zonal_stats[zone].majority = majority

            del zonal_counts

    ###START OF MPI CODE
    if is_mpi_enabled():
        #master sends all workers the updated zonal_stats
        from mpi4py import MPI
        mpi_world = MPI.COMM_WORLD
        mpi_size = mpi_world.size
        mpi_world.barrier()
        if is_mpi_master() is False:
            #We are a worker, receive zonal stats from master
            zonal_bands = mpi_world.recv(source=0)
        else:
            print('Sending zonal bands to workers...')
            #We are the master, send all workers the zonal_stats
            for n_recvd in range(1,mpi_size): #does not include master
                mpi_world.send(zonal_bands, dest=n_recvd)
                #print('Sent zonal bands to',n_recvd,'...')
        mpi_world.barrier()
    ###END OF MPI CODE

    if ('stddev' in stats):
        main(input_files, empty_output_parameters, func=zonal_stats_stddev, force_blocksize=force_blocksize, args=None)
        ###START OF MPI CODE
        if is_mpi_enabled():
            #workers must send stats to master, master will collect all stats and merge them
            from mpi4py import MPI
            mpi_world = MPI.COMM_WORLD
            mpi_size = mpi_world.size
            mpi_world.barrier()
            if is_mpi_master() is False:
                #We are a worker, send all zonal stats to master
                mpi_world.send(zonal_bands, dest=0)
            else:
                print('Receiving zonal bands from workers...')
                #We are the master, we must receive the zonal stats from each worker
                for n_recvd in range(1,mpi_size): #does not include master
                    status = MPI.Status()
                    worker_zonal_bands = mpi_world.recv(source = MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                    requesting_rank = status.Get_source()
                    #print('Received zonal bands from',requesting_rank,'...')
                    #merge worker zonal stats with ours
                    merge_zonal_bands_sqsum(worker_zonal_bands)
            mpi_world.barrier()
        ###END OF MPI CODE

        if is_mpi_master():
            print('Calculating std. dev....')
            for band in zonal_bands.keys():
                zonal_counts = zonal_bands[band].zonal_counts
                for zone in zonal_counts.keys():
                    zonal_count = zonal_counts[zone]
                    zonal_stat = zonal_bands[band].zonal_stats[zone]
                    zonal_stat.stddev = math.sqrt(zonal_count.sqsum / (zonal_count.count - 1)) if zonal_count.count > 1 else 0.0
                   #print('Band ',str(band),' (Zone: ',str(zone),')','SUM:',zonal_count.totalsum,'COUNT:',zonal_count.count,'HIST.keys.count:',len(zonal_count.histogram.keys()),'minimum:',zonal_stat.minimum,'median:',zonal_stat.median,'mean:',zonal_stat.mean,'maximum:',zonal_stat.maximum,'stddev:',zonal_stat.stddev)

        ###START OF MPI CODE
        if is_mpi_enabled():
            #master sends all workers the updated zonal_stats
            from mpi4py import MPI
            mpi_world = MPI.COMM_WORLD
            mpi_size = mpi_world.size
            mpi_world.barrier()
            if is_mpi_master() is False:
                #We are a worker, receive zonal stats from master
                zonal_bands = mpi_world.recv(source=0)
            else:
                print('Sending zonal bands to workers...')
                #We are the master, send all workers the zonal_stats
                for n_recvd in range(1,mpi_size): #does not include master
                    mpi_world.send(zonal_bands, dest=n_recvd)
                    #print('Sent zonal bands to',n_recvd,'...')
            mpi_world.barrier()
        ###END OF MPI CODE

    output_zonal_stats = {}
    for band in zonal_bands.keys():
        output_zonal_stats[input_name+'('+str(band)+')'] = zonal_bands[band].zonal_stats
        if is_mpi_master():
            print('Band ',str(band),': ','Zonal_Stats Count:',len(zonal_bands[band].zonal_stats.keys()))

    if (len(output_parameters) > 0):
        args = {}
        args['stats'] = stats
        main(input_files, output_parameters, func=zonal_stats_output, force_blocksize=force_blocksize, args=args)

    zonal_stop_time = time.perf_counter()
    zonal_total_time = zonal_stop_time - zonal_start_time
    if is_mpi_master():
        print('Zonal Stats total time:',zonal_total_time,'secs',timedelta(seconds=zonal_total_time))

    return output_zonal_stats
