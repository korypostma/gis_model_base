import os
import sys
import traceback
import gc
import math
import time
from datetime import timedelta #To show duration of runs

from osgeo import gdal
import numpy as np
import tqdm #for progress bar

###############################################################################
###############################################################################
#Valid output_datatypes: GDT_Byte, GDT_UInt16, GDT_Int16, GDT_UInt32,
#                        GDT_Int32, GDT_Float32, GDT_Float64, GDT_CInt16,
#                        GDT_CInt32, GDT_CFloat32, GDT_CFloat64
###############################################################################

class Block:
    def __init__(self, x, cols, y, rows):
        self.x = x
        self.cols = cols
        self.y = y
        self.rows = rows

def implant_key_band(key, band_num):
    return (key+'('+str(band_num)+')')

def extract_key_band(key):
    try:
        (dskey, band_num) = key.split('(', 1)
    except ValueError:
        print('error getting key and band for:', key)
        raise SystemExit(1)
    try:
        (band_num, junk) = band_num.split(')', 1)
    except ValueError:
        print('error getting band_num and junk for:', band_num)
        raise SystemExit(1)
    return (dskey, int(band_num))

def master_process_blocks(ds_in, ds_out, in_extent, force_blocksize):
    """
    Used to give blocks to worker MPI processes to produce ds_out output for the master MPI process
    """
    from mpi4py import MPI
    mpi_world = MPI.COMM_WORLD
    mpi_size = mpi_world.size
    outputs_time = 0.0

    SIZE_X = int(in_extent[4])
    SIZE_Y = int(in_extent[5])
    BLK_SZ_X = 0
    BLK_SZ_Y = 0

    bands = {}
    #total_bands = 0
    #Save off each of the raster bands in each dataset
    for key,ds in ds_in.items():
        BAND_COUNT = ds.RasterCount
        #total_bands += BAND_COUNT
        for i in range(BAND_COUNT):
            band = ds.GetRasterBand(i+1)
            (BLK_SZ_X, BLK_SZ_Y) = band.GetBlockSize()
            bands[implant_key_band(key, i+1)] = band

    #Increase the block size to read in more data for processing
    if force_blocksize is not None:
        BLK_SZ_X = force_blocksize[0]
        BLK_SZ_Y = force_blocksize[1]

    BLK_SZ_X = min(BLK_SZ_X, SIZE_X)
    BLK_SZ_Y = min(BLK_SZ_Y, SIZE_Y)

    #Calculate offsets for each band
    offsets = get_extent_offsets(in_extent, ds_in)
    #Go through each block and process it
    total_blocks = (math.ceil(SIZE_X / BLK_SZ_X) * math.ceil(SIZE_Y / BLK_SZ_Y))
    list_of_blocks = []
    print('Raster Size: {}x{}\nBlock Size: {}x{}'.format(SIZE_X,SIZE_Y,BLK_SZ_X,BLK_SZ_Y))
    for y in range(0, SIZE_Y, BLK_SZ_Y):
        rows = min(BLK_SZ_Y, SIZE_Y - y)
        for x in range(0, SIZE_X, BLK_SZ_X):
            cols = min(BLK_SZ_X, SIZE_X - x)
            list_of_blocks.append(Block(x,cols,y,rows))

    n_blocks = len(list_of_blocks)
    print('n_blocks=',n_blocks)
    print('total_blocks=',total_blocks)

    if (mpi_size <= 1):
        print('MPI reports no workers available to process blocks, exiting...')
        raise SystemExit(1)

    n_blocks_sent = 0
    n_finished = 0
    for i_worker in range(1, mpi_size):
        if i_worker > n_blocks:
            b_continue = False
            mpi_world.send(b_continue, dest=i_worker)
        else:
            b_continue = True
            mpi_world.send(b_continue, dest=i_worker)
            block = list_of_blocks[n_blocks_sent]
            mpi_world.send(block, dest=i_worker)
            n_blocks_sent += 1

    pbar = tqdm.tqdm(total=total_blocks)
    while n_finished < n_blocks:
        status = MPI.Status()
        results = mpi_world.recv(source = MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        requesting_rank = status.Get_source()
        #print('Received data from',requesting_rank,'...')
        #output the data to the correct ds_out
        #outputs start
        block_start_time = time.perf_counter()
        block = None
        try:
            block = results['BLOCK']
        except KeyError:
            print('ERROR: results does not contain a BLOCK key, throwing these results away')
            print('results keys found:',results.keys())
            print('ERROR: Unexpected error:')
            print(sys.exc_info())
        if block is not None:
            for key,out_data in results.items():
                if key == 'BLOCK':
                    continue
                #Convert key back to original key and raster number
                (dskey, band_num) = extract_key_band(key)
                #has_valid_data = np.any(out_data == 1)
                #print('key="{}" dskey="{}" band_num="{}" has_valid_data="{}"'.format(key,dskey,band_num,has_valid_data))
                #print(type(out_data))
                #print(out_data)
                try:
                    ds_out[dskey].GetRasterBand(band_num).WriteArray(out_data, block.x, block.y)
                except KeyError:
                    print('ERROR: model writing to non-existant output:',dskey)
                    print('ds_out keys found:',ds_out.keys())
                    print('ERROR: Unexpected error:')
                    print(sys.exc_info())
                    sys.stdout.flush()
                    raise SystemExit(1)
                except ValueError:
                    print('ERROR: Unexpected error:')
                    print(sys.exc_info())
                    print('key:',key,'out_data.shape:',out_data.shape)
                    print('Trace:')
                    print(traceback.print_exc())
                    sys.stdout.flush()
                    raise SystemExit(1)
        #outputs end
        block_stop_time = time.perf_counter()
        outputs_time += block_stop_time - block_start_time
        del results
        n_finished += 1
        pbar.update()

        #Time to send new block if there are blocks remaining to be sent
        if n_blocks_sent == n_blocks:
            block = None
        else:
            block = list_of_blocks[n_blocks_sent]

        if block is None:
            b_continue = False
            mpi_world.send(b_continue, dest=requesting_rank)
        else:
            b_continue = True
            mpi_world.send(b_continue, dest=requesting_rank)
            mpi_world.send(block, dest=requesting_rank)
            n_blocks_sent += 1
    #print out timers
    print('Outputs Write time:',outputs_time,'secs',timedelta(seconds=outputs_time))
    return True

def worker_process_blocks(ds_in, in_extent, func, focal_size, args=None):
    from mpi4py import MPI
    mpi_world = MPI.COMM_WORLD
    bands = {}
    #Save off each of the raster bands in each dataset
    for key,ds in ds_in.items():
        BAND_COUNT = ds.RasterCount
        for i in range(BAND_COUNT):
            band = ds.GetRasterBand(i+1)
            bands[implant_key_band(key, i+1)] = band

    #Calculate offsets for each band
    offsets = get_extent_offsets(in_extent, ds_in, verbose=False)

    b_continue = mpi_world.recv(source=0)
    while b_continue:
        block = mpi_world.recv(source=0)
        results = worker_process_block(block, bands, offsets, func, focal_size, args)
        results['BLOCK'] = block
        try:
            mpi_world.send(results, dest=0)
        except OverflowError:
            print('ERROR: Unexpected error:')
            print(sys.exc_info())
            print('results:',results)
            print('Trace:')
            print(traceback.print_exc())
            sys.stdout.flush()
            raise SystemExit(1)
        b_continue = mpi_world.recv(source=0)
        del results
    return True

def get_data(key, dskey, block, bands, offsets, focal_size, no_data):
    #TODO this code has been modified to support custom extents (max, AOI, custom) and may not work in all cases, we will have to fix those cases where it fails
    this_offset = offsets[dskey]
    if (focal_size < 1):
        focal_size = 1

    x_size = bands[key].XSize
    y_size = bands[key].YSize
    focal_odd = focal_size % 2
    focal_even = (focal_size+1) % 2
    top_left_offset = (focal_size-focal_even)//2
    bottom_right_offset = (focal_size-focal_odd)//2
    #NOTE the old code only worked for min_extent, for supporting both we need to always output the expected number of cols and rows given from the block
    cols = block.cols
    rows = block.rows
    if (no_data is None):
        no_data = 0.0
    #data = np.full(shape=(cols+focal_size-1, rows+focal_size-1), fill_value=no_data)
    data = np.full(shape=(rows+top_left_offset+bottom_right_offset, cols+top_left_offset+bottom_right_offset), fill_value=no_data)

    #Check left-hand side of x/cols
    left_x = offsets[dskey][0] + block.x - top_left_offset
    if (left_x < 0):
        x_offset = 0 - left_x
    else:
        x_offset = 0
    x = left_x + x_offset

    #Check right-hand side of x/cols
    right_x = offsets[dskey][0] + block.x + cols + bottom_right_offset
    if (right_x > x_size):
        cols_offset = right_x - x_size
    else:
        cols_offset = 0
    cols = top_left_offset + cols + bottom_right_offset - cols_offset - x_offset

    #Check top side of y/rows
    top_y = offsets[dskey][1] + block.y - top_left_offset
    if (top_y < 0):
        y_offset = 0 - top_y
    else:
        y_offset = 0
    y = top_y + y_offset

    #Check bottom side of y/rows
    bottom_y = offsets[dskey][1] + block.y + rows + bottom_right_offset
    if (bottom_y > y_size):
        rows_offset = bottom_y - y_size
    else:
        rows_offset = 0
    rows = top_left_offset + rows + bottom_right_offset - rows_offset - y_offset

    if (cols < 0) or (rows < 0):
        #return no_data filled data if both cols and rows is filled with zeroes
        return data

    #print('key=',key,'x=',x,'y=',y,'cols=',cols,'rows=',rows,'x_offset=',x_offset,'y_offset=',y_offset)
    #print('y=',int(y_offset),int(y_offset+rows),'x=',int(x_offset),int(x_offset+cols))
    #data[int(y_offset):int(y_offset+rows),int(x_offset):int(x_offset+cols)] = (bands[key].ReadAsArray(int(x), int(y), int(cols), int(rows))).astype(float)
    data[int(y_offset):int(y_offset+rows),int(x_offset):int(x_offset+cols)] = bands[key].ReadAsArray(int(x), int(y), int(cols), int(rows))

    return data

def worker_process_block(block, bands, offsets, func, focal_size, args=None):
    #Read in the band to operate func on and setup mask
    #in_data = np.zeros((block.rows, block.cols, BAND_COUNT), dtype=float)
    in_data = {}
    no_data = {}

    #inputs start
    block_start_time = time.perf_counter()
    for key,band in bands.items():
        (dskey, band_num) = extract_key_band(key)
        no_data[key] = bands[key].GetNoDataValue()
        in_data[key] = get_data(key, dskey, block, bands, offsets, focal_size, no_data[key])
    #inputs end
    block_stop_time = time.perf_counter()
    inputs_time = block_stop_time - block_start_time
    #print('Inputs Read time:',inputs_time,'secs',timedelta(seconds=inputs_time))

    #Run the function for all of the bands with the given arguments
    #model start
    block_start_time = time.perf_counter()
    has_results = False
    while(has_results == False):
        try:
            if args is None:
                results = func(in_data, no_data)
            else:
                results = func(in_data, no_data, **args)
            has_results = True
        except MemoryError as error:
            #Try to collect garbage to continue processing
            gc.collect()
            has_results = False
            print('{}: {}'.format(type(error).__name__, error))
    #model end
    block_stop_time = time.perf_counter()
    model_time = block_stop_time - block_start_time
    #print('Model time:',model_time,'secs',timedelta(seconds=model_time))
    del in_data
    del no_data

    return results

def non_mpi_process_blocks(ds_in, ds_out, in_extent, focal_size, force_blocksize, func, args=None):
    """
    Used to call func on the bands to produce ds_out output
    """
    inputs_time = 0.0
    model_time = 0.0
    outputs_time = 0.0

    #This is here for testing purposes and should normally always be true
    allow_processing = True
    SIZE_X = int(in_extent[4])
    SIZE_Y = int(in_extent[5])
    BLK_SZ_X = 0
    BLK_SZ_Y = 0

    bands = {}
    total_bands = 0
    #Save off each of the raster bands in each dataset
    for key,ds in ds_in.items():
        BAND_COUNT = ds.RasterCount
        total_bands += BAND_COUNT
        for i in range(BAND_COUNT):
            band = ds.GetRasterBand(i+1)
            (BLK_SZ_X, BLK_SZ_Y) = band.GetBlockSize()
            bands[implant_key_band(key, i+1)] = band

    #Increase the block size to read in more data for processing
    if force_blocksize is not None:
        BLK_SZ_X = force_blocksize[0]
        BLK_SZ_Y = force_blocksize[1]

    BLK_SZ_X = min(BLK_SZ_X, SIZE_X)
    BLK_SZ_Y = min(BLK_SZ_Y, SIZE_Y)

    #Calculate offsets for each band
    offsets = get_extent_offsets(in_extent, ds_in)
    #Go through each block and process it
    total_blocks = (math.ceil(SIZE_X / BLK_SZ_X) * math.ceil(SIZE_Y / BLK_SZ_Y))
    print('Raster Size: {}x{}\nBlock Size: {}x{}'.format(SIZE_X,SIZE_Y,BLK_SZ_X,BLK_SZ_Y))
    pbar = tqdm.tqdm(total=total_blocks)
    for y in range(0, SIZE_Y, BLK_SZ_Y):
        rows = min(BLK_SZ_Y, SIZE_Y - y)
        for x in range(0, SIZE_X, BLK_SZ_X):
            cols = min(BLK_SZ_X, SIZE_X - x)

            if allow_processing:
                #Read in the band to operate func on and setup mask
                #in_data = np.zeros((rows, cols, BAND_COUNT), dtype=float)
                in_data = {}
                no_data = {}

                #inputs start
                block_start_time = time.perf_counter()
                block = Block(x,cols,y,rows)
                for key,band in bands.items():
                    (dskey, band_num) = extract_key_band(key)
                    no_data[key] = bands[key].GetNoDataValue()
                    in_data[key] = get_data(key, dskey, block, bands, offsets, focal_size, no_data[key])
                #inputs end
                del block
                block_stop_time = time.perf_counter()
                inputs_time += block_stop_time - block_start_time

                #Run the function for all of the bands with the given arguments
                #model start
                block_start_time = time.perf_counter()
                has_results = False
                while(has_results == False):
                    try:
                        if args is None:
                            results = func(in_data, no_data)
                        else:
                            results = func(in_data, no_data, **args)
                        has_results = True
                    except MemoryError as error:
                        #Try to collect garbage to continue processing
                        gc.collect()
                        has_results = False
                        print('{}: {}'.format(type(error).__name__, error))
                #model end
                block_stop_time = time.perf_counter()
                model_time += block_stop_time - block_start_time

                #print('\n')
                #outputs start
                block_start_time = time.perf_counter()
                for key,out_data in results.items():
                    #Convert key back to original key and raster number
                    (dskey, band_num) = extract_key_band(key)
                    #has_valid_data = np.any(out_data == 1)
                    #print('key="{}" dskey="{}" band_num="{}" has_valid_data="{}"'.format(key,dskey,band_num,has_valid_data))
                    #print(type(out_data))
                    #print(out_data)
                    ds_out[dskey].GetRasterBand(band_num).WriteArray(out_data, x, y)
                #outputs end
                block_stop_time = time.perf_counter()
                outputs_time += block_stop_time - block_start_time
                del in_data
                del no_data
                del results
            pbar.update()
    #print out timers
    print('Inputs Read time:',inputs_time,'secs',timedelta(seconds=inputs_time))
    print('Model time:',model_time,'secs',timedelta(seconds=model_time))
    print('Outputs Write time:',outputs_time,'secs',timedelta(seconds=outputs_time))
    return True

def get_max_extent(ds_in, verbose=True):
    base_ds = ds_in[next(iter(ds_in))]
    geo = base_ds.GetGeoTransform()
    x_min = geo[0]
    x_res = geo[1]
    y_min = geo[3]
    y_res = geo[5]
    x_size = base_ds.RasterXSize
    y_size = base_ds.RasterYSize
    x_max = x_min + x_res * x_size
    y_max = y_min + y_res * y_size
    for key,ds in ds_in.items():
        geo = ds.GetGeoTransform()
        size = (ds.RasterXSize, ds.RasterYSize)
        geo_max = (geo[0] + geo[1] * size[0], geo[3] + geo[5] * size[1])
        if verbose:
            print("{}: geo:{} size:{},{} geo_max:{}".format(key,geo,size[0],size[1],geo_max))
        if ((abs(x_res - geo[1]) > 1e-4) or (abs(y_res - geo[5]) > 1e-4)):
            print("Both x and y resolution must be the same, failed on:",ds.GetDescription())
            sys.exit(2)
        if (x_res >= 0.0):
            x_min = min(x_min, geo[0])
            x_max = max(x_max, geo_max[0])
        else:
            x_min = max(x_min, geo[0])
            x_max = min(x_max, geo_max[0])
        if (y_res >= 0.0):
            y_min = min(y_min, geo[3])
            y_max = max(y_max, geo_max[1])
        else:
            y_min = max(y_min, geo[3])
            y_max = min(y_max, geo_max[1])

    max_extent = (x_min, y_min, x_max, y_max, abs((x_max - x_min) / x_res), abs((y_max - y_min) / y_res))
    if verbose:
        print("Max Extent(xm,ym,xM,yM,sz_x,sz_y):",max_extent)
    return max_extent

def get_min_extent(ds_in, verbose=True):
    base_ds = ds_in[next(iter(ds_in))]
    geo = base_ds.GetGeoTransform()
    x_min = geo[0]
    x_res = geo[1]
    y_min = geo[3]
    y_res = geo[5]
    x_size = base_ds.RasterXSize
    y_size = base_ds.RasterYSize
    x_max = x_min + x_res * x_size
    y_max = y_min + y_res * y_size
    for key,ds in ds_in.items():
        geo = ds.GetGeoTransform()
        size = (ds.RasterXSize, ds.RasterYSize)
        geo_max = (geo[0] + geo[1] * size[0], geo[3] + geo[5] * size[1])
        if verbose:
            print('{}: geo:{} size:{},{} geo_max:{}'.format(key,geo,size[0],size[1],geo_max))
        if ((abs(x_res - geo[1]) > 1e-4) or (abs(y_res - geo[5]) > 1e-4)):
            print('Both x and y resolution must be the same, failed on:',ds.GetDescription())
            sys.exit(2)
        if (x_min < geo[0] and geo[0] < x_max):
            x_min = geo[0]
        if (x_min < geo_max[0] and geo_max[0] < x_max):
            x_max = geo_max[0]
        #The following fails for y, it never gets set properly
        #if (y_min < geo[3] and geo[3] < y_max):
        #    y_min = geo[3]
        #if (y_min < geo_max[1] and geo_max[1] < y_max):
        #    y_max = geo_max[1]
        if (y_min > geo[3] and geo[3] > y_max):
            y_min = geo[3]
        if (y_min > geo_max[1] and geo_max[1] > y_max):
            y_max = geo_max[1]

    min_extent = (x_min, y_min, x_max, y_max, abs((x_max - x_min) / x_res), abs((y_max - y_min) / y_res))
    if verbose:
        print('Min Extent(xm,ym,xM,yM,sz_x,sz_y):',min_extent)
    return min_extent

def round_extent(extent):
    return (round(extent[0]),round(extent[1]),round(extent[2]),round(extent[3]),round(extent[4]),round(extent[5]))

def get_extent_from_parameter(extent, ds_in, verbose=True):
    if verbose:
        print('Loading extent from',extent)
    if type(extent) is tuple:
        if (len(extent) == 6):
            return extent
        else:
            print('ERROR: extent parameter is a tuple but not of length 6, len=',len(extent),'extent=',extent)
            raise SystemExit(3)
    if type(extent) is str:
        if (extent == 'min'):
            return get_min_extent(ds_in, verbose=verbose)
        if (extent == 'max'):
            return get_max_extent(ds_in, verbose=verbose)
        ds = load_ds(extent, verbose=verbose)
        if ds is None:
            print('ERROR: unable to load raster of extent parameter: extent=',extent)
            raise SystemExit(3)
        return get_min_extent({'aoi':ds}, verbose=verbose)
    print('ERROR: extent parameter is not one of min, max, tuple, or str: extent=',extent)
    raise SystemExit(3)

def get_extent_offsets(in_extent, ds_in, verbose=True):
    (x_min, y_min, x_max, y_max, x_size, y_size) = in_extent
    offsets = {}
    i = 0
    for key,ds in ds_in.items():
        geo = ds.GetGeoTransform()
        #NOTE this only works when using get_min_extent, so we use the following code instead to support both min/max
        #offset = (abs((geo[0] - x_min) / geo[1]), abs((geo[3] - y_min) / geo[5]))
        offset = (round((x_min - geo[0]) / geo[1]), round((y_min - geo[3]) / geo[5]))
        #if verbose:
        #    print('============================')
        #    print('key=',key)
        #    print('in_extent=',in_extent)
        #    print('geo=',geo)
        #    print('offset=',offset)
        offsets[key] = tuple(offset)
        if verbose:
            print('{}: offsets:{}'.format(key,offsets[key]))
        i+=1
    return offsets

def load_ds(filepath, readonly=True, verbose=True):
    """
    Load a raster dataset
    """
    if not readonly:
        if verbose:
            print('Loading',filepath,'as writeable')
        return gdal.Open(filepath, gdal.GA_Update)

    if verbose:
        print('Loading',filepath,'as READ-only')
    return gdal.Open(filepath, gdal.GA_ReadOnly)

def create(path, rows, cols, affine, datatype, proj, bands, driver='HFA', options=['COMPRESS=YES']):
    """
    Create a GeoTif and return the data set to work with.
    If the file exists at the given path, this will attempt to remove it.
    """
    ds = (gdal
          .GetDriverByName(driver)
          .Create(path, cols, rows, bands, datatype, options=options))

    ds.SetGeoTransform(affine)
    ds.SetProjection(proj)

    return ds

def main(input_files, output_parameters, func, extent='min', focal_size=1, force_non_mpi=False, force_blocksize=None, gdal_cachemax=1536*1024*1024, args=None):
    """
    input_files:
    output_parameters:
    func: model function to use
    extent: one of 'min','max',tuple (in this format Top Left x, y, Bottom Right x, y, Raster Size x, y), string pointing to raster to use as AOI
    focal_size: when running focal stats what is the size of the focal kernel, this will return focal_size//2 extra pixels around the blocksize
    force_non_mpi: set to True to run in single-threaded mode regardless of availability of MPI
    force_blocksize: a tuple of size 2 used to set the blocksize for all blocks, edge blocks may still be smaller
    gdal_cachemax: parameter used to configure how much cache gdal should use
    args: keyword args (kwargs) to pass down to the model function set in the func parameter
    """
    try:
        from mpi4py import MPI
        mpi_world = MPI.COMM_WORLD
        mpi_size = mpi_world.size
    except ModuleNotFoundError:
        print('MPI is not installed, defaulting to non-MPI mode')
        mpi_size = 1
    except ImportError:
        print('MPI is not installed, defaulting to non-MPI mode')
        mpi_size = 1

    if (mpi_size <= 1) or (force_non_mpi is True):
        non_mpi_main(input_files=input_files, output_parameters=output_parameters, func=func, extent=extent, focal_size=focal_size, force_blocksize=force_blocksize, gdal_cachemax=gdal_cachemax, args=args)
    else:
        mpi_main(input_files=input_files, output_parameters=output_parameters, func=func, extent=extent, focal_size=focal_size, force_blocksize=force_blocksize, gdal_cachemax=gdal_cachemax, args=args)

def non_mpi_main(input_files, output_parameters, func, extent, focal_size, force_blocksize=None, gdal_cachemax=1536*1024*1024, args=None):
    start_time = time.perf_counter()
    (output_files, output_datatypes, output_band_counts, output_drivers, output_options) = output_parameters
    gdal.SetCacheMax(gdal_cachemax) #Default 4,096 MB
    ds_in = {}
    for key,in_file in input_files.items():
        #print('Loading',in_file)
        ds_in[key] = load_ds(in_file, verbose=True)
        if ds_in[key] is None:
            print('Image cannot be opened:',in_file)
            raise SystemExit(1)
    ds = ds_in[next(iter(ds_in))]
    (BLK_SZ_X, BLK_SZ_Y) = ds.GetRasterBand(1).GetBlockSize()
    SIZE_X = ds.RasterXSize
    SIZE_Y = ds.RasterYSize
    GEO = ds.GetGeoTransform()
    X_RES = GEO[1]
    Y_RES = GEO[5]
    PROJ = ds.GetProjection()
    #print('ds[0]:',BLK_SZ_X,BLK_SZ_Y,SIZE_X,SIZE_Y,GEO,X_RES,Y_RES,PROJ)
    error = 0
    print('================================================================================')
    for key,ds in ds_in.items():
        this_BAND_COUNT = ds.RasterCount
        (this_BLK_SZ_X, this_BLK_SZ_Y) = ds.GetRasterBand(1).GetBlockSize()
        this_SIZE_X = ds.RasterXSize
        this_SIZE_Y = ds.RasterYSize
        this_geo = ds.GetGeoTransform()
        this_proj = ds.GetProjection()
        this_x_res = this_geo[1]
        this_y_res = this_geo[5]
        #print('IN :',key,'=',ds.GetDescription(),'size: ('+str(this_SIZE_X)+','+str(this_SIZE_Y)+')  Block size: ('+str(this_BLK_SZ_X)+','+str(this_BLK_SZ_Y)+')','geo:',this_geo,'proj:',this_proj)
        print('IN :',key,'=',ds.GetDescription(),'bands:',this_BAND_COUNT,'size: ('+str(this_SIZE_X)+','+str(this_SIZE_Y)+')  Block size: ('+str(this_BLK_SZ_X)+','+str(this_BLK_SZ_Y)+')','geo:',this_geo)
        if (abs(this_x_res - X_RES) > 0.1):
            error = 1
            print('ERROR: {} GeoTransform x-res {} does not match {} as expected'.format(key, this_x_res, X_RES))
        if (abs(this_y_res - Y_RES) > 0.1):
            error = 1
            print('ERROR: {} GeoTransform y-res {} does not match {} as expected'.format(key, this_y_res, Y_RES))
        #if (this_SIZE_X != SIZE_X):
        #    error = 1
        #    print('ERROR: {} RasterXSize {} does not match {} as expected'.format(key, ds.RasterXSize, SIZE_X))
        #if (this_SIZE_Y != SIZE_Y):
        #    error = 1
        #    print('ERROR: {} RasterYSize {} does not match {} as expected'.format(key, ds.RasterYSize, SIZE_Y))
        #if (this_geo != GEO):
        #    error = 1
        #    print('ERROR: {} GeoTransform {} does not match {} as expected'.format(key, this_geo, GEO))
        #if (this_proj != PROJ):
        #    error = 1
        #    print('ERROR: {} Projection {} does not match {} as expected'.format(key, this_proj, PROJ))
    if error == 1:
        print('ERROR: Errors encountered in reading input files, unable to continue')
        raise SystemExit(1)

    print('================================================================================')

    in_extent = get_extent_from_parameter(extent, ds_in)
    print('in_extent=',in_extent)
    print('rounded in_extent=',round_extent(in_extent))

    #Get geo transform and set the min extent values
    GEO = (in_extent[0], GEO[1], GEO[2], in_extent[1], GEO[4], GEO[5])
    SIZE_X = int(in_extent[4])
    SIZE_Y = int(in_extent[5])
    print('Output GEO:',GEO,' Output Size:',(SIZE_X,SIZE_Y))

    print('================================================================================')
    ds_out = {}
    for key,out_file in output_files.items():
        ds_out[key] = create(out_file, SIZE_Y, SIZE_X, GEO, output_datatypes[key], PROJ, output_band_counts[key], driver=output_drivers[key], options=output_options[key])
        if ds_out[key] is None:
            print('Image cannot be created:',out_file)
            raise SystemExit(1)
        this_SIZE_X = ds_out[key].RasterXSize
        this_SIZE_Y = ds_out[key].RasterYSize
        (this_BLK_SZ_X, this_BLK_SZ_Y) = ds_out[key].GetRasterBand(1).GetBlockSize()
        this_geo = ds_out[key].GetGeoTransform()
        #this_proj = ds_out[key].GetProjection()
        #print('OUT:',key,'=',ds_out[key].GetDescription(),'size: ('+str(this_SIZE_X)+','+str(this_SIZE_Y)+')  Block size: ('+str(this_BLK_SZ_X)+','+str(this_BLK_SZ_Y)+')','geo:',this_geo,'proj:',this_proj)
        print('OUT:',key,'=',ds_out[key].GetDescription(),'size: ('+str(this_SIZE_X)+','+str(this_SIZE_Y)+')  Block size: ('+str(this_BLK_SZ_X)+','+str(this_BLK_SZ_Y)+')','geo:',this_geo)

    print('================================================================================')

    try:
        non_mpi_process_blocks(ds_in, ds_out, in_extent, focal_size=focal_size, force_blocksize=force_blocksize, func=func, args=args)
    except:
        print('ERROR: Unexpected error:')
        print(sys.exc_info())
        print('Trace:')
        print(traceback.print_exc())
        print('Removing outputs due to error:')
        #NOTE: there was an error raised so delete all outputs
        for key,out_file in output_files.items():
            try:
                os.remove(out_file)
                print('Removed',key,':',out_file)
            except:
                print('ERROR unable to remove',key,':',out_file)
                pass
        print('Done removing outputs due to error, exiting with exit code 1')
        sys.stdout.flush()
        raise SystemExit(1)
    del ds_in
    for key,out_file in output_files.items():
        del ds_out[key]
    stop_time = time.perf_counter()
    total_time = stop_time - start_time
    print('================================================================================')
    print('Total time:',total_time,'secs',timedelta(seconds=total_time))

def mpi_main(input_files, output_parameters, func, extent, focal_size, force_blocksize=None, gdal_cachemax=1536*1024*1024, args=None):
    from mpi4py import MPI
    start_time = time.perf_counter()

    mpi_world = MPI.COMM_WORLD
    mpi_world.barrier()
    mpi_master = (mpi_world.rank == 0)

    (output_files, output_datatypes, output_band_counts, output_drivers, output_options) = output_parameters
    gdal.SetCacheMax(gdal_cachemax) #Default 4,096 MB
    ds_in = {}
    for key,in_file in input_files.items():
        #if mpi_master:
        #    print('Loading',in_file)
        ds_in[key] = load_ds(in_file, verbose=mpi_master)
        if ds_in[key] is None:
            print('Image cannot be opened:',in_file)
            raise SystemExit(1)
    ds = ds_in[next(iter(ds_in))]
    (BLK_SZ_X, BLK_SZ_Y) = ds.GetRasterBand(1).GetBlockSize()
    SIZE_X = ds.RasterXSize
    SIZE_Y = ds.RasterYSize
    GEO = ds.GetGeoTransform()
    X_RES = GEO[1]
    Y_RES = GEO[5]
    PROJ = ds.GetProjection()
    #print('ds[0]:',BLK_SZ_X,BLK_SZ_Y,SIZE_X,SIZE_Y,GEO,X_RES,Y_RES,PROJ)
    error = 0

    if mpi_master:
        print('================================================================================')

    for key,ds in ds_in.items():
        this_BAND_COUNT = ds.RasterCount
        (this_BLK_SZ_X, this_BLK_SZ_Y) = ds.GetRasterBand(1).GetBlockSize()
        this_SIZE_X = ds.RasterXSize
        this_SIZE_Y = ds.RasterYSize
        this_geo = ds.GetGeoTransform()
        this_proj = ds.GetProjection()
        this_x_res = this_geo[1]
        this_y_res = this_geo[5]
        if mpi_master:
            #print('IN :',key,'=',ds.GetDescription(),'size: ('+str(this_SIZE_X)+','+str(this_SIZE_Y)+')  Block size: ('+str(this_BLK_SZ_X)+','+str(this_BLK_SZ_Y)+')','geo:',this_geo,'proj:',this_proj)
            print('IN :',key,'=',ds.GetDescription(),'bands:',this_BAND_COUNT,'size: ('+str(this_SIZE_X)+','+str(this_SIZE_Y)+')  Block size: ('+str(this_BLK_SZ_X)+','+str(this_BLK_SZ_Y)+')','geo:',this_geo)
        if (abs(this_x_res - X_RES) > 0.1):
            error = 1
            print('ERROR: {} GeoTransform x-res {} does not match {} as expected'.format(key, this_x_res, X_RES))
        if (abs(this_y_res - Y_RES) > 0.1):
            error = 1
            print('ERROR: {} GeoTransform y-res {} does not match {} as expected'.format(key, this_y_res, Y_RES))
        #if (this_SIZE_X != SIZE_X):
        #    error = 1
        #    print('ERROR: {} RasterXSize {} does not match {} as expected'.format(key, ds.RasterXSize, SIZE_X))
        #if (this_SIZE_Y != SIZE_Y):
        #    error = 1
        #    print('ERROR: {} RasterYSize {} does not match {} as expected'.format(key, ds.RasterYSize, SIZE_Y))
        #if (this_geo != GEO):
        #    error = 1
        #    print('ERROR: {} GeoTransform {} does not match {} as expected'.format(key, this_geo, GEO))
        #if (this_proj != PROJ):
        #    error = 1
        #    print('ERROR: {} Projection {} does not match {} as expected'.format(key, this_proj, PROJ))
    if error == 1:
        print('ERROR: Errors encountered in reading input files, unable to continue')
        raise SystemExit(1)

    if mpi_master:
        print('================================================================================')

    in_extent = get_extent_from_parameter(extent, ds_in, verbose=mpi_master)

    if mpi_master:
        print('in_extent=',in_extent)
        print('rounded in_extent=',round_extent(in_extent))
        #Get geo transform and set the min extent values
        GEO = (in_extent[0], GEO[1], GEO[2], in_extent[1], GEO[4], GEO[5])
        SIZE_X = int(in_extent[4])
        SIZE_Y = int(in_extent[5])
        print('Output GEO:',GEO,' Output Size:',(SIZE_X,SIZE_Y))

        print('================================================================================')
        ds_out = {}
        for key,out_file in output_files.items():
            ds_out[key] = create(out_file, SIZE_Y, SIZE_X, GEO, output_datatypes[key], PROJ, output_band_counts[key], driver=output_drivers[key], options=output_options[key])
            if ds_out[key] is None:
                print('Image cannot be created:',out_file)
                raise SystemExit(1)
            this_SIZE_X = ds_out[key].RasterXSize
            this_SIZE_Y = ds_out[key].RasterYSize
            (this_BLK_SZ_X, this_BLK_SZ_Y) = ds_out[key].GetRasterBand(1).GetBlockSize()
            this_geo = ds_out[key].GetGeoTransform()
            #this_proj = ds_out[key].GetProjection()
            #print('OUT:',key,'=',ds_out[key].GetDescription(),'size: ('+str(this_SIZE_X)+','+str(this_SIZE_Y)+')  Block size: ('+str(this_BLK_SZ_X)+','+str(this_BLK_SZ_Y)+')','geo:',this_geo,'proj:',this_proj)
            print('OUT:',key,'=',ds_out[key].GetDescription(),'size: ('+str(this_SIZE_X)+','+str(this_SIZE_Y)+')  Block size: ('+str(this_BLK_SZ_X)+','+str(this_BLK_SZ_Y)+')','geo:',this_geo)

        print('================================================================================')

        try:
            master_process_blocks(ds_in, ds_out, in_extent, force_blocksize=force_blocksize)
        except:
            print('ERROR: Unexpected error:')
            print(sys.exc_info())
            print('Trace:')
            print(traceback.print_exc())
            print('Removing outputs due to error:')
            #NOTE: there was an error raised so delete all outputs
            for key,out_file in output_files.items():
                try:
                    os.remove(out_file)
                    print('Removed',key,':',out_file)
                except:
                    print('ERROR unable to remove',key,':',out_file)
                    pass
            print('Done removing outputs due to error, exiting with exit code 1')
            sys.stdout.flush()
            raise SystemExit(1)
        for key,out_file in output_files.items():
            del ds_out[key]
    else: #worker process, not the master
        try:
            worker_process_blocks(ds_in, in_extent, func=func, focal_size=focal_size, args=args)
        except:
            print('ERROR: Unexpected error:')
            print(sys.exc_info())
            print('Trace:')
            print(traceback.print_exc())
            sys.stdout.flush()
            raise SystemExit(1)
    for key,in_file in input_files.items():
        del ds_in[key]
    stop_time = time.perf_counter()
    total_time = stop_time - start_time
    if mpi_master:
        print('================================================================================')
        print('Total time:',total_time,'secs',timedelta(seconds=total_time))
    mpi_world.barrier()

