from model_base.model_base import *
from osgeo import gdal

import os
import sys

def model(in_data, no_data):
    out_data = {}

    out_data['out(1)'] = in_data['in_new(1)'] - in_data['in_old(1)']

    return out_data

if __name__ == '__main__':
    arg_count = len(sys.argv)-1
    if (arg_count == 3):
        in_file_1 = sys.argv[1]
        in_file_2 = sys.argv[2]
        out_file = sys.argv[3]
    else:
        print('Usage: '+sys.argv[0]+' in_file_1 in_file_2 out_file')
        print(sys.argv[0]+' in_raster.img in_raster_2.img out.img')
        sys.exit(1)

    #if os.name == 'nt':
    #    #base_path = 'P:/'
    #    base_path = 'D:/SMB3/shrubfs1'
    #else:
    #    base_path = '/caldera/projects/usgs/eros/rcmap'

    input_files = {
            'in_old': in_file_1,
            'in_new': in_file_2,
            }

    #Setup outputs starting with the burn stack (all outputs combined into one)
    output_files = {
            'out': out_file,
            }
    output_datatypes = {
            'out': gdal.GDT_Byte,
            }
    output_band_counts = {
            'out': 1,
            }
    output_drivers = {
            'out': 'GTiff',
            }
    output_options = {
            'out': ['COMPRESS=LZW','NBITS=1','BIGTIFF=YES','BLOCKXSIZE=128','BLOCKYSIZE=128'],
            }

    output_parameters = (output_files, output_datatypes, output_band_counts, output_drivers, output_options)
    main(input_files, output_parameters, func=model, args=None)

