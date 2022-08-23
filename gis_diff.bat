REM Single Process
REM python script.py {parameters}
REM Multi Process using MPI
REM mpiexec -n 8 python script.py {parameters}

python gis_diff.py data\20220819_mr_col76_wv3_multi_071917_utm13.tree_mask.tif data\20220823_mr_col76_wv3_multi_071917_utm13.tree_mask.tif out.tif
