import numpy as np
from matplotlib import pyplot as plt
import h5py
import os
from glob import glob
from itertools import repeat
import UMPA

from . import do_it_all_for_me
from . import generate_rgb

def arr_stitch(arr_list):
    '''
    Simple function for stitching 2d arrays together
    '''
    stitcher = arr_list[0][:, :-4]

    for i in range(1, len(arr_list)):
        if i != len(arr_list) - 1:
            stitcher = np.append(stitcher, arr_list[i][:, 3:-4], axis=1)
        else:
            stitcher = np.append(stitcher, arr_list[i][:, 3:], axis=1)

    return stitcher


def arr_stitch_3d(arr_list):
    '''
    Simple function for stitching 3d arrays together
    '''
    stitcher = arr_list[0][:, :-4, :]

    for i in range(1, len(arr_list)):
        if i != len(arr_list) - 1:
            stitcher = np.append(stitcher, arr_list[i][:, 3:-4, :], axis=1)
        else:
            stitcher = np.append(stitcher, arr_list[i][:, 3:, :], axis=1)

    return stitcher


def do_it_all_for_me_multiprocessing(sams, refs, save_path, final_nw=5, final_step=40,
                                     pos_list=None, sigma_max=15, max_shift=5, ROI=None, blur_extra=0.45, n_process=16, savepng=False):
    '''
    This is a VERY simple parallel implementation using python multiprocessing module
    Note that it is memory and resource inefficient!!!!!
    n_process -> int, number of separate processes to use
    See 'do_it_all_for_me' for description of other arguments
    '''

    try:
        from multiprocessing import Pool

    except:
        print('Multiproccesing not installed, bad things might happen now')

    ## Splitting ROIs
    if ROI is None:
        if pos_list is None:
            ROI_end_0 = sams[0].shape[0] - (2*final_nw) - (2*max_shift) - 1
            ROI_end_1 = sams[0].shape[1] - (2 * final_nw) - (2 * max_shift) - 1
            ROI = [[0, ROI_end_0, 1], [0, ROI_end_1, 1]]


    main_ROI_start = ROI[1][0]
    main_ROI_end = ROI[1][1]
    main_ROI_width = main_ROI_end - main_ROI_start
    print(main_ROI_width)

    ROI_edges = []

    for i in range(n_process + 1):
        j = (main_ROI_width * i) // n_process
        ROI_edges.append(j)

    ROIs = []
    for i in range(n_process):
        if i == 0:
            start = ROI_edges[i]
        else:
            start = ROI_edges[i] - (3* final_step)
        if i == n_process - 1:
            end = ROI_edges[i + 1]
        else:
            end = ROI_edges[i + 1] + (3* final_step)
        roi = [ROI[0],[main_ROI_start + start, main_ROI_start + end, ROI[1][2]]]
        ROIs.append(roi)

    ## making temp savepaths
    if not os.path.exists(save_path + '_temp/'):
        os.mkdir(save_path + '_temp/')

    savepaths = []
    for i in range(n_process):
        tmp_savepath = save_path + '_temp/temp_ROI_no_' + str(i).zfill(3)
        savepaths.append(tmp_savepath)

    args = zip(repeat(sams), repeat(refs), savepaths, repeat(final_nw), repeat(final_step), repeat(pos_list),
               repeat(sigma_max), repeat(max_shift), ROIs, repeat(blur_extra), repeat(savepng))
    # print(args[0])
    try:
        with Pool(n_process) as pool:
            #pool.starmap(fake_do_it_all_for_me, args)
            output = pool.starmap(do_it_all_for_me, args)
    except:
        print('Multiprocessing gave some errors but I dont know how to avoid this. Maybe everything will be ok.')

    outs = sorted(glob(save_path + '_temp/' + '*.h5'))

    t_list = []
    dpcx_list = []
    dpcy_list = []
    gp_list = []

    for out in outs:
        F = h5py.File(out, 'r')
        t_list.append(np.array(F['transmission_image_stage_2']))
        dpcx_list.append(np.array(F['dpcx_stage_2']))
        dpcy_list.append(np.array(F['dpcy_stage_2']))
        gp_list.append(np.array(F['gauss_properties_stage_2']))
        F.close()

    t_im = arr_stitch(t_list)
    dpcx_im = arr_stitch(dpcx_list)
    dpcy_im = arr_stitch(dpcy_list)
    gp_im = arr_stitch_3d(gp_list)

    # now for some saving

    out_path = outs[0].split('/')
    out_path[-1] = out_path[-1][15:]
    out_path[-2] = out_path[-2][:-5]
    out_path[-2] += out_path[-1]
    del (out_path[-1])
    out_path = '/'.join(out_path)

    F = h5py.File(out_path, 'a')
    F.create_dataset('gauss_properties_stage_2', data=gp_im)
    F.create_dataset('dpcx_stage_2', data=dpcx_im)
    F.create_dataset('dpcy_stage_2', data=dpcy_im)
    F.create_dataset('transmission_image_stage_2', data=t_im)
    F.close()

    rgb = generate_rgb(gp_im, log_scale='True')[0]
    fig = plt.figure(figsize=(7, 4))
    plt.imshow(rgb)
    plt.title('Multiprocess Output')
    plt.savefig(out_path[:-5] + '.png')
    plt.close(fig)
    return rgb

def test_reconstruction_multiprocessing(save_path = ''):
    '''
    Test code for the multiprocessing
    '''

    final_nw = 5
    final_step =5
    num_frames = 25
    max_iter_final = 200

    save_path = save_path + 'test_reconstruction'
    savename = save_path + '_N_' + str(num_frames) + '_step_' + str(final_step) + '_Nw_' + str(
        final_nw) + '_final_max_iter_' + str(max_iter_final)

    sim = UMPA.utils.prep_simul()

    sams = sim['meas']
    refs = sim['ref']

    rgb = do_it_all_for_me_multiprocessing(sams, refs, savename, final_nw=final_nw, final_step=final_step, pos_list=None, sigma_max=1.5, blur_extra=0.05, n_process=8)
    return rgb

if __name__ == '__main__':
    test_reconstruction_multiprocessing()