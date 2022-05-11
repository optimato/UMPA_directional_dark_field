import numpy as np
from matplotlib import pyplot as plt
import h5py
import os
from glob import glob
from itertools import repeat
import UMPA
import traceback

from . import do_it_all_for_me
from . import generate_rgb

'''
This was badly put together to save RAM and work - not for sharing because its a bit spaghetti
'''

def arr_stitch(arr_list):
    '''
    Simple function for stitching 2d arrays together
    '''
    stitcher = arr_list[0][:, :-3]

    for i in range(1, len(arr_list)):
        if i != len(arr_list) - 1:
            stitcher = np.append(stitcher, arr_list[i][:, 3:-3], axis=1)
        else:
            stitcher = np.append(stitcher, arr_list[i][:, 3:], axis=1)

    return stitcher


def arr_stitch_3d(arr_list):
    '''
    Simple function for stitching 3d arrays together
    '''
    stitcher = arr_list[0][:, :-3, :]

    for i in range(1, len(arr_list)):
        if i != len(arr_list) - 1:
            stitcher = np.append(stitcher, arr_list[i][:, 3:-3, :], axis=1)
        else:
            stitcher = np.append(stitcher, arr_list[i][:, 3:, :], axis=1)

    return stitcher


def do_it_all_for_me_multiprocessing_memsave(sams, refs, save_path, final_nw=5, final_step=40,
                                     pos_list=None, sigma_max=15, max_shift=5, ROI=None, blur_extra=0.45, n_process=16, savepng=True):
    print('Im the new version')
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
    if ROI is not None:
        print('This only works if no ROI is set - please crop the inputs')
    if pos_list is not None:
        print('This is going to catastrophically fail - its only compatable with diffuser stepping for now')
    umpa_chop = (final_nw) + (max_shift) + 8

    main_ROI_width = sams[0].shape[1]
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
            start = ROI_edges[i] - umpa_chop - (3*final_step)
        if i == n_process - 1:
            end = ROI_edges[i + 1]
        else:
            end = ROI_edges[i + 1] + umpa_chop + (3*final_step)
        roi = [[0, sams[0].shape[0], 1], [start, end, 1]]
        ROIs.append(roi)

    print(ROIs)

    ## lets try some memory saving tricks!!!!
    n_rois = len(ROIs)
    sams_crop = []
    refs_crop = []
    new_ROIs = []
    for i in range(n_rois):
        sam_crop = [sam[ROIs[i][0][0]:ROIs[i][0][1], ROIs[i][1][0]:ROIs[i][1][1]] for sam in sams]
        ref_crop = [ref[ROIs[i][0][0]:ROIs[i][0][1], ROIs[i][1][0]:ROIs[i][1][1]] for ref in refs]
        roi = None#[[0,0,ROIs[i][0][2]],[0,0,ROIs[i][1][2]]]
        print(sam_crop[0].shape)
        sams_crop.append(sam_crop)
        refs_crop.append(ref_crop)
        new_ROIs.append(roi)

    #return sams, sams_crop, ROIs
    del sams, refs



    ## making temp savepaths
    if not os.path.exists(save_path + '_temp/'):
        os.mkdir(save_path + '_temp/')

    savepaths = []
    for i in range(n_process):
        tmp_savepath = save_path + '_temp/temp_ROI_no_' + str(i).zfill(3)
        savepaths.append(tmp_savepath)

    #args = zip(repeat(sams), repeat(refs), savepaths, repeat(final_nw), repeat(final_step), repeat(pos_list),
    #           repeat(sigma_max), repeat(max_shift), ROIs, repeat(blur_extra), repeat(savepng))
    args = zip(sams_crop, refs_crop, savepaths, repeat(final_nw), repeat(final_step), repeat(pos_list),
               repeat(sigma_max), repeat(max_shift), new_ROIs, repeat(blur_extra), repeat(savepng))
    # print(args[0])
    try:
        with Pool(n_process) as pool:
            #pool.starmap(fake_do_it_all_for_me, args)
            output = pool.starmap(do_it_all_for_me, args)
    except:
        traceback.print_last()
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

def test_reconstruction_multiprocessing_memsave(save_path = '/home/rs3g18/Documents/2Dstar/DDF-output/test_data/'):
    '''
    Test code for the multiprocessing
    '''

    final_nw = 2
    final_step = 1
    num_frames = 25
    max_iter_final = 200

    save_path = save_path + 'test_reconstruction7'
    savename = save_path + '_N_' + str(num_frames) + '_step_' + str(final_step) + '_Nw_' + str(
        final_nw) + '_final_max_iter_' + str(max_iter_final)

    sim = UMPA.utils.prep_simul()

    sams = sim['meas']
    refs = sim['ref']

    sams = [sam[:500, :500].copy() for sam in sams]
    refs = [ref[:500, :500].copy() for ref in refs]

    rgb = do_it_all_for_me_multiprocessing_memsave(sams, refs, savename, final_nw=final_nw, final_step=final_step, pos_list=None, sigma_max=1.5, blur_extra=0.05, n_process=16)
    print(rgb.shape)
    return rgb

if __name__ == '__main__':
    test_reconstruction_multiprocessing_memsave()