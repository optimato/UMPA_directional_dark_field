from utils import *
import UMPA
import numpy as np
from matplotlib import pyplot as plt

from scipy.optimize import brent
from datetime import datetime
from in_out import load_esrf_data
from single_model import solver_at_resolution
from scipy import ndimage

class multi_resolution_solver:
    def __init__(self, sams, refs, final_step, final_nw, n_iters_bin=3, n_iters_final = 2, max_shift=5, step_multiplier=3,
                 max_sig=10, blur_extra=0.45):

        self.single_res_models = []
        self.n_iters_total = n_iters_bin + n_iters_final
        self.max_sig = max_sig
        self.n_iters_bin = n_iters_bin
        self.n_iters_final = n_iters_final

        self.stage = 0
        self.blur_extra = blur_extra

        for i in range(n_iters_bin):
            self.single_res_models.append(solver_at_resolution(sams, refs, step=final_step*(step_multiplier**(n_iters_bin - i - 1)), Nw=final_nw, max_shift=max_shift, max_sig= self.max_sig, blur_extra = self.blur_extra))
            print('Created model number {}, step is {}'.format(i, final_step * (step_multiplier ** (n_iters_bin - i - 1))))
        for i in range(n_iters_final):
            self.single_res_models.append(solver_at_resolution(sams, refs, step=final_step, Nw=final_nw, max_shift=max_shift, max_sig=self.max_sig))
            print('Created final {}, step is {}'.format(i, final_step))


    def solve_model(self, stage = None, method='golden', maxiter=20, tol=None, mode = None):
        '''
        stage = 0 for the first model
        '''
        if stage is None:
            stage = self.stage

        if mode is None:
            if stage <= 1:
                mode = 'rot'
            else:
                mode = 'new'

        self.single_res_models[stage].initial_run_of_model()
        self.single_res_models[stage].optimise_image(method=method, maxiter=maxiter, tol=tol, mode = mode)
        rgb = self.single_res_models[stage].return_rgb()
        return rgb

    def send_output_to_next_model(self, stage=None, blur=None):
        '''
        stage = 0 for first model

        TODO: make it so that it doesnt error if stage > max
        '''

        if stage is None:
            stage = self.stage
            self.stage += 1
            if stage == 0:
                blur = 3
            else:
                blur = 2

        #removing the bits that have the opposite symmetry to make bluring work

        output_without_symmetry = np.zeros_like(self.single_res_models[stage].final_vals)

        for i in range(output_without_symmetry.shape[0]):
            for j in range(output_without_symmetry.shape[1]):
                if self.single_res_models[stage].final_vals[i, j, 0] < 0:
                    output_without_symmetry[i, j, 0] = -1 * self.single_res_models[stage].final_vals[i, j, 0]
                    output_without_symmetry[i, j, 1] = -1 * self.single_res_models[stage].final_vals[i, j, 1]
                else:
                    output_without_symmetry[i, j, 0] = self.single_res_models[stage].final_vals[i, j, 0]
                    output_without_symmetry[i, j, 1] = self.single_res_models[stage].final_vals[i, j, 1]

                output_without_symmetry[i, j, 2] = abs(self.single_res_models[stage].final_vals[i, j, 2])

        if stage < self.n_iters_total - 1:

            #output_without_symmetry = self.single_res_models[stage].final_vals
            #upscale
            new_arr = np.zeros_like(self.single_res_models[stage+1].initial_vals)

            scale_y = self.single_res_models[stage].final_vals.shape[0] / self.single_res_models[stage+1].initial_vals.shape[0]
            scale_x = self.single_res_models[stage].final_vals.shape[1] / self.single_res_models[stage+1].initial_vals.shape[1]

            for i in range(new_arr.shape[0]):
                for j in range(new_arr.shape[1]):
                    py = int((i * scale_y) // 1)
                    px = int((j * scale_x) // 1)
                    new_arr[i, j, :] = output_without_symmetry[py, px, :]

            # blur
            new_arr[:, :, 0] = ndimage.uniform_filter((new_arr[:,:,0]), size=(blur, blur), mode='reflect') #* np.sign(ndimage.median_filter(new_arr[:,:,0], size=(blur, blur), mode='reflect'))
            new_arr[:, :, 1] = ndimage.uniform_filter((new_arr[:,:,1]), size=(blur, blur), mode='reflect') #* np.sign(ndimage.median_filter(new_arr[:,:,1], size=(blur, blur), mode='reflect'))
            new_arr[:, :, 2] = ndimage.median_filter((new_arr[:,:,2]), size=(blur, blur), mode='reflect')

            self.single_res_models[stage+1].initial_vals = new_arr
            print(f'successfully passed to next model and blurred with kernel size {blur}')

        else:
            print('Stage out of range')


def example_reconstruction(max_iter_final = 20000):

    final_nw = 10
    final_step = 20
    num_frames = 25

    save_path = '/home/rs3g18/Documents/2Dstar/DDF-output/test_data/test6'
    savename = save_path + '_N_' + str(num_frames) + '_step_' + str(final_step) + '_Nw_' + str(
        final_nw) + '_final_max_iter_' + str(max_iter_final)

    sim = UMPA.utils.prep_simul()

    sams = sim['meas']
    refs = sim['ref']

    big_model = multi_resolution_solver(sams, refs, final_step, final_nw, step_multiplier=3, n_iters_final=2,
                                        blur_extra = 0.05)

    for i in range(big_model.n_iters_total):
        if i < big_model.n_iters_total - 1:
            rgb = big_model.solve_model(maxiter=10000)
        else:
            rgb = big_model.solve_model(maxiter=max_iter_final, tol=1e-12, mode = 'new')

        rgb = big_model.single_res_models[i].return_rgb(sigma_max=5)

        fig = plt.figure(figsize=(7, 4))
        plt.imshow(rgb)
        plt.title('Reconstrution stage ' + str(i + 1))
        plt.savefig(savename + 'recon_stage_' + str(i) + '.png')
        plt.close(fig)

        F = h5py.File(savename + '.h5', 'a')

        F.create_dataset('gauss_properties_stage_' + str(i), data=big_model.single_res_models[i].gauss_properties)
        F.create_dataset('final_vals_stage_' + str(i), data=big_model.single_res_models[i].final_vals)
        F.create_dataset('transmission_image_stage_' + str(i), data=big_model.single_res_models[i].result['T'])
        F.create_dataset('dpcx_stage_' + str(i), data=big_model.single_res_models[i].result['dx'])
        F.create_dataset('dpcy_stage_' + str(i), data=big_model.single_res_models[i].result['dy'])
        F.create_dataset('initial_vals_' + str(i), data=big_model.single_res_models[i].initial_vals)
        F.close()

        big_model.send_output_to_next_model()

    print('done')
    return big_model

if __name__ == "__main__":
    import h5py
    big_model = example_reconstruction()


