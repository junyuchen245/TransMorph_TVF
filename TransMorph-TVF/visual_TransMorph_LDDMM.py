import glob, sys
import os, losses, utils
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph
import pystrum.pynd.ndutils as nd

def main():
    test_dir = 'D:/DATA/OASIS/Test/'
    model_idx = -1
    weights = [1, 1]
    alpha = 0.01
    gamma = 0.001
    time_steps = 12
    model_folder = 'TransMorphTVFLDDMM_tsteps_{}_mse_{}_mse_{}_LDDMM_alpha{}gamma{}/'.format(time_steps, weights[0], weights[1], alpha, gamma)
    model_dir = 'experiments/' + model_folder
    config = CONFIGS_TM['TransMorph-Large']
    model = TransMorph.TransMorphTVFBackward(config, time_steps=time_steps)
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()
    reg_model = utils.register_model(config.img_size, 'nearest')
    reg_model.cuda()
    file_names = glob.glob(test_dir + '*.pkl')
    reppad = nn.ReplicationPad3d(time_steps)
    with torch.no_grad():
        stdy_idx = 0
        for data in file_names:
            if stdy_idx != 0:
                stdy_idx += 1
                continue
            x, y, x_seg, y_seg = utils.pkload(data)
            x, y = x[None, None, ...], y[None, None, ...]
            x = np.ascontiguousarray(x)
            y = np.ascontiguousarray(y)
            x, y = torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()
            file_name = file_names[stdy_idx].split('\\')[-1].split('.')[0][2:]
            print(file_name)
            model.eval()
            x_in = torch.cat((x, y),dim=1)
            x_def, flow_for, flow_inv, flows, flows_out = model(x_in)
            flow = flow_for.cpu().detach().numpy()[0]#.astype(np.float16)
            y_def = model.spatial_trans(y, flow_inv)
            grid_img = mk_grid_img(8, 1, config.img_size)
            # xdefs = []
            def_grids = []
            for i in range(time_steps):
                flow_tmp = flows_out[i][:, :, time_steps:-time_steps, time_steps:-time_steps, time_steps:-time_steps]
                flow_tmp = reppad(flow_tmp)
                flow_a = model.tri_up(flow_tmp)
                # xdefs.append(x_def[i].cpu().detach().numpy()[0, 0, :, :, 120])
                def_grid = model.spatial_trans(grid_img.float(), flow_a)
                def_grids.append(def_grid.cpu().detach().numpy()[0, 0, :, :, :])
            det = jacobian_determinant_vxm(flow)
            print(np.sum(det))

            x_flow = model.spatial_trans(grid_img.float(), flow_for)
            y_flow = model.spatial_trans(grid_img.float(), flow_inv)
            slice_num = 90
            plt.figure(figsize=(12, 12), dpi=180)
            plt.subplot(4, 2, 1)
            plt.imshow(x.cpu().detach().numpy()[0, 0, :, slice_num, :], cmap='gray', vmin=0, vmax=0.8)
            plt.title('x')
            plt.subplot(4, 2, 2)
            plt.imshow(y.cpu().detach().numpy()[0, 0, :, slice_num, :], cmap='gray', vmin=0, vmax=0.8)
            plt.title('y')
            plt.subplot(4, 2, 3)
            plt.axis('off')
            plt.imshow(x_def.cpu().detach().numpy()[0, 0, :, slice_num, :], cmap='gray', vmin=0, vmax=0.8)
            plt.title('x_def')
            plt.subplot(4, 2, 4)
            plt.axis('off')
            plt.imshow(y_def.cpu().detach().numpy()[0, 0, :, slice_num, :], cmap='gray', vmin=0, vmax=0.8)
            plt.title('y_def')
            plt.subplot(4, 2, 5)
            plt.axis('off')
            plt.imshow(x_flow.cpu().detach().numpy()[0, 0, :, slice_num, :], cmap='gray')
            plt.title('Forward flow')
            plt.subplot(4, 2, 6)
            plt.axis('off')
            plt.imshow(y_flow.cpu().detach().numpy()[0, 0, :, slice_num, :], cmap='gray')
            plt.title('Inverse flow')
            plt.subplot(4, 2, 7)
            plt.axis('off')
            plt.imshow((x_def-y).cpu().detach().numpy()[0, 0, :, slice_num, :], vmin=-0.5, vmax=0.5)
            plt.title('Forward diff')
            plt.subplot(4, 2, 8)
            plt.axis('off')
            plt.imshow((y_def-x).cpu().detach().numpy()[0, 0, :, slice_num, :], vmin=-0.5, vmax=0.5)
            plt.title('Inverse diff')
            plt.show()
            for i in range(time_steps):
                plt.subplot(int(np.ceil(np.sqrt(time_steps))), int(np.ceil(np.sqrt(time_steps))), i + 1)
                plt.axis('off')
                plt.imshow(def_grids[i][:, 80, :], cmap='gray')
            plt.show()
            flow_back = model.spatial_trans(x_flow.float(), flow_inv)
            plt.figure(figsize=(12, 12), dpi=180)
            plt.imshow(flow_back.cpu().detach().numpy()[0, 0, :, slice_num, :], cmap='gray')
            plt.show()
            sys.exit(0)

def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[0], grid_step):
        grid_img[j+line_thickness-1, :, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img

def jacobian_determinant_vxm(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    disp = disp.transpose(1, 2, 3, 0)
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]

def make_girds(flow):
    img_sz_x, img_sz_y = flow.shape[1:]
    x = np.arange(0, img_sz_x, 1)
    y = np.arange(0, img_sz_y, 1)
    X, Y = np.meshgrid(x, y)
    phix = X.transpose(1, 0)
    phiy = Y.transpose(1, 0)
    return phix, phiy

def apply_flow(flow, phix, phiy):
    img_sz_x, img_sz_y = flow.shape[1:]
    u = flow[0, :, :]
    v = flow[1, :, :]
    for i in range(0, img_sz_x):
        for j in range(0, img_sz_y):
            # add the displacement for each p(k) in the sum
            phix[i, j] = phix[i, j] + u[i, j]
            phiy[i, j] = phiy[i, j] + v[i, j]
    return phix, phiy

def plot_grid(gridx, gridy, **kwargs):
    for i in range(gridx.shape[0]):
        plt.plot(gridx[i,:], gridy[i,:], linewidth=0.8, **kwargs)
    for i in range(gridx.shape[1]):
        plt.plot(gridx[:,i], gridy[:,i], linewidth=0.8, **kwargs)

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 1
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()