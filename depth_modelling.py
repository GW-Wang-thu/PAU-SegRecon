import copy

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
# from networks.ResUNet import GenerateDataloader, UT_SegmentationNet_light, UT_SegmentationNet_Normal, UT_SegmentationNet_tiny, MultiClassFocalLossWithAlpha
from utils.make_figures import evaluator
from scipy.io import savemat, loadmat


def calculate_depth(image, mask, depth_pix=0.2*(4/3.5)/15, threshold=0.1):
    image = image * 255.0 / np.max(image)
    mask_threshold = (image >= threshold*np.max(image))
    start_surf = np.argmax(image * mask_threshold * (mask==1), axis=1)
    end_surf = np.argmax(image * mask_threshold * (mask==2), axis=1)
    defect_surf = np.argmax(image * mask_threshold * (mask==3), axis=1)
    end_surf_ms = (end_surf - start_surf) * depth_pix
    end_surf_ms[end_surf <= defect_surf] = np.nan
    end_surf_ms[end_surf <= start_surf] = np.nan    # 后加
    end_surf_ms[end_surf == 0] = np.nan
    defect_surf_ms = (defect_surf - start_surf) * depth_pix
    defect_surf_ms[defect_surf == 0] = np.nan
    defect_surf_ms[defect_surf <= start_surf] = np.nan    # 后加
    start_surf_zero = np.zeros_like(defect_surf_ms)
    return start_surf_zero, end_surf_ms, defect_surf_ms


def plot_1d_defect(dep_surf, defect_surf, x_pix=0.1):
    x = np.linspace(0, (defect_surf.shape[0]-1)*x_pix, defect_surf.shape[0])
    zero_surf = np.zeros_like(dep_surf)
    plt.plot(x, zero_surf, label='top_surf')
    plt.plot(x, - dep_surf, label='backward_surf')
    plt.plot(x, - defect_surf, label='delamination')
    plt.show()


def plot_2d_defect(dep_surf_array, defect_surf_array, x_pix=0.1, y_pix=1, save_dir=None):

    x, y = np.meshgrid(np.linspace(0, x_pix * (dep_surf_array.shape[1] - 1), dep_surf_array.shape[1]),
                       np.linspace(0, y_pix * (dep_surf_array.shape[0] - 1), dep_surf_array.shape[0]))
    top_surf = np.zeros_like(x)
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    a2 = fig.add_subplot(122)

    # 绘制曲面

    # ax.plot_surface(x, y, top_surf)
    ax.plot_surface(x, y, - defect_surf_array)
    ax.plot_surface(x, y, - dep_surf_array)

    # 设置坐标轴标签
    ax.set_xlabel('X mm')
    ax.set_ylabel('Y mm')
    ax.set_zlabel('Depth mm')
    # a2.imshow(dep_surf_array, vmin=0, vmax=2)
    a2.imshow(defect_surf_array, vmin=0, vmax=2)
    if save_dir is not None:
        fig.savefig(save_dir)

    # 显示图形
    plt.show()


def error_summary():
    direc = r'E:\Data\Experiments\UT_202311\DATASET\B_4\ReconsResults'
    end_surf = 2.0
    case_with_labels = ['big2']#['thin1']#
    # pix_range = (55, 130)         # specimen #1
    # pix_range = (50, 125)         # specimen #2 part I
    # pix_range = (125, 250)        # specimen #2 part II
    pix_range = (250, 325)        # specimen #2 part III
    stx = pix_range[0]
    endx = pix_range[1]
    model_opts = [{'model_type': 'UNet', 'param_level': 'light-shallow', 'dataset': 'ndep'},
                  {'model_type': 'UNet', 'param_level': 'light-shallow', 'dataset': 'yz-lsin'},
                  {'model_type': 'UNet', 'param_level': 'light', 'dataset': 'ndep'},
                  {'model_type': 'UNet', 'param_level': 'light', 'dataset': 'yz-lsin'},
                  {'model_type': 'UNet', 'param_level': 'light-deep', 'dataset': 'ndep'},
                  {'model_type': 'UNet', 'param_level': 'light-deep', 'dataset': 'yz-lsin'},
                  {'model_type': 'TUNet', 'param_level': 'light-sparse', 'dataset': 'ndep'},
                  {'model_type': 'TUNet', 'param_level': 'light-sparse-yz', 'dataset': 'yz-lsin'}]

    for case_name in case_with_labels:
        tmp_mat_label = os.path.join(direc, case_name+'_depth_GT.mat')
        mat_label = loadmat(tmp_mat_label)
        label_defect = mat_label['defect_surf'][stx:endx, :]
        for model_opt in model_opts:
            model_type = model_opt['model_type']
            param_level = model_opt['param_level']
            dataset = model_opt['dataset']
            tmp_mat_pre = os.path.join(direc, case_name+'_'+model_type+'_'+param_level+'_' + dataset +'_depth.mat')
            mat_pre = loadmat(tmp_mat_pre)
            # pred_results = copy.deepcopy(mat_pre['defect_surf'])
            pred_results = copy.deepcopy(mat_pre['end_surf'][stx:endx, :])
            pred_defect = copy.deepcopy(mat_pre['defect_surf'][stx:endx, :])
            pred_end = copy.deepcopy(mat_pre['end_surf'][stx:endx, :])
            for i in range(pred_results.shape[0]):
                for j in range(pred_results.shape[1]):
                    if not np.isnan(pred_defect[i, j]):
                        pred_results[i, j] = pred_defect[i, j]
            error = []
            num_f_points = 0
            area_defect = 0
            area_defect_label = 0
            I = 0
            U = 0
            for i in range(label_defect.shape[0]):
                for j in range(label_defect.shape[1]):
                    if not np.isnan(pred_defect[i, j]):    # 如果是 pred defect 上的
                        area_defect += 1
                    if not np.isnan(label_defect[i, j]):    # 如果是 label defect 上的
                        area_defect_label += 1
                        if not np.isnan(pred_results[i, j]):    # 如果预测该点为缺陷 交集
                            error.append(np.abs(label_defect[i, j] - pred_results[i, j]))
                        else:                                   # 如果预测该点不为缺陷，但实际是缺陷
                            error.append(2.0)
                            num_f_points += 1
                    else:           # 如果是end surface
                        if not np.isnan(pred_results[i, j]):    # 如果预测该点不是nan
                            error.append(np.abs(end_surf - pred_results[i, j]))
                        else:
                            error.append(2.0)
                            num_f_points += 1

            for i in range(label_defect.shape[0]):
                for j in range(label_defect.shape[1]):
                    if not np.isnan(label_defect[i, j]):    # 如果是 label defect 上的
                        U += 1
                        if not np.isnan(pred_defect[i, j]):    # 如果预测该点为缺陷 交集
                            I += 1
                    else:
                        if not np.isnan(pred_defect[i, j]):    # 如果预测该点为缺陷 交集
                            U += 1

            plt.imshow(pred_results)
            plt.show()
            error_def = np.nanmean(np.abs(pred_defect - label_defect))
            error_end = np.nanmean(np.abs(pred_end - end_surf))
            # print("area pred: ", area_defect, '\t Ground truth: ', area_defect_label)
            print('case: ', case_name, ' model: ', model_type, '\t params: ', param_level, '\t dataset: ', dataset, '\t mean error: %.4f'%(np.mean(error)), '\t IOU: %.4f'%(I/U), '\t Fail points: %.4f'%(num_f_points/(label_defect.shape[0] * label_defect.shape[1])))

            # 比较 label 中被标记为 defect 的部分

def error_summary_DA():
    direc = r'E:\Data\Experiments\UT_202311\DATASET\B_4\ReconsResults'
    end_surf = 2.0
    case_with_labels = ['thin1']#['big2']#
    pix_range = (55, 130)         # specimen #1
    # pix_range = (50, 125)         # specimen #2 part I
    # pix_range = (125, 250)        # specimen #2 part II
    # pix_range = (250, 325)        # specimen #2 part III
    stx = pix_range[0]
    endx = pix_range[1]
    model_opts = [{'model_type': 'UNet', 'param_level': 'light-shallow', 'dataset': 'ndep'},
                  {'model_type': 'UNet', 'param_level': 'light-shallow', 'dataset': 'yz-lsin'},
                  {'model_type': 'UNet', 'param_level': 'light', 'dataset': 'ndep'},
                  {'model_type': 'UNet', 'param_level': 'light', 'dataset': 'yz-lsin'},
                  {'model_type': 'UNet', 'param_level': 'light-deep', 'dataset': 'ndep'},
                  {'model_type': 'UNet', 'param_level': 'light-deep', 'dataset': 'yz-lsin'},
                  {'model_type': 'TUNet', 'param_level': 'light-sparse', 'dataset': 'ndep'},
                  {'model_type': 'TUNet', 'param_level': 'light-sparse-yz', 'dataset': 'yz-lsin'}]

    for case_name in case_with_labels:
        tmp_mat_label = os.path.join(direc, case_name+'_depth_GT.mat')
        mat_label = loadmat(tmp_mat_label)
        label_defect = mat_label['defect_surf'][stx:endx, :]
        for model_opt in model_opts:
            model_type = model_opt['model_type']
            param_level = model_opt['param_level']
            dataset = model_opt['dataset']
            tmp_mat_pre = os.path.join(direc, case_name+'_'+model_type+'_'+param_level+'_' + dataset +'_depth.mat')
            mat_pre = loadmat(tmp_mat_pre)
            # pred_results = copy.deepcopy(mat_pre['defect_surf'])
            pred_endsurf = copy.deepcopy(mat_pre['end_surf'][stx:endx, :])
            pred_defect = copy.deepcopy(mat_pre['defect_surf'][stx:endx, :])
            error = []
            num_f_points = 0
            for i in range(label_defect.shape[0]):
                for j in range(label_defect.shape[1]):
                    if not np.isnan(label_defect[i, j]):    # 如果是 label defect 上的
                        if not np.isnan(pred_defect[i, j]):    # 如果预测该点为缺陷 交集
                            error.append(np.abs(label_defect[i, j] - pred_defect[i, j]))
                        else:                                   # 如果预测该点不为缺陷，但实际是缺陷
                            error.append(2.0)
                    else:           # 如果是end surface
                        if not np.isnan(pred_endsurf[i, j]):    # 如果预测该点不是nan
                            error.append(np.abs(end_surf - pred_endsurf[i, j]))
                        else:
                            error.append(2.0)

            # print("area pred: ", area_defect, '\t Ground truth: ', area_defect_label)
            print('case: ', case_name, ' model: ', model_type, '\t params: ', param_level, '\t dataset: ', dataset, '\t mean error: %.4f'%(np.mean(error)))

            # 比较 label 中被标记为 defect 的部分


def defect_modelling():
    # directory = r'E:\Data\Experiments\UT_202311\DATASET\B_Demo\eval\Segmentation_dep'

    params_dir = r'E:\Data\Experiments\UT_202311\DATASET\B_4\params'
    # bimg_dir = r'E:\Data\Experiments\UT_202311\DATASET\B_3\Segmentation_raw'
    bimg_dir = r'E:\Data\Experiments\UT_202311\DATASET\INIT\ALL_B_IMG'
    save_dir = r'E:\Data\Experiments\UT_202311\DATASET\B_4\ReconsResults\threshold\\'
    case_names = ['67-a', '22-a', 'big2', 'thin1']
    model_opts = [{'model_type': 'UNet', 'param_level': 'light-shallow', 'dataset': 'ndep'},
                  {'model_type': 'UNet', 'param_level': 'light-shallow', 'dataset': 'yz-lsin'},
                  {'model_type': 'UNet', 'param_level': 'light', 'dataset': 'ndep'},
                  {'model_type': 'UNet', 'param_level': 'light', 'dataset': 'yz-lsin'},
                  {'model_type': 'UNet', 'param_level': 'light-deep', 'dataset': 'ndep'},
                  {'model_type': 'UNet', 'param_level': 'light-deep', 'dataset': 'yz-lsin'},
                  {'model_type': 'TUNet', 'param_level': 'light-sparse', 'dataset': 'ndep'},
                  {'model_type': 'TUNet', 'param_level': 'light-sparse-yz', 'dataset': 'yz-lsin'}]
    for case_name in case_names:
        for model_opt in model_opts:
            model_type = model_opt['model_type']
            param_level = model_opt['param_level']
            dataset = model_opt['dataset']
            my_evaluator = evaluator(model=model_type, params_level=param_level, dataset=dataset, device='cuda', params_dir=params_dir)
            print(case_name, model_opt)
            img_array, mask_array = my_evaluator.eval_b_imgs(file_dir=bimg_dir, case_name=case_name)
            start_surf, end_surf, defect_surf = calculate_depth(img_array.astype('int32'), mask_array, threshold=0.12)
            # plot_2d_defect(end_surf, defect_surf, save_dir=save_dir + case_name+'_'+model_type+'_'+param_level+'_' + dataset +'_depth.png')

            savemat(save_dir + case_name+'_'+model_type+'_'+param_level+'_' + dataset +'_depth.mat', {'start_surf': start_surf, 'end_surf': end_surf, 'defect_surf': defect_surf})
        # modelling_defect_1D(os.path.join(directory, '1_38-a_161_v.npy'))
        # modelling_defect_2D(directory, fname='1_120-a')

def contrast_specimen_label():
    img_dir = r'E:\Data\Experiments\UT_202311\DATASET\INIT\ALL_B_IMG\\'
    case_name = 'thin1'
    idx = '103'
    img = cv2.imread(os.path.join(img_dir, case_name+'_'+idx+'.png'), cv2.IMREAD_GRAYSCALE)
    img = img.astype('int32') * 255 / np.max(img)
    pass
    save_dir = r'E:\Data\Experiments\UT_202311\DATASET\B_4\ReconsResults\\'
    defect_surf = np.loadtxt(save_dir+case_name+'GT.csv', delimiter=',')
    defect_surf[defect_surf == 0] = np.nan
    start_surf = np.zeros_like(defect_surf)
    end_surf = np.ones_like(defect_surf) * 2.0
    savemat(save_dir + case_name+'_depth_GT.mat', {'start_surf': start_surf, 'end_surf': end_surf, 'defect_surf': defect_surf})


if __name__ == '__main__':
    defect_modelling()
    # error_summary()
    error_summary_DA()
    # contrast_specimen_label()