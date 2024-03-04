import os.path

import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision.models as models
from networks.ResUNet import GenerateDataloader, UT_SegmentationNet_Normal, MultiClassFocalLossWithAlpha, \
    UT_SegmentationNet_tiny, UT_SegmentationNet_light, UT_SegmentationNet_Normal_deep, \
    UT_SegmentationNet_light_deep, UT_SegmentationNet_tiny_deep, UT_SegmentationNet_light_shallow, \
    UT_SegmentationNet_Normal_shallow, UT_SegmentationNet_tiny_shallow #UT_Segmentation_DepthPriorNet_Light, UT_Segmentation_DepthPriorNet, Multi_2Class_FocalLossWithAlpha
from networks.TransUNet import UT_SegTransUNet_Normal, UT_SegTransUNet_light, UT_SegTransUNet_tiny,\
    UT_SegTransUNet_utiny
from networks.LossFuncs import DiceLoss
import json
import time
from torchsummary import summary as sm
from utils.make_figures import gen_mask_demo_img


class accuracy_evaluation():
    def __init__(self, n_class = 4):
        self.n_class = n_class
        self.pre_show = None
        self.lab_show = None

    def evaluate_all(self, predict, labels, mode=0):
        if mode == 1:
            _, predicted_st = torch.max(predict[:, 0:2, :, :].data, dim=1)
            _, predicted_end = torch.max(predict[:, 2:4, :, :].data, dim=1)
            _, predicted_def = torch.max(predict[:, 4:, :, :].data, dim=1)
            # plt.imshow(predicted_st[1, :, :].cpu().detach().numpy())
            # plt.show()
            pre = torch.zeros_like(predicted_st)
            pre[predicted_st == 1] = 1
            pre[predicted_end == 1] = 2
            pre[predicted_def == 1] = 3
            lab = labels[:, 0, :, :]
            lab[labels[:, 1, :, :] == 1] = 2
            lab[labels[:, 2, :, :] == 1] = 3
            lab[labels[:, 3, :, :] == 1] = 4
            lab = lab.cpu().detach().numpy()
            pre = pre.cpu().detach().numpy()
        else:
            _, predicted = torch.max(predict.data, dim=1)
            pre = predicted.cpu().detach().numpy()
            lab = labels.cpu().detach().numpy()
        PA = self._multi_pa(pre, lab)
        IOU = self._multi_iou(pre, lab)
        DICE = self._multi_dice(pre, lab)

        self.pre_show = pre
        self.lab_show = lab

        return PA, IOU, DICE

    def show_seg_img(self, all=False, title='', save=False):
        if self.pre_show is None:
            return None
        if all:
            lenth = self.pre_show.shape[0]
        else:
            lenth = 1
        for i in range(lenth):
            plt.title(title)
            plt.subplot(1, 2, 1)
            plt.imshow(self.pre_show[i, :, :], vmax=4)
            plt.subplot(1, 2, 2)
            plt.imshow(self.lab_show[i, :, :], vmax=4)
            if save:
                plt.savefig('Demo//'+title+'.png')
            plt.show()

    def plot_loss(self, dict, title='', save=False):
        train_epoch = dict['train epoch'][1:]
        valid_epoch = dict['valid epoch'][1:]
        train_loss = dict['train loss'][1:]
        valid_loss = dict['valid loss'][1:]
        train_rec = np.array(dict['train rec'][1:])
        valid_rec = np.array(dict['valid rec'][1:])
        plt.subplot(2, 2, 1)
        plt.title('Log CE Loss')
        plt.plot(train_epoch, np.log(np.array(train_loss)), label='train')
        plt.plot(valid_epoch, np.log(np.array(valid_loss)), label='valid')
        plt.legend()
        plt.subplot(2, 2, 2)
        plt.title('PA Score')
        plt.plot(train_epoch, train_rec[:, 0], label='train')
        plt.plot(valid_epoch, valid_rec[:, 0], label='valid')
        plt.legend()
        plt.subplot(2, 2, 3)
        plt.title('IOU Score')
        plt.plot(train_epoch, train_rec[:, 1], label='train')
        plt.plot(valid_epoch, valid_rec[:, 1], label='valid')
        plt.legend()
        plt.subplot(2, 2, 4)
        plt.title('Dice Score')
        plt.plot(train_epoch, train_rec[:, 2], label='train')
        plt.plot(valid_epoch, valid_rec[:, 2], label='valid')
        plt.legend()
        if save:
            plt.savefig('Demo//'+title+'.png')
        plt.show()

    def _multi_pa(self, s, g):
        pa = []
        for i in range(self.n_class):
            tp = ((s == i) & (g == i)).sum()
            tn = ((s != i) & (g != i)).sum()
            fp = ((s != i) & (g == i)).sum()
            fn = ((s == i) & (g != i)).sum()
            pa.append((tp + tn) / (tp + tn + fp + fn))
        # pa /= (self.n_class)
        return pa

    def _multi_iou(self, s, g):
        iou = []
        for i in range(self.n_class):
            tp = ((s == i) & (g == i)).sum()
            fp = ((s != i) & (g == i)).sum()
            fn = ((s == i) & (g != i)).sum()
            if tp > 0:
                iou.append(tp / (tp + fp + fn))
            else:
                iou.append(np.nan)
        # iou /= (self.n_class)
        return iou

    def _multi_dice(self, s, g):
        dice = []
        for i in range(self.n_class):
            tp = ((s == i) & (g == i)).sum()
            fp = ((s != i) & (g == i)).sum()
            fn = ((s == i) & (g != i)).sum()
            if tp > 0:
                dice.append(2 * tp / (2 * tp + fp + fn))
            else:
                dice.append(np.nan)
        # dice /= (self.n_class)
        return dice


def train(opt, model_type='normal', device='cuda', savedir='params', num_epochs=500, dataset_dir=r'E:\Data\Experiments\UT_202311\DATASET\B_5'):
    train_dataloader = GenerateDataloader(dataset_dir + r'\Segmentation_'+opt, type='Train', model='seg', opt=opt, batch_size=8, mode=0)
    valid_dataloader = GenerateDataloader(dataset_dir + r'\Segmentation_'+opt, type='Valid', model='seg', opt=opt, batch_size=8, mode=0)

    if model_type == 'normal':
        model = UT_SegmentationNet_Normal(num_class=4).to(device)
        if opt.split('-')[0] == 'yz':
            model = UT_SegmentationNet_Normal(num_class=4, in_channels=3).to(device)
    if model_type == 'normal-shallow':
        model = UT_SegmentationNet_Normal_shallow(num_class=4).to(device)
        if opt.split('-')[0] == 'yz':
            model = UT_SegmentationNet_Normal_shallow(num_class=4, in_channels=3).to(device)
    if model_type == 'tiny':
        model = UT_SegmentationNet_tiny(num_class=4, ).to(device)
        if opt.split('-')[0] == 'yz':
            model = UT_SegmentationNet_tiny(num_class=4, in_channels=3).to(device)
    if model_type == 'tiny-shallow':
        model = UT_SegmentationNet_tiny_shallow(num_class=4, ).to(device)
        if opt.split('-')[0] == 'yz':
            model = UT_SegmentationNet_tiny_shallow(num_class=4, in_channels=3).to(device)
    elif model_type == 'light':
        model = UT_SegmentationNet_light(num_class=4, ).to(device)
        if opt.split('-')[0] == 'yz':
            model = UT_SegmentationNet_light(num_class=4, in_channels=3).to(device)
    elif model_type == 'light-shallow':
        model = UT_SegmentationNet_light_shallow(num_class=4, ).to(device)
        if opt.split('-')[0] == 'yz':
            model = UT_SegmentationNet_light_shallow(num_class=4, in_channels=3).to(device)
    elif model_type == 'light-deep':
        model = UT_SegmentationNet_light_deep(num_class=4, ).to(device)
        if opt.split('-')[0] == 'yz':
            model = UT_SegmentationNet_light_deep(num_class=4, in_channels=3).to(device)

    loss_layer = DiceLoss(n_classes=4)
    if os.path.exists(savedir + '//SegUNet_'+model_type+'_'+opt+"_loss_rec.json"):
        with open(savedir + '//SegUNet_'+model_type+'_'+opt+"_loss_rec.json", 'r') as f:
            loss_rec = json.load(f)
        steps = 50 * ((loss_rec['train epoch'][-1] + 1) // 50)
        model.load_state_dict(torch.load(savedir + '//SegUNet_Last_'+model_type+'_'+opt+'.pth'))
        max_iou = np.max(np.array(loss_rec['valid rec'])[:, 1])
    else:
        max_iou = 0
        steps = 0
        loss_rec = {'train epoch': [], 'valid epoch': [], 'train loss': [], 'valid loss': [], 'train rec': [],
                    'valid rec': []}
    num_epochs -= steps
    # 定义损失函数和优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.9, weight_decay=1e-4)
    evaluator = accuracy_evaluation(n_class=3)

    # 训练模型
    for n in range(num_epochs):
        epoch = n + steps
        mean_train_loss = 0
        tiou = []
        tdice = []
        tPA = []
        for i, (images, labels, fname) in enumerate(train_dataloader):

            # 前向传播
            outputs = model(images)
            focal_loss = loss_layer(outputs, labels, weight=[0.8, 0.7, 1.0, 0.4])
            mean_train_loss += focal_loss.item()

            # 反向传播和优化
            optimizer.zero_grad()
            focal_loss.backward()
            optimizer.step()
            tmp_pa, tmp_iou, tmp_dice = evaluator.evaluate_all(outputs, labels, 0)
            tPA.append(tmp_pa)
            tiou.append(tmp_iou)
            tdice.append(tmp_dice)

        loss_rec['train epoch'].append(epoch)
        loss_rec['train loss'].append(mean_train_loss / len(train_dataloader.dataset))
        loss_rec['train rec'].append([np.nanmean(tPA), np.nanmean(tiou), np.nanmean(tdice)])

        if epoch % 10 == 0:
            # evaluator.show_seg_img()
            mean_loss = 0
            viou = []
            vdice = []
            vPA = []
            for j, (images, labels, fname) in enumerate(valid_dataloader):
                # 前向传播
                outputs = model(images)
                focal_loss = loss_layer(outputs, labels, weight=[0.8, 0.7, 1.0, 0.4])
                mean_loss += focal_loss.item()
                _, predicted = torch.max(outputs.data, dim=1)
                tmp_pa, tmp_iou, tmp_dice = evaluator.evaluate_all(outputs, labels, 0)
                vPA.append(tmp_pa)
                viou.append(tmp_iou)
                vdice.append(tmp_dice)
            loss_rec['valid epoch'].append(epoch)
            loss_rec['valid loss'].append(mean_loss / len(valid_dataloader.dataset))
            loss_rec['valid rec'].append([np.nanmean(vPA), np.nanmean(viou), np.nanmean(vdice)])
            # evaluator.show_seg_img()
            # 打印训练信息
            print('Epoch [{}/{}], Model: {}, Train Loss: {:.4f}, T_PA={:.4f}, T_IOU={:.4f}, T_Dice={:.4f}, Valid Loss: {:.4f}, V_PA={:.4f}, V_IOU={:.4f}, V_Dice={:.4f}'.
                  format(epoch + 1, num_epochs, model_type+'_'+opt,
                         mean_train_loss / len(train_dataloader.dataset),
                         np.nanmean(tPA), np.nanmean(tiou), np.nanmean(tdice),
                         mean_loss / len(valid_dataloader.dataset),
                         np.nanmean(vPA), np.nanmean(viou), np.nanmean(vdice)))

            if len(loss_rec['valid rec']) > 2:
                if np.nanmean(viou) > max_iou:
                    max_iou = np.nanmean(viou)
                    torch.save(model.state_dict(), savedir + '//SegUNet_Best_'+ model_type + '_' + opt + '.pth')
        if epoch % 50 == 49:
            torch.save(model.state_dict(), savedir + '//SegUNet_Last_'+model_type+'_'+opt+'.pth')

            with open(savedir + '//SegUNet_'+model_type+'_'+opt+"_loss_rec.json", "w") as f:
                json.dump(loss_rec, f)
            evaluator.plot_loss(loss_rec)


def train_TransUNet(opt='ndep', model_type='normal', device='cuda', savedir='params', num_epochs=500, dataset_dir=r'E:\Data\Experiments\UT_202311\DATASET\B_5'):
    train_dataloader = GenerateDataloader(dataset_dir + r'\Segmentation_'+opt, type='Train', model='seg', opt=opt, batch_size=8, mode=0)
    valid_dataloader = GenerateDataloader(dataset_dir + r'\Segmentation_'+opt, type='Valid', model='seg', opt=opt, batch_size=8, mode=0)
    if model_type == 'normal':
        model = UT_SegTransUNet_Normal(in_channels=2, num_class=4).to(device)
    elif model_type == 'light':
        model = UT_SegTransUNet_light(in_channels=2, num_class=4).to(device)
    elif model_type == 'light-LNF':
        model = UT_SegTransUNet_light(in_channels=2, num_class=4, LN=False).to(device)
    elif model_type == 'light-yz':
        model = UT_SegTransUNet_light(in_channels=3, num_class=4).to(device)
    elif model_type == 'light-hidden256':
        model = UT_SegTransUNet_light(in_channels=2, num_class=4, hidden_size=256).to(device)
    elif model_type == 'light-sparse':
        model = UT_SegTransUNet_light(in_channels=2, num_class=4, grid_size=(16, 4)).to(device)
    elif model_type == 'light-sparse-yz':
        model = UT_SegTransUNet_light(in_channels=3, num_class=4, grid_size=(16, 4)).to(device)
    elif model_type == 'light-ultra-sparse':
        model = UT_SegTransUNet_light(in_channels=2, num_class=4, grid_size=(8, 2)).to(device)
    elif model_type == 'tiny':
        model = UT_SegTransUNet_tiny(in_channels=2, num_class=4).to(device)
    elif model_type == 'ultra-tiny':
        model = UT_SegTransUNet_utiny(in_channels=2, num_class=4, hidden_size=64).to(device)
    elif model_type == 'ultra-tiny-sparse':
        model = UT_SegTransUNet_utiny(in_channels=2, num_class=4, hidden_size=64, grid_size=(16, 4)).to(device)
    elif model_type == 'ultra-tiny-ultra-sparse':
        model = UT_SegTransUNet_utiny(in_channels=2, num_class=4, hidden_size=64, grid_size=(8, 2)).to(device)

    if os.path.exists(savedir + '//SegTransUNet_' + model_type+ '_' + opt+ "_loss_rec.json"):
        with open(savedir + '//SegTransUNet_'+model_type+ '_' + opt+"_loss_rec.json", 'r') as f:
            loss_rec = json.load(f)
        steps = 50 * ((loss_rec['train epoch'][-1] + 1) // 50)
        model.load_state_dict(torch.load(savedir + '//SegTransUNet_Last_'+model_type+ '_' + opt+'.pth'))
        max_iou = np.max(np.array(loss_rec['valid rec'])[:, 1])

    else:
        max_iou = 0
        steps = 0
        loss_rec = {'train epoch': [], 'valid epoch': [], 'train loss': [], 'valid loss': [], 'train rec': [], 'valid rec': []}
    num_epochs -= steps
    # 定义损失函数和优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.9, weight_decay=1e-4)
    evaluator = accuracy_evaluation(n_class=3)
    loss_layer = DiceLoss(n_classes=4)


    # 训练模型
    for n in range(num_epochs):
        epoch = n + steps
        mean_train_loss = 0
        tiou = []
        tdice = []
        tPA = []
        for i, (images, labels, fname) in enumerate(train_dataloader):

            # 前向传播
            outputs = model(images)
            focal_loss = loss_layer(outputs, labels, weight=[0.8, 0.7, 1.0, 0.4])
            mean_train_loss += focal_loss.item()

            # 反向传播和优化
            optimizer.zero_grad()
            focal_loss.backward()
            optimizer.step()
            tmp_pa, tmp_iou, tmp_dice = evaluator.evaluate_all(outputs, labels, 0)
            tPA.append(tmp_pa)
            tiou.append(tmp_iou)
            tdice.append(tmp_dice)

        loss_rec['train epoch'].append(epoch)
        loss_rec['train loss'].append(mean_train_loss / len(train_dataloader.dataset))
        loss_rec['train rec'].append([np.nanmean(tPA), np.nanmean(tiou), np.nanmean(tdice)])

        if epoch % 10 == 0:
            # evaluator.show_seg_img()
            mean_loss = 0
            viou = []
            vdice = []
            vPA = []
            for j, (images, labels, fname) in enumerate(valid_dataloader):
                # 前向传播
                outputs = model(images)
                focal_loss = loss_layer(outputs, labels)
                mean_loss += focal_loss.item()
                _, predicted = torch.max(outputs.data, dim=1)
                tmp_pa, tmp_iou, tmp_dice = evaluator.evaluate_all(outputs, labels, 0)
                vPA.append(tmp_pa)
                viou.append(tmp_iou)
                vdice.append(tmp_dice)
            loss_rec['valid epoch'].append(epoch)
            loss_rec['valid loss'].append(mean_loss / len(valid_dataloader.dataset))
            loss_rec['valid rec'].append([np.nanmean(vPA), np.nanmean(viou), np.nanmean(vdice)])
            # evaluator.show_seg_img()
            # 打印训练信息
            print('Epoch [{}/{}], Model: {}, Train Loss: {:.4f}, T_PA={:.4f}, T_IOU={:.4f}, T_Dice={:.4f}, Valid Loss: {:.4f}, V_PA={:.4f}, V_IOU={:.4f}, V_Dice={:.4f}'.
                  format(epoch + 1, num_epochs, model_type + '_' + opt,
                         mean_train_loss / len(train_dataloader.dataset),
                         np.nanmean(tPA), np.nanmean(tiou), np.nanmean(tdice),
                         mean_loss / len(valid_dataloader.dataset),
                         np.nanmean(vPA), np.nanmean(viou), np.nanmean(vdice)))

            if len(loss_rec['valid rec']) > 2:
                if np.nanmean(viou) > max_iou:
                    max_iou = np.nanmean(viou)
                    torch.save(model.state_dict(), savedir + '//SegTransUNet_Best_'+ model_type+ '_' + opt+ '.pth')
        if epoch % 50 == 49:
            torch.save(model.state_dict(), savedir + '//SegTransUNet_Last_'+model_type+ '_' + opt+'.pth')

            with open(savedir + '//SegTransUNet_'+model_type+ '_' + opt+"_loss_rec.json", "w") as f:
                json.dump(loss_rec, f)
            evaluator.plot_loss(loss_rec)


def eval(opt, model_type='normal', device='cuda', savedir = 'params', dataset_dir=r'E:\Data\Experiments\UT_202311\DATASET\B_5', summary=False, save_pic=False):
    valid_dataloader = GenerateDataloader(dataset_dir + r'\Segmentation_'+opt, type='Valid', model='seg', opt=opt, batch_size=1, mode=0, shuffle=False)
    # valid_dataloader = GenerateDataloader(dataset_dir + r'\Segmentation_'+opt, type='Valid', model='seg', opt=opt, batch_size=8, mode=0)
    if model_type == 'normal':
        model = UT_SegmentationNet_Normal(num_class=4).to(device)
        if opt.split('-')[0] == 'yz':
            model = UT_SegmentationNet_Normal(num_class=4, in_channels=3).to(device)
    if model_type == 'normal-shallow':
        model = UT_SegmentationNet_Normal_shallow(num_class=4).to(device)
        if opt.split('-')[0] == 'yz':
            model = UT_SegmentationNet_Normal_shallow(num_class=4, in_channels=3).to(device)
    if model_type == 'tiny':
        model = UT_SegmentationNet_tiny(num_class=4, ).to(device)
        if opt.split('-')[0] == 'yz':
            model = UT_SegmentationNet_tiny(num_class=4, in_channels=3).to(device)
    if model_type == 'tiny-shallow':
        model = UT_SegmentationNet_tiny_shallow(num_class=4, ).to(device)
        if opt.split('-')[0] == 'yz':
            model = UT_SegmentationNet_tiny_shallow(num_class=4, in_channels=3).to(device)
    elif model_type == 'light':
        model = UT_SegmentationNet_light(num_class=4, ).to(device)
        if opt.split('-')[0] == 'yz':
            model = UT_SegmentationNet_light(num_class=4, in_channels=3).to(device)
    elif model_type == 'light-shallow':
        model = UT_SegmentationNet_light_shallow(num_class=4, ).to(device)
        if opt.split('-')[0] == 'yz':
            model = UT_SegmentationNet_light_shallow(num_class=4, in_channels=3).to(device)
    elif model_type == 'light-deep':
        model = UT_SegmentationNet_light_deep(num_class=4, ).to(device)
        if opt.split('-')[0] == 'yz':
            model = UT_SegmentationNet_light_deep(num_class=4, in_channels=3).to(device)

    if summary:
        print(model_type)
        if opt.split('-')[0] == 'yz':
            sm(model, (3, 512, 61))
        else:
            sm(model, (2, 512, 61))
        return None

    model.load_state_dict(torch.load(savedir + '//SegUNet_Best_'+ model_type + '_' + opt + '.pth'))
    with open(savedir + '//SegUNet_'+model_type+'_'+opt+"_loss_rec.json", 'r') as f:
        loss_rec = json.load(f)

    evaluator = accuracy_evaluation(n_class=3)
    evaluator.plot_loss(loss_rec, title=opt+'_'+model_type, save=True)

    iou = []
    dice = []
    PA = []
    time_consumption = 0
    for j, (images, labels, fname) in enumerate(valid_dataloader):
        # 前向传播
        now = time.perf_counter()
        images = images.to(device)
        outputs = model(images)
        # for i in range(10):
        #     outputs = model(images)
        time_consumption += (time.perf_counter() - now)
        _, predicted = torch.max(outputs.data, dim=1)
        tmp_pa, tmp_iou, tmp_dice = evaluator.evaluate_all(outputs, labels.to(device), 0)
        tmp_init_img = 128 * (images[0, -1, :, :].detach().cpu().numpy() + 1.0)
        mask_img = gen_mask_demo_img(tmp_init_img.astype('uint8'), mask=predicted[0, :, :].cpu().detach().numpy())
        mask_img_gt = gen_mask_demo_img(tmp_init_img.astype('uint8'), mask=labels[0, :, :].cpu().detach().numpy())
        if save_pic:
            cv2.imwrite(os.path.join(dataset_dir, 'ValidImgs', fname[0].split('\\')[-1][:-4]+'_U-'+model_type+'_'+opt+'.bmp'), mask_img)
            cv2.imwrite(os.path.join(dataset_dir, 'ValidImgs', fname[0].split('\\')[-1][:-4]+'_GT'+'.bmp'), mask_img_gt)
        # evaluator.show_seg_img(title=str(mode)+'_'+opt+'_'+model_type+fname[0].split('\\')[-1].split('.')[-2], save=True)

        PA.append(tmp_pa)
        iou.append(tmp_iou)
        dice.append(tmp_dice)

    print('Model: {}, \tDevice: {}, \tMode: {}, \tTime Consumption: {:.5f}, \tV_PA={:.4f}, \tV_IOU={:.4f}, \tV_Dice={:.4f}'.format(
        model_type.ljust(15), device, opt.ljust(10), time_consumption / (j+1),
        np.nanmean(np.array(PA)),
        np.nanmean(np.array(iou)),
        np.nanmean(np.array(dice))))


def eval_TransUNet(opt='ndep', model_type='normal', device='cuda', savedir = 'params', dataset_dir=r'E:\Data\Experiments\UT_202311\DATASET\B_5', summary=False, save_pic=False):
    valid_dataloader = GenerateDataloader(dataset_dir + r'\Segmentation_'+opt, type='Valid', model='seg', opt=opt, batch_size=1, mode=0, shuffle=False)
    if model_type == 'normal':
        model = UT_SegTransUNet_Normal(in_channels=2, num_class=4).to(device)
    elif model_type == 'light':
        model = UT_SegTransUNet_light(in_channels=2, num_class=4).to(device)
    elif model_type == 'light-LNF':
        model = UT_SegTransUNet_light(in_channels=2, num_class=4, LN=False).to(device)
    elif model_type == 'light-yz':
        model = UT_SegTransUNet_light(in_channels=3, num_class=4).to(device)
    elif model_type == 'light-hidden256':
        model = UT_SegTransUNet_light(in_channels=2, num_class=4, hidden_size=256).to(device)
    elif model_type == 'light-sparse':
        model = UT_SegTransUNet_light(in_channels=2, num_class=4, grid_size=(16, 4)).to(device)
    elif model_type == 'light-sparse-yz':
        model = UT_SegTransUNet_light(in_channels=3, num_class=4, grid_size=(16, 4)).to(device)
    elif model_type == 'light-ultra-sparse':
        model = UT_SegTransUNet_light(in_channels=2, num_class=4, grid_size=(8, 2)).to(device)
    elif model_type == 'tiny':
        model = UT_SegTransUNet_tiny(in_channels=2, num_class=4).to(device)
    elif model_type == 'ultra-tiny':
        model = UT_SegTransUNet_utiny(in_channels=2, num_class=4, hidden_size=64).to(device)
    elif model_type == 'ultra-tiny-sparse':
        model = UT_SegTransUNet_utiny(in_channels=2, num_class=4, hidden_size=64, grid_size=(16, 4)).to(device)
    elif model_type == 'ultra-tiny-ultra-sparse':
        model = UT_SegTransUNet_utiny(in_channels=2, num_class=4, hidden_size=64, grid_size=(8, 2)).to(device)

    model.load_state_dict(torch.load(savedir + '//SegTransUNet_Best_'+ model_type+ '_' + opt + '.pth'))
    if summary:
        print(model_type)
        if model_type == 'light-yz':
            sm(model, (3, 512, 61))
        else:
            sm(model, (2, 512, 61))
        return None

    with open(savedir + '//SegTransUNet_'+model_type+ '_' + opt+"_loss_rec.json", 'r') as f:
        loss_rec = json.load(f)

    evaluator = accuracy_evaluation(n_class=3)
    evaluator.plot_loss(loss_rec, title='TransUNet'+'_'+model_type, save=True)

    iou = []
    dice = []
    PA = []
    time_consumption = 0
    for j, (images, labels, fname) in enumerate(valid_dataloader):
        # 前向传播
        now = time.perf_counter()
        images = images.to(device)
        outputs = model(images)
        # for i in range(10):
        #     outputs = model(images)
        time_consumption += (time.perf_counter() - now)
        _, predicted = torch.max(outputs.data, dim=1)
        tmp_pa, tmp_iou, tmp_dice = evaluator.evaluate_all(outputs, labels.to(device), mode=0)
        tmp_init_img = 128 * (images[0, -1, :, :].detach().cpu().numpy() + 1.0)
        mask_img = gen_mask_demo_img(tmp_init_img.astype('uint8'), mask=predicted[0, :, :].cpu().detach().numpy())
        if save_pic:
            cv2.imwrite(os.path.join(dataset_dir, 'ValidImgs', fname[0].split('\\')[-1][:-4]+'_T-'+model_type+'_'+opt+'.bmp'), mask_img)
        # evaluator.show_seg_img(title=str(mode)+'_'+opt+'_'+model_type+fname[0].split('\\')[-1].split('.')[-2], save=True)
        PA.append(tmp_pa)
        iou.append(tmp_iou)
        dice.append(tmp_dice)

    print('Model: {}, \tDevice: {}, \tMode: {}, \tTime Consumption: {:.5f}, \tV_PA={:.4f}, \tV_IOU={:.4f}, \tV_Dice={:.4f}'.format(
        ('TUN-' + model_type).ljust(15), device, opt.ljust(10), time_consumption / (j + 1),
        np.nanmean(np.array(PA)),
        np.nanmean(np.array(iou)),
        np.nanmean(np.array(dice))))



if __name__ == '__main__':
    dataset_dir = r'E:\Data\Experiments\UT_202311\DATASET\B_4'
    params_dir = r'E:\Data\Experiments\UT_202311\DATASET\B_4\params'
    num_epoch = 500
    device = 'cpu'
    save_pic = True

    # # TransUNet 的几种不同 参数量级和网格 type
    # train_TransUNet(opt='ndep', model_type='normal', savedir=params_dir, dataset_dir=dataset_dir, num_epochs=num_epoch)
    train_TransUNet(opt='ndep', model_type='light', savedir=params_dir, dataset_dir=dataset_dir, num_epochs=num_epoch)
    # train_TransUNet(opt='ndep', model_type='light-hidden256', savedir=params_dir, dataset_dir=dataset_dir, num_epochs=num_epoch)
    # train_TransUNet(opt='ndep', model_type='light-sparse', savedir=params_dir, dataset_dir=dataset_dir, num_epochs=num_epoch)

    train_TransUNet(opt='ndep', model_type='light-ultra-sparse', savedir=params_dir, dataset_dir=dataset_dir, num_epochs=num_epoch)
    train_TransUNet(opt='z-sin', model_type='light-sparse', savedir=params_dir, dataset_dir=dataset_dir, num_epochs=num_epoch)
    train_TransUNet(opt='z-linear', model_type='light-sparse', savedir=params_dir, dataset_dir=dataset_dir, num_epochs=num_epoch)
    train_TransUNet(opt='yz-lsin', model_type='light-sparse-yz', savedir=params_dir, dataset_dir=dataset_dir, num_epochs=num_epoch)
    train_TransUNet(opt='yz-sin', model_type='light-sparse-yz', savedir=params_dir, dataset_dir=dataset_dir, num_epochs=num_epoch)

    # train_TransUNet(opt='ndep', model_type='tiny', savedir=params_dir, dataset_dir=dataset_dir, num_epochs=num_epoch)
    # train_TransUNet(opt='ndep', model_type='ultra-tiny', savedir=params_dir, dataset_dir=dataset_dir, num_epochs=num_epoch)

    # train_TransUNet(opt='ndep', model_type='ultra-tiny-sparse', savedir=params_dir, dataset_dir=dataset_dir, num_epochs=num_epoch)
    # train_TransUNet(opt='ndep', model_type='ultra-tiny-ultra-sparse', savedir=params_dir, dataset_dir=dataset_dir, num_epochs=num_epoch)

    # # UNet 的几种不同深度
    # train(opt='ndep', model_type='light-shallow', savedir=params_dir, dataset_dir=dataset_dir, num_epochs=num_epoch)
    # train(opt='ndep', model_type='light', savedir=params_dir, dataset_dir=dataset_dir, num_epochs=num_epoch)
    # train(opt='ndep', model_type='light-deep', savedir=params_dir, dataset_dir=dataset_dir, num_epochs=num_epoch)
    #
    # # UNet 的几种不同深度编码
    # train(opt='z-sin', model_type='light-shallow', savedir=params_dir, dataset_dir=dataset_dir, num_epochs=num_epoch)
    # train(opt='z-linear', model_type='light-shallow', savedir=params_dir, dataset_dir=dataset_dir, num_epochs=num_epoch)
    # train(opt='yz-lsin', model_type='light-shallow', savedir=params_dir, dataset_dir=dataset_dir, num_epochs=num_epoch)
    # train(opt='yz-sin', model_type='light-shallow', savedir=params_dir, dataset_dir=dataset_dir, num_epochs=num_epoch)
    # train(opt='ndep', model_type='light', savedir=params_dir, dataset_dir=dataset_dir, num_epochs=num_epoch)

    # train(opt='z-sin', model_type='light', savedir=params_dir, dataset_dir=dataset_dir, num_epochs=num_epoch)
    # train(opt='z-linear', model_type='light', savedir=params_dir, dataset_dir=dataset_dir, num_epochs=num_epoch)
    # train(opt='yz-lsin', model_type='light', savedir=params_dir, dataset_dir=dataset_dir, num_epochs=num_epoch)
    # train(opt='yz-sin', model_type='light', savedir=params_dir, dataset_dir=dataset_dir, num_epochs=num_epoch)
    #
    # train(opt='z-sin', model_type='light-deep', savedir=params_dir, dataset_dir=dataset_dir, num_epochs=num_epoch)
    # train(opt='z-linear', model_type='light-deep', savedir=params_dir, dataset_dir=dataset_dir, num_epochs=num_epoch)
    # train(opt='yz-lsin', model_type='light-deep', savedir=params_dir, dataset_dir=dataset_dir, num_epochs=num_epoch)
    # train(opt='yz-sin', model_type='light-deep', savedir=params_dir, dataset_dir=dataset_dir, num_epochs=num_epoch)
    #
    # # UNet 的几个不同大小的模型
    # train(opt='ndep', model_type='tiny-shallow', savedir=params_dir, dataset_dir=dataset_dir, num_epochs=num_epoch)
    train(opt='ndep', model_type='normal-shallow', savedir=params_dir, dataset_dir=dataset_dir, num_epochs=num_epoch)


    #
    # # 评价模型效果
    # print('******\tUNET\t 模型深度\t******')
    # eval(opt='ndep', model_type='light-shallow', savedir=params_dir, dataset_dir=dataset_dir, device=device, save_pic=save_pic)
    # eval(opt='ndep', model_type='light', savedir=params_dir, dataset_dir=dataset_dir, device=device, save_pic=save_pic)
    # eval(opt='ndep', model_type='light-deep', savedir=params_dir, dataset_dir=dataset_dir, device=device, save_pic=save_pic)
    #
    # print('******\t UNET \t 不同嵌入位置编码方式\t******')
    # eval(opt='ndep', model_type='light', savedir=params_dir, dataset_dir=dataset_dir, device=device, save_pic=save_pic)
    # eval(opt='z-sin', model_type='light', savedir=params_dir, dataset_dir=dataset_dir, device=device, save_pic=save_pic)
    # eval(opt='z-linear', model_type='light', savedir=params_dir, dataset_dir=dataset_dir, device=device, save_pic=save_pic)
    # eval(opt='yz-lsin', model_type='light', savedir=params_dir, dataset_dir=dataset_dir, device=device, save_pic=save_pic)
    # eval(opt='yz-sin', model_type='light', savedir=params_dir, dataset_dir=dataset_dir, device=device, save_pic=save_pic)
    #
    # print('******\t UNET - shallow \t 不同嵌入位置编码方式\t******')
    # eval(opt='ndep', model_type='light-shallow', savedir=params_dir, dataset_dir=dataset_dir, device=device, save_pic=save_pic)
    # eval(opt='z-sin', model_type='light-shallow', savedir=params_dir, dataset_dir=dataset_dir, device=device, save_pic=save_pic)
    # eval(opt='z-linear', model_type='light-shallow', savedir=params_dir, dataset_dir=dataset_dir, device=device, save_pic=save_pic)
    # eval(opt='yz-lsin', model_type='light-shallow', savedir=params_dir, dataset_dir=dataset_dir, device=device, save_pic=save_pic)
    # eval(opt='yz-sin', model_type='light-shallow', savedir=params_dir, dataset_dir=dataset_dir, device=device, save_pic=save_pic)
    #
    # print('******\t UNET - deep \t 不同嵌入位置编码方式\t******')
    # eval(opt='ndep', model_type='light-deep', savedir=params_dir, dataset_dir=dataset_dir, device=device, save_pic=save_pic)
    # eval(opt='z-sin', model_type='light-deep', savedir=params_dir, dataset_dir=dataset_dir, device=device, save_pic=save_pic)
    # eval(opt='z-linear', model_type='light-deep', savedir=params_dir, dataset_dir=dataset_dir, device=device, save_pic=save_pic)
    # eval(opt='yz-lsin', model_type='light-deep', savedir=params_dir, dataset_dir=dataset_dir, device=device, save_pic=save_pic)
    # eval(opt='yz-sin', model_type='light-deep', savedir=params_dir, dataset_dir=dataset_dir, device=device, save_pic=save_pic)
    #
    # print('******\t UNET \t 不同参数量\t******')
    # eval(opt='ndep', model_type='light-shallow', savedir=params_dir, dataset_dir=dataset_dir, device=device, save_pic=save_pic)
    # eval(opt='ndep', model_type='tiny-shallow', savedir=params_dir, dataset_dir=dataset_dir, device=device, save_pic=save_pic)
    eval(opt='ndep', model_type='normal-shallow', savedir=params_dir, dataset_dir=dataset_dir, device=device, save_pic=save_pic)
    eval(opt='ndep', model_type='normal-shallow', savedir=params_dir, dataset_dir=dataset_dir, device=device, save_pic=save_pic)


    # print('******\t TransUNet \t 网格密度影响')
    eval_TransUNet(opt='ndep', model_type='light', savedir=params_dir, dataset_dir=dataset_dir, device=device, save_pic=save_pic)
    # eval_TransUNet(opt='ndep', model_type='light-sparse', savedir=params_dir, dataset_dir=dataset_dir, device=device, save_pic=save_pic)
    # eval_TransUNet(opt='ndep', model_type='light-ultra-sparse', savedir=params_dir, dataset_dir=dataset_dir, device=device)


    # print('******\t TransUNet \t 嵌入位置编码影响')
    # eval_TransUNet(opt='ndep', model_type='light-sparse', savedir=params_dir, dataset_dir=dataset_dir, device=device, save_pic=save_pic)
    # eval_TransUNet(opt='z-sin', model_type='light-sparse', savedir=params_dir, dataset_dir=dataset_dir, device=device, save_pic=save_pic)
    # eval_TransUNet(opt='z-linear', model_type='light-sparse', savedir=params_dir, dataset_dir=dataset_dir, device=device, save_pic=save_pic)
    # eval_TransUNet(opt='yz-lsin', model_type='light-sparse-yz', savedir=params_dir, dataset_dir=dataset_dir, device=device, save_pic=save_pic)
    # eval_TransUNet(opt='yz-sin', model_type='light-sparse-yz', savedir=params_dir, dataset_dir=dataset_dir, device=device, save_pic=save_pic)


    # print('******\t UNet \t 网络参数summary\t******')
    # eval(opt='ndep', model_type='light-shallow', savedir=params_dir, dataset_dir=dataset_dir, summary=True)
    # eval(opt='ndep', model_type='light', savedir=params_dir, dataset_dir=dataset_dir, summary=True)
    # eval(opt='ndep', model_type='light-deep', savedir=params_dir, dataset_dir=dataset_dir, summary=True)
    # eval(opt='ndep', model_type='tiny-shallow', savedir=params_dir, dataset_dir=dataset_dir, summary=True)
    eval(opt='ndep', model_type='normal-shallow', savedir=params_dir, dataset_dir=dataset_dir, summary=True)
    #
    # print('******\t TransUNet \t 网络参数summary\t******')
    # eval_TransUNet(opt='ndep', model_type='light', savedir=params_dir, dataset_dir=dataset_dir, summary=True)
    # eval_TransUNet(opt='ndep', model_type='tiny', savedir=params_dir, dataset_dir=dataset_dir, summary=True)
    # eval_TransUNet(opt='ndep', model_type='ultra-tiny', savedir=params_dir, dataset_dir=dataset_dir, summary=True)
    # eval_TransUNet(opt='ndep', model_type='light-sparse', savedir=params_dir, dataset_dir=dataset_dir, summary=True)
    # eval_TransUNet(opt='ndep', model_type='light-ultra-sparse', savedir=params_dir, dataset_dir=dataset_dir, summary=True)
    eval_TransUNet(opt='ndep', model_type='light', savedir=params_dir, dataset_dir=dataset_dir, summary=True)


'''不用的'''
    # eval_TransUNet(opt='ndep', model_type='ultra-tiny', savedir=params_dir, dataset_dir=dataset_dir, device=device)
    # eval_TransUNet(opt='ndep', model_type='ultra-tiny-sparse', savedir=params_dir, dataset_dir=dataset_dir, device=device)
    # eval_TransUNet(opt='ndep', model_type='ultra-tiny-ultra-sparse', savedir=params_dir, dataset_dir=dataset_dir, device=device)
    # print('******\t TransUNet \t 不使用Layer Normalization')
    # train_TransUNet(opt='ndep', model_type='light-LNF', savedir=params_dir, dataset_dir=dataset_dir, num_epochs=num_epoch)
    # eval_TransUNet(opt='ndep', model_type='light-LNF', savedir=params_dir, dataset_dir=dataset_dir, device=device)

    # print('******\t TransUNet \t 不同参数量\t******')
    # eval_TransUNet(opt='ndep', model_type='light', savedir=params_dir, dataset_dir=dataset_dir, device=device, save_pic=save_pic)
    # eval_TransUNet(opt='ndep', model_type='tiny', savedir=params_dir, dataset_dir=dataset_dir, device=device, save_pic=save_pic)
    # eval_TransUNet(opt='ndep', model_type='ultra-tiny', savedir=params_dir, dataset_dir=dataset_dir, device=device)
    # print('******\t TransUNet \t Hidden Size影响')
    # eval_TransUNet(opt='ndep', model_type='light', savedir=params_dir, dataset_dir=dataset_dir, device=device)
    # eval_TransUNet(opt='ndep', model_type='light-hidden256', savedir=params_dir, dataset_dir=dataset_dir, device=device)
    # train(opt='ndep', model_type='tiny', savedir=params_dir, dataset_dir=dataset_dir, num_epochs=num_epoch)
    # train(opt='ndep', model_type='normal', savedir=params_dir, dataset_dir=dataset_dir, num_epochs=num_epoch)