import argparse
import os
import numpy as np
import random
import torch
import torch.optim
import torch.utils.data
import csv  # 添加 csv 模块
from torch.utils.tensorboard import SummaryWriter
from BraTS import get_datasets
from models.model import CKD
from models import DataAugmenter
from utils import mkdir, save_best_model, save_seg_csv, cal_dice, cal_confuse, save_test_label, AverageMeter, save_checkpoint
from torch.backends import cudnn
from monai.metrics.hausdorff_distance import HausdorffDistanceMetric
from monai.metrics.meandice import DiceMetric
from monai.losses.dice import DiceLoss
from monai.inferers import sliding_window_inference

parser = argparse.ArgumentParser(description='BraTS')
parser.add_argument('--exp-name', default="CKD", type=str)
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--dataset-folder',default="", type=str, help="Please reference the README file for the detailed dataset structure.")
parser.add_argument('--workers', default=1, type=int, help="The value of CPU's num_worker")
parser.add_argument('--end-epoch', default=500, type=int, help="Maximum iterations of the model")
parser.add_argument('--batch-size', default=1, type=int)
parser.add_argument('--lr', default=1e-4, type=float) 
parser.add_argument('--devices', default=0, type=int)
parser.add_argument('--resume', default=False, type=bool)
parser.add_argument('--tta', default=True, type=bool, help="test time augmentation")
parser.add_argument('--seed', default=1)
parser.add_argument('--val', default=1, type=int, help="Validation frequency of the model")


def init_randon(seed):
    torch.manual_seed(seed)        
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed) 
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    cudnn.benchmark = False         
    cudnn.deterministic = True

def init_folder(args):
    args.base_folder =  mkdir(os.path.dirname(os.path.realpath(__file__)))
    args.dataset_folder = mkdir(os.path.join(args.base_folder, args.dataset_folder))
    args.best_folder = mkdir(f"{args.base_folder}/best_model/{args.exp_name}")
    args.writer_folder = mkdir(f"{args.base_folder}/writer/{args.exp_name}")
    args.pred_folder = mkdir(f"{args.base_folder}/pred/{args.exp_name}")
    args.checkpoint_folder = mkdir(f"{args.base_folder}/checkpoint/{args.exp_name}")
    args.csv_folder = mkdir(f"{args.base_folder}/csv/{args.exp_name}")
    print(f"The code folder are located in {os.path.dirname(os.path.realpath(__file__))}")
    print(f"The dataset folder located in {args.dataset_folder}")


def main(args):  
    writer = SummaryWriter(args.writer_folder)
    model = CKD(embed_dim=32, output_dim=3, img_size=(128, 128, 128), patch_size=(4, 4, 4), in_chans=1, depths=[2, 2, 2], num_heads=[2, 4, 8, 16], window_size=(7, 7, 7), mlp_ratio=4.).cuda()
    criterion = DiceLoss(sigmoid=True).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5, amsgrad=True)

    if args.mode == "train":
        train_dataset = get_datasets(args.dataset_folder, "train")
        train_val_dataset = get_datasets(args.dataset_folder, "train_val")
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
        train_val_loader = torch.utils.data.DataLoader(train_val_dataset, batch_size=1, shuffle=False, num_workers=args.workers)
        train_manager(args, train_loader, train_val_loader, model, criterion, optimizer, writer)
    
    elif args.mode == "test":
        print("start test")
        model.load_state_dict(torch.load(os.path.join(args.best_folder, "best_model.pkl")))
        model.eval()
        test_dataset = get_datasets(args.dataset_folder, "test")
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.workers)
        test(args, "test", test_loader, model, writer)  # 确保传递 writer 参数
   

def train_manager(args, train_loader, train_val_loader, model, criterion, optimizer, writer):
    best_loss = np.inf
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.end_epoch, eta_min=1e-5)
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(os.path.join(args.checkpoint_folder, "checkpoint.pth.tar"))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        scheduler.load_state_dict(checkpoint['scheduler'])
    print(f"start train from epoch = {start_epoch}")
    
    for epoch in range(start_epoch, args.end_epoch):
        model.train()
        writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)
        train_loss = train(train_loader, model, criterion, optimizer, scheduler, epoch, writer)
        if (epoch + 1) % args.val == 0:
            model.eval()
            with torch.no_grad():
                train_val_loss = train_val(train_val_loader, model, criterion, epoch, writer)
                if train_val_loss < best_loss:
                    best_loss = train_val_loss
                    save_best_model(args, model)
        save_checkpoint(args, dict(epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict()))
        print(f"epoch = {epoch}, train_loss = {train_loss}, train_val_loss = {train_val_loss}, best_loss = {best_loss}")
    
    print("finish train epoch")

def train(data_loader, model, criterion, optimizer, scheduler, epoch, writer):
    # 初始化权重
    channel_weights = torch.ones(3, device='cuda')

    train_loss_meter = AverageMeter('Loss', ':.4e')
    wt_loss_meter = AverageMeter('WT_Loss', ':.4e')
    et_loss_meter = AverageMeter('ET_Loss', ':.4e')
    tc_loss_meter = AverageMeter('TC_Loss', ':.4e')

    wt_weight_meter = AverageMeter('WT_Weight', ':.4e')
    et_weight_meter = AverageMeter('ET_Weight', ':.4e')
    tc_weight_meter = AverageMeter('TC_Weight', ':.4e')

    epoch_channel_losses = []  # 用于记录整个 epoch 的通道损失
    epoch_channel_weights = []  # 用于记录整个 epoch 的通道权重

    for i, data in enumerate(data_loader):
        torch.cuda.empty_cache()
        data_aug = DataAugmenter().cuda()
        label = data["label"].cuda()
        images = data["image"].cuda()
        images, label = data_aug(images, label)
        pred = model(images)
        # 分别计算每个通道的损失
        channel_losses = [criterion(pred[:, c:c+1], label[:, c:c+1]) for c in range(pred.shape[1])]
        # 根据公式动态调整权重
        total_loss = sum(channel_losses)
        n = len(channel_losses)
        channel_weights = torch.tensor(
            [(l / total_loss * n).item() for l in channel_losses],
            device='cuda'
        )
        # 使用权重对通道损失进行加权求和
        weighted_loss = sum(w * l for w, l in zip(channel_weights, channel_losses))

        # 记录当前 batch 的通道损失和权重
        epoch_channel_losses.append([l.item() for l in channel_losses])
        epoch_channel_weights.append(channel_weights.tolist())

        train_loss_meter.update(weighted_loss.item())
        wt_loss_meter.update(channel_losses[2].item())  # WT
        et_loss_meter.update(channel_losses[0].item())  # ET
        tc_loss_meter.update(channel_losses[1].item())  # TC

        wt_weight_meter.update(channel_weights[2].item())  # WT
        et_weight_meter.update(channel_weights[0].item())  # ET
        tc_weight_meter.update(channel_weights[1].item())  # TC

        # 反向传播
        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()
    scheduler.step()
    torch.cuda.empty_cache()

    # 打印整个 epoch 的通道损失和权重
    avg_channel_losses = np.mean(epoch_channel_losses, axis=0)
    avg_channel_weights = np.mean(epoch_channel_weights, axis=0)
    print(f"Epoch {epoch}, Average Channel Losses: {avg_channel_losses}, Average Weights: {avg_channel_weights}")

    # 记录到 TensorBoard
    writer.add_scalar("loss/train", train_loss_meter.avg, epoch)
    writer.add_scalar("loss/WT", wt_loss_meter.avg, epoch)
    writer.add_scalar("loss/ET", et_loss_meter.avg, epoch)
    writer.add_scalar("loss/TC", tc_loss_meter.avg, epoch)

    writer.add_scalar("weights/WT", wt_weight_meter.avg, epoch)
    writer.add_scalar("weights/ET", et_weight_meter.avg, epoch)
    writer.add_scalar("weights/TC", tc_weight_meter.avg, epoch)
    return train_loss_meter.avg

def train_val(data_loader, model, criterion, epoch, writer):
    train_val_loss_meter = AverageMeter('Loss', ':.4e')
    for i, data in enumerate(data_loader):
        label = data["label"].cuda()
        images = data["image"].cuda()
        pred = model(images)
        train_val_loss = criterion(pred, label)
        train_val_loss_meter.update(train_val_loss.item())
    writer.add_scalar("loss/train_val", train_val_loss_meter.avg, epoch)
    return train_val_loss_meter.avg

def inference(model, input, batch_size, overlap):
    def _compute(input):
        return sliding_window_inference(inputs=input, roi_size=(128, 128, 128), sw_batch_size=batch_size, predictor=model, overlap=overlap)
    return _compute(input)


def test(args, mode, data_loader, model, writer):
    metrics_dict = []

    for i, data in enumerate(data_loader):
        patient_id = data["patient_id"][0]
        inputs = data["image"]
        targets = data["label"].cuda()
        pad_list = data["pad_list"]
        nonzero_indexes = data["nonzero_indexes"]
        inputs = inputs.cuda()
        model.cuda()
        with torch.no_grad():  
            if args.tta:
                predict = torch.sigmoid(inference(model, inputs, batch_size=2, overlap=0.6))
                # 翻转 dim=2 并恢复
                augmented_input = inputs.flip(dims=(2,))
                augmented_pred = inference(model, augmented_input, batch_size=2, overlap=0.6)
                predict += torch.sigmoid(augmented_pred.flip(dims=(2,)))
                # 翻转 dim=3 并恢复
                augmented_input = inputs.flip(dims=(3,))
                augmented_pred = inference(model, augmented_input, batch_size=2, overlap=0.6)
                predict += torch.sigmoid(augmented_pred.flip(dims=(3,)))
                # 翻转 dim=4 并恢复
                augmented_input = inputs.flip(dims=(4,))
                augmented_pred = inference(model, augmented_input, batch_size=2, overlap=0.6)
                predict += torch.sigmoid(augmented_pred.flip(dims=(4,)))
                # 翻转 dim=(2,3) 并恢复
                augmented_input = inputs.flip(dims=(2,3))
                augmented_pred = inference(model, augmented_input, batch_size=2, overlap=0.6)
                predict += torch.sigmoid(augmented_pred.flip(dims=(2,3)))
                # 翻转 dim=(2,4) 并恢复
                augmented_input = inputs.flip(dims=(2,4))
                augmented_pred = inference(model, augmented_input, batch_size=2, overlap=0.6)
                predict += torch.sigmoid(augmented_pred.flip(dims=(2,4)))
                # 翻转 dim=(3,4) 并恢复
                augmented_input = inputs.flip(dims=(3,4))
                augmented_pred = inference(model, augmented_input, batch_size=2, overlap=0.6)
                predict += torch.sigmoid(augmented_pred.flip(dims=(3,4)))
                # 翻转 dim=(2,3,4) 并恢复
                augmented_input = inputs.flip(dims=(2,3,4))
                augmented_pred = inference(model, augmented_input, batch_size=2, overlap=0.6)
                predict += torch.sigmoid(augmented_pred.flip(dims=(2,3,4)))
                # 平均所有增强结果
                predict = predict / 8.0 
            else:
                predict = torch.sigmoid(inference(model, inputs, batch_size=2, overlap=0.6))
                
        targets = targets[:, :, pad_list[-4]:targets.shape[2]-pad_list[-3], pad_list[-6]:targets.shape[3]-pad_list[-5], pad_list[-8]:targets.shape[4]-pad_list[-7]]
        predict = predict[:, :, pad_list[-4]:predict.shape[2]-pad_list[-3], pad_list[-6]:predict.shape[3]-pad_list[-5], pad_list[-8]:predict.shape[4]-pad_list[-7]]
        predict = (predict>0.5).squeeze()
        targets = targets.squeeze()
        dice_metrics = cal_dice(predict, targets, patient_id)
        confuse_metric = cal_confuse(predict, targets, patient_id)
        # 正确提取每个指标的Dice和HD数值
        et_dice = dice_metrics[0]["DICE"]  # ET类的Dice
        et_hd = dice_metrics[0]["HAUSSDORF"]  # ET类的HD

        tc_dice = dice_metrics[1]["DICE"]  # TC类的Dice
        tc_hd = dice_metrics[1]["HAUSSDORF"]  # TC类的HD

        wt_dice = dice_metrics[2]["DICE"]  # WT类的Dice
        wt_hd = dice_metrics[2]["HAUSSDORF"]  # WT类的HD
        et_sens, tc_sens, wt_sens = confuse_metric[0][0], confuse_metric[1][0], confuse_metric[2][0]
        et_spec, tc_spec, wt_spec = confuse_metric[0][1], confuse_metric[1][1], confuse_metric[2][1]
        # 将所有指标转换为 Python 原生类型
        metrics_dict.append(dict(
            id=patient_id,
            et_dice=float(et_dice), tc_dice=float(tc_dice), wt_dice=float(wt_dice), 
            et_hd=float(et_hd), tc_hd=float(tc_hd), wt_hd=float(wt_hd),
            et_sens=float(et_sens), tc_sens=float(tc_sens), wt_sens=float(wt_sens),
            et_spec=float(et_spec), tc_spec=float(tc_spec), wt_spec=float(wt_spec),
        ))
        full_predict = np.zeros((155, 240, 240))
        predict = reconstruct_label(predict)
        full_predict[slice(*nonzero_indexes[0]), slice(*nonzero_indexes[1]), slice(*nonzero_indexes[2])] = predict
        save_test_label(args, mode, patient_id, full_predict)
        save_seg_csv(args, f"csv/{args.exp_name}", metrics_dict)
  
def reconstruct_label(image):
    if type(image) == torch.Tensor:
        image = image.cpu().numpy()
    c1, c2, c3 = image[0], image[1], image[2]
    image = (c3 > 0).astype(np.uint8)
    image[(c2 == False)*(c3 == True)] = 2
    image[(c1 == True)*(c3 == True)] = 4
    return image

if __name__=='__main__':
    args=parser.parse_args()
    for arg in vars(args):
        print(format(arg, '<20'), format(str(getattr(args, arg)), '<'))
    if torch.cuda.device_count() == 0:
        raise RuntimeWarning("Can not run without GPUs")
    init_randon(args.seed)
    init_folder(args)
    torch.cuda.set_device(args.devices)
    main(args)



