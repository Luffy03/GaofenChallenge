import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import shutil
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import OrderedDict
from options import *
from utils import *
from dataset import *
from models import *
from torch import optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from sync_batchnorm import convert_model
print("PyTorch Version: ", torch.__version__)
print('cuda', torch.version.cuda)
os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1, 2"
# device = torch.device("cuda:0")
# print('Device:', device)


def cal_loss(out_x, out_y, out_cd, label1, label2, label_cd):
    criterion = F1_BCE_loss()
    # criterion = Dice_BCE_loss()

    loss1 = criterion(out_x, label1.float())
    loss2 = criterion(out_y, label2.float())

    # change part
    loss_cd = criterion(out_cd, label_cd.float())

    loss_all = loss1 + loss2 + loss_cd * 2
    return loss_all


def main():
    cudnn.enabled = True
    cudnn.benchmark = True

    opt = Base_Options().parse()
    log_path, checkpoint_path, predict_path, _ = create_save_path(opt)
    train_txt_path, val_txt_path, whole_txt_path = create_data_path(opt)

    # Define logger
    logger, tensorboard_log_dir = create_logger(log_path)
    logger.info(opt)
    # Define Transformation
    train_transform = transforms.Compose([
        trans.RandomHorizontalFlip(),
        trans.RandomVerticleFlip(),
        trans.RandomRotate90()
    ])

    # load train and val dataset
    train_dataset = Aug_Dataset(opt, train_txt_path, flag='train', transform=None)
    val_dataset = CD_Dataset(opt, whole_txt_path, flag='val', transform=None)

    # Create training and validation dataloaders
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                              num_workers=opt.num_workers, pin_memory=opt.pin)

    model = Net_fpn(opt.backbone)
    model = nn.DataParallel(model, device_ids=[0, 1, 2])

    model = model.to(device)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.AdamW(model.parameters(), lr=opt.base_lr, amsgrad=True)

    # scheduler = CosineAnnealingLR(optimizer, T_max=opt.num_epochs, eta_min=0)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    # patience = 3
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=patience, min_lr=4e-5,
    #                               verbose=True)

    best_metric = -1e16

    for epoch in range(opt.num_epochs):
        time_start = time.time()

        train(opt, epoch, model, train_loader, optimizer, logger)
        # for cosine
        # scheduler.step()

        metric = validate(opt, epoch, model, val_dataset, logger)
        # scheduler.step(metric)

        logger.info('best_val_metric:%.4f current_val_metric:%.4f' % (best_metric, metric))
        if metric > best_metric:
            logger.info('epoch:%d Save to model_best' % (epoch))
            torch.save({'state_dict': model.module.state_dict()},
                       os.path.join(checkpoint_path, 'model_best.pth'), _use_new_zipfile_serialization=False)
            best_metric = metric
        if epoch == 80:
            torch.save({'state_dict': model.module.state_dict()},
                       os.path.join(checkpoint_path, 'model_best_epoch80.pth'),
                       _use_new_zipfile_serialization=False)

        if epoch == 120:
            torch.save({'state_dict': model.module.state_dict()},
                       os.path.join(checkpoint_path, 'model_best_epoch100.pth'),
                       _use_new_zipfile_serialization=False)

        end_time = time.time()
        time_cost = end_time - time_start
        logger.info("Epoch %d Time %d ----------------------" % (epoch, time_cost))
        logger.info('\n')


def train(opt, epoch, model, loader, optimizer, logger):
    model.train()
    loss_m = AverageMeter()

    last_idx = len(loader) - 1

    for batch_idx, batch in enumerate(loader):
        step = epoch * len(loader) + batch_idx
        adjust_learning_rate(opt.base_lr, optimizer, step, len(loader), num_epochs=40)
        lr = get_lr(optimizer)

        last_batch = batch_idx == last_idx
        img1, img2, label1, label2, change, name = batch
        img1, img2, label1, label2, change = img1.cuda(non_blocking=True), \
                                             img2.cuda(non_blocking=True), \
                                             label1.cuda(non_blocking=True), \
                                             label2.cuda(non_blocking=True), \
                                             change.cuda(non_blocking=True),
        if opt.mixup is True:
            img1, img2, label1, label2, change = mixup([img1, img2, label1, label2, change])

        out_x, out_y, out_cd = model.forward(img1, img2)
        loss = cal_loss(out_x, out_y, out_cd, label1, label2, change)
        loss_m.update(loss.item(), img1.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log_interval = len(loader) // 3
        if last_batch or batch_idx % log_interval == 0:
            logger.info('Train:{} [{:>4d}/{} ({:>3.0f}%)] '
                    'Loss:({loss.avg:>6.4f}) '
                    'LR:{lr:.3e} '.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=loss_m,
                        lr=lr))

    return OrderedDict([('loss', loss_m.avg)])


def validate(opt, epoch, model, val_dataset, logger):
    compute_metric = IOUMetric(num_classes=2)
    hist1 = np.zeros([2, 2])
    hist2 = np.zeros([2, 2])
    hist_cd = np.zeros([2, 2])

    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, pin_memory=opt.pin)
    model.eval()

    for batch_idx, batch in tqdm(enumerate(val_loader)):
        img1, img2, label1, label2, change, name = batch
        with torch.no_grad():
            img1, img2 = img1.cuda(), img2.cuda()
            x, y, out = model(img1, img2)

        x, y, out = (x > 0.5).long(), (y > 0.5).long(), (out > 0.5).long()

        x, y, out = x.data.cpu().numpy(), \
                    y.data.cpu().numpy(), \
                    out.data.cpu().numpy()
        label1, label2, change = label1.data.cpu().numpy(), \
                                 label2.data.cpu().numpy(), \
                                 change.data.cpu().numpy()

        hist1 += compute_metric.get_hist(x, label1)
        hist2 += compute_metric.get_hist(y, label2)
        hist_cd += compute_metric.get_hist(out, change)

    f_score1, f_score2, f_score_cd = cal_fscore(hist1), cal_fscore(hist2), cal_fscore(hist_cd)
    f_score = (f_score1 + f_score2) * 0.2 + f_score_cd * 0.6
    logger.info('epoch:%d f_score1:%s f_score2:%s f_score_cd:%s current_fscore:%s'
                % (epoch, f_score1, f_score2, f_score_cd, f_score))

    return f_score


if __name__ == '__main__':
   main()