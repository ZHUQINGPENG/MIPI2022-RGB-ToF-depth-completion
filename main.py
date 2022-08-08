import os
import time
import torch
import numpy as np
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from dataload.dataload import DataLoad
from loss.loss import Loss
from metric.metric import Metric
from summary.summary import Summary
from config import args
from model.mobileunetv2 import mobileunetv2
import utility
from tqdm import tqdm

def train(args,gpu=0):
    torch.cuda.set_device(gpu)

    ### Load data
    data_train = DataLoad(args, 'train')
    data_val = DataLoad(args, 'val')

    sampler_train = DistributedSampler(
        data_train, num_replicas=args.num_gpus, rank=gpu)
    sampler_val = DistributedSampler(
        data_val, num_replicas=args.num_gpus, rank=gpu)

    batch_size = args.batch_size // args.num_gpus

    loader_train = DataLoader(
        dataset=data_train, batch_size=batch_size, shuffle=False,
        num_workers=args.num_threads, pin_memory=True, sampler=sampler_train,
        drop_last=True)
    loader_val = DataLoader(
        dataset=data_val, batch_size=args.num_summary, shuffle=False,
        num_workers=args.num_threads, pin_memory=True, sampler=sampler_val,
        drop_last=False)

    ### Set up Network
    net = mobileunetv2(args)
    net.cuda()

    if args.pretrain is not None:
        assert os.path.exists(args.pretrain), \
            "file not found: {}".format(args.pretrain)

        checkpoint = torch.load(args.pretrain)
        if gpu==0:
            net.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['net'].items()}, strict=False)
            print('Load network parameters from : {}'.format(args.pretrain))

    ### Set up Loss
    loss = Loss(args)

    ### Set up Optimizer
    optimizer, scheduler = utility.make_optimizer_scheduler(args, net)

    ### Set up Metric
    metric = Metric(args)

    if gpu == 0:
        utility.backup_source_code(args.save_dir + '/code')
        try:
            os.makedirs(args.save_dir, exist_ok=True)
            os.makedirs(args.save_dir + '/train', exist_ok=True)
            os.makedirs(args.save_dir + '/val', exist_ok=True)
        except OSError:
            pass

        writer_train = Summary(args.save_dir, 'train', args,loss.loss_name, metric.metric_name)
        writer_val = Summary(args.save_dir, 'val', args, loss.loss_name, metric.metric_name)

    if args.warm_up:
        warm_up_cnt = 0.0
        warm_up_max_cnt = len(loader_train)+1.0

    ### Training
    for epoch in range(args.epochs+1):
        net.train()
        sampler_train.set_epoch(epoch)

        num_sample = len(loader_train) * loader_train.batch_size * args.num_gpus

        if gpu == 0:
            pbar = tqdm(total=num_sample)
            log_cnt = 0.0
            log_loss = 0.0

        for batch, sample in enumerate(loader_train):
            sample = {key: val.cuda(gpu) for key, val in sample.items() if val is not None}
            
            if epoch == 0 and args.warm_up:
                warm_up_cnt += 1

                for param_group in optimizer.param_groups:
                    lr_warm_up = param_group['initial_lr'] * warm_up_cnt / warm_up_max_cnt
                    param_group['lr'] = lr_warm_up

            optimizer.zero_grad()

            output = net(sample)

            loss_sum, loss_val = loss(sample, output)

            # Divide by batch size
            loss_sum = loss_sum / loader_train.batch_size
            loss_val = loss_val / loader_train.batch_size

            loss_sum.backward()

            optimizer.step()

            if gpu == 0:
                metric_val = metric.evaluate(sample['gt'], output)
                writer_train.add(loss_val, metric_val)

                log_cnt += 1
                log_loss += loss_sum.item()

                current_time = time.strftime('%y%m%d@%H:%M:%S')
                error_str = '{:<10s}| {} | Loss1 = {:.4f}'.format(
                    'Train', current_time, log_loss / log_cnt)

                if epoch == 1 and args.warm_up:
                    list_lr = []
                    for g in optimizer.param_groups:
                        list_lr.append(round(g['lr'], 6))
                    error_str = '{} | Lr Warm Up : {}'.format(error_str,list_lr)

                pbar.set_description(error_str)
                pbar.update(loader_train.batch_size * args.num_gpus)

        if gpu == 0:
            pbar.close()

            writer_train.update(epoch, sample, output)
            
            if args.save_full or epoch == args.epochs:
                state = {
                    'net': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'args': args
                }
            else:
                state = {
                    'net': net.state_dict(),
                    'args': args
                }
            torch.save(state, f'{args.save_dir}/model_{epoch:05d}.pt')
        while(not os.path.exists('{}/model_{:05d}.pt'.format(args.save_dir, epoch))):
            #print('waiting for model')
            pass
        if gpu==0:
            checkpoint = torch.load('{}/model_{:05d}.pt'.format(args.save_dir, epoch))
            net.load_state_dict(checkpoint['net'], strict=False)
        
        ### Validation
        torch.set_grad_enabled(False)
        net.eval()
        num_sample = len(loader_val) * loader_val.batch_size * args.num_gpus

        if gpu == 0:
            pbar = tqdm(total=num_sample)
            log_cnt = 0.0
            log_loss = 0.0

        for batch, sample in enumerate(loader_val):
            sample = {key: val.cuda(gpu) for key, val in sample.items() if val is not None}
            output = net(sample)
            loss_sum, loss_val = loss(sample, output)

            loss_sum = loss_sum / loader_val.batch_size
            loss_val = loss_val / loader_val.batch_size
            if gpu == 0:
                metric_val = metric.evaluate(sample['gt'], output)
                writer_val.add(loss_val, metric_val)

                log_cnt += 1
                log_loss += loss_sum.item()

                current_time = time.strftime('%y%m%d@%H:%M:%S')
                error_str = '{:<10s}| {} | Loss = {:.4f} | epoch {}'.format(
                    'Val', current_time, log_loss / log_cnt, epoch)
                pbar.set_description(error_str)
                #print(error_str)
                pbar.update(loader_val.batch_size * args.num_gpus)

        if gpu == 0:
            pbar.close()
            writer_val.update(epoch, sample, output)
            writer_val.save(epoch, batch, sample, output)

        torch.set_grad_enabled(True)

        scheduler.step()

def test(args):
    ### Prepare dataset
    data_test = DataLoad(args, 'test')

    loader_test = DataLoader(dataset=data_test, batch_size=1, shuffle=False)

    ###  Set up Network
    net = mobileunetv2(args)
    net.cuda()

    if args.pretrain is not None:
        assert os.path.exists(args.pretrain), \
            "file not found: {}".format(args.pretrain)

        checkpoint = torch.load(args.pretrain)
        net.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['net'].items()},strict=False)
        # net.load_state_dict(checkpoint['net'], strict=True)

    net.eval()

    num_sample = len(loader_test)*loader_test.batch_size

    output_path = os.path.join('../results', args.test_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    flist = []
    pbar = tqdm(total=num_sample)
    import cv2
    from matplotlib import pyplot as plt
    for batch, sample in enumerate(loader_test):
        sample = {key: val.cuda() if val is not None and not key == 'name' else val for key, val in sample.items()}

        output = net(sample)

        pbar.set_description('Testing')
        pbar.update(loader_test.batch_size)

        output_image = output.detach().cpu().numpy()
        
        def visualize(img,name):
            plt.imshow(img,cmap='jet_r')
            plt.axis('off')
            plt.colorbar()
            plt.savefig(name)
            plt.close()

        for i in range(output_image.shape[0]):
            cv2.imwrite(os.path.join(output_path, f'{batch}.exr'), (output_image[i, 0, :, :]).astype(np.float32))
            # visualize(output_image[i, 0, :, :],os.path.join(output_path, f'{batch}.png'))

            flist.append(f'{batch}.exr')    

        with open(f'{output_path}/data.list', 'w') as f:
            for item in flist:
                f.write("%s\n" % item)

    pbar.close()



if __name__ == "__main__":
    if args.test_only:
        test(args)
    else:
        train(args)

        

