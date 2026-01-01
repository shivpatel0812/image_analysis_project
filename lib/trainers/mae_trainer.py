import os
import math
import torch
import torchvision
import torchvision.transforms as transforms
import sys
sys.path.append('..')

import lib.models as models
import lib.networks as networks
from lib.datasets import ImageListDataset

import wandb
from .base_trainer import BaseTrainer

class MAETrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = 'MAE'
        self.scaler = torch.cuda.amp.GradScaler(enabled=(args.gpu is not None))

    def build_model(self):
        if self.model_name != 'Unknown' and self.model is None:
            args = self.args
            print(f"=> creating model {self.model_name} of arch {args.arch}")
            self.model = getattr(models, self.model_name)(
                encoder=getattr(networks, args.enc_arch),
                decoder=getattr(networks, args.dec_arch),
                args=args
            )
            self.wrap_model()
        elif self.model_name == 'Unknown':
            raise ValueError("=> Model name is still unknown")
        else:
            raise ValueError("=> Model has been created. Do not create twice")

    def build_optimizer(self):
        assert self.model is not None and self.wrapped_model is not None
        print("=> creating optimizer")
        args = self.args
        optim_params = self.get_parameter_groups()
        self.optimizer = torch.optim.AdamW(
            optim_params,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay
        )

    def _normalize(self):
        args = self.args
        in_chans = int(getattr(args, 'in_chans', 3))
        if in_chans == 1:
            return transforms.Normalize(mean=[0.5], std=[0.225])
        if args.mean_std_type == 'IMN':
            return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if args.mean_std_type == 'MED':
            return transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.225, 0.225, 0.225])
        raise ValueError(f"Unsuported mean_std_type {args.mean_std_type}")

    def get_mae_train_augmentation(self):
        args = self.args
        normalize = self._normalize()
        return [
            transforms.RandomResizedCrop(args.input_size, scale=(args.crop_min, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

    def get_mae_val_augmentation(self):
        args = self.args
        normalize = self._normalize()
        return [
            transforms.Resize(int(1.15 * args.input_size)),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            normalize
        ]

    def build_dataloader(self):
        if self.dataloader is not None:
            raise ValueError("Dataloader has been created. Do not create twice.")
        print("=> creating dataloader")
        args = self.args

        augmentation = self.get_mae_train_augmentation()
        train_dataset = ImageListDataset(
            data_root=args.data_path,
            listfile=args.tr_listfile,
            transform=transforms.Compose(augmentation),
            nolabel=True
        )

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        self.dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=(train_sampler is None),
            num_workers=self.workers,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True
        )
        self.iters_per_epoch = len(self.dataloader)

        val_augmentation = self.get_mae_val_augmentation()
        val_dataset = ImageListDataset(
            data_root=args.data_path,
            listfile=args.va_listfile,
            transform=transforms.Compose(val_augmentation),
            nolabel=True
        )

        self.val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.vis_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )

    def run(self):
        args = self.args
        niters = args.start_epoch * self.iters_per_epoch
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                self.dataloader.sampler.set_epoch(epoch)
            niters = self.epoch_train(epoch, niters)
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank == 0):
                if (epoch + 1) % args.vis_freq == 0:
                    self.vis_reconstruction(niters)
                if epoch == 0 or (epoch + 1) % args.save_freq == 0:
                    self.save_checkpoint(
                        {
                            'epoch': epoch + 1,
                            'arch': args.arch,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'scaler': self.scaler.state_dict(),
                        },
                        is_best=False,
                        filename=f'{args.ckpt_dir}/checkpoint_{epoch:04d}.pth.tar'
                    )

    def patches2image(self, patches, color_chans=3, n_group=3):
        B, L, C = patches.shape
        grid_size = int(math.sqrt(L))
        patch_size = int(math.sqrt(C // color_chans))
        image_size = grid_size * patch_size
        patches = patches.reshape(B, grid_size, grid_size, color_chans, patch_size, patch_size)
        image = patches.permute(0, 3, 1, 4, 2, 5).reshape(B, color_chans, image_size, image_size)
        if B % n_group != 0:
            n_group = 1
        n_per_row = B // n_group
        grid_of_images = torchvision.utils.make_grid(image, nrow=n_per_row)
        grid_of_images = grid_of_images.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        return grid_of_images

    def epoch_train(self, epoch, niters):
        args = self.args
        train_loader = self.dataloader
        model = self.wrapped_model
        optimizer = self.optimizer
        scaler = self.scaler

        model.train()

        for i, image in enumerate(train_loader):
            self.adjust_learning_rate(epoch + i / self.iters_per_epoch, args)

            if args.gpu is not None:
                image = image.cuda(args.gpu, non_blocking=True)

            if i == 0 and epoch == 0:
                try:
                    print(f"DEBUG: Image shape: {tuple(image.shape)}, expected channels: {args.in_chans}")
                except Exception:
                    pass

            optimizer.zero_grad(set_to_none=True)

            if args.gpu is not None:
                with torch.amp.autocast('cuda', enabled=True):
                    loss = model(image, return_image=False)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = model(image, return_image=False)
                loss.backward()
                optimizer.step()

            if i % args.print_freq == 0:
                print(
                    f"Epoch: {epoch:03d}/{args.epochs} | "
                    f"Iter: {i:05d}/{self.iters_per_epoch} | "
                    f"TotalIter: {niters:06d} | "
                    f"Init Lr: {self.lr:.05f} | "
                    f"Lr: {optimizer.param_groups[0]['lr']:.05f} | "
                    f"Loss: {loss.item():.03f}"
                )
                if args.rank == 0 and not getattr(args, 'disable_wandb', False):
                    wandb.log(
                        {
                            "lr": optimizer.param_groups[0]['lr'],
                            "Loss": loss.item(),
                        },
                        step=niters,
                    )

            niters += 1

        return niters

    def vis_reconstruction(self, niters=0):
        args = self.args
        loader = self.val_dataloader
        model = self.model
        model.eval()

        for i, image in enumerate(loader):
            if i > 0:
                break
            if args.gpu is not None:
                image = image.cuda(args.gpu, non_blocking=True)

            with torch.no_grad():
                _, x, recon, masked_x = model(image, return_image=True)

            vis_tensor = torch.cat([x, masked_x, recon], dim=0)
            vis_grid = self.patches2image(vis_tensor, color_chans=args.in_chans)

            vis_grid = wandb.Image(vis_grid, caption=f"iter{niters:06d}")
            wandb.log({"vis": vis_grid}, step=niters)

    def resume(self):
        args = self.args
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                loc = f'cuda:{args.gpu}'
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scaler.load_state_dict(checkpoint['scaler'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
