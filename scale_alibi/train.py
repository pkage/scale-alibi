from dataclasses import dataclass, asdict
import os
from pathlib import Path


import torch
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.transforms import Compose
import wandb

from . import console
from .croma.pretrain_croma import CROMA, get_mask
from .dataset.loader import PMTileDataset, RemoveChannels, LoresMultimodalDataset, ChannelsFirstImageOrder




# --- MODELS ---

@dataclass
class CromaParams:
    # probably need to change these for your hardware
    lores_dataset_path: Path
    radar_dataset_path: Path
    batch_size: int = 32
    
    # hyperparams, total guess for now
    mask_ratio: float = 0.4 # mask ratio (ratio of patches in a sequence to keep)
    epochs: int = 10
    learning_rate: float = 1e4

    # probably shouldn't change these
    total_channels: int = 5
    num_patches: int = 256
    patch_size: int = 16


@dataclass
class TrainParams:
    run_name: str
    checkpoint_dir: Path
    device: str
    amp: bool
    nccl_bind: str
    

# --- COMMON ---

def setup_gloo(rank: int, world_size: int) -> dist.HashStore:
    store = dist.HashStore()
    
    # gloo for cpu
    dist.init_process_group(
        backend='gloo',

        store=store,
        world_size=world_size,
        rank=rank
    )
    
    return store


def setup_nccl(rank: int, world_size: int, tcp_bind: str):
    '''
    Setup NCCL process group
    '''
    # tcp_bind = 'tcp://10.1.1.20:23456'
    dist.init_process_group(
        backend='nccl',

        init_method=tcp_bind,
        rank=rank,
        world_size=world_size
    )


def cleanup():
    # Clean up the process group
    dist.destroy_process_group()


# --- CROMA specifics ---

def croma_train(rank: int, world_size: int, croma_params: CromaParams, train_params: TrainParams):
    # dist setup
    setup_nccl(
        rank,
        world_size,
        train_params.nccl_bind
    )

    if rank == 0:
        wandb_project = os.getenv('WANDB_PROJECT')
        if wandb_project is None:
            console.print('[yellow]wandb project is None, this might be a problem!')
        console.print(f'wandb project: {wandb_project}')

        wandb.init(
            project=wandb_project,
            group='croma',

            config={
                **asdict(croma_params),
                **asdict(train_params)
            }
        )
    # pick the device
    device = torch.device(f'cuda:{rank}')


    # load and create the datasets
    lores_dset = PMTileDataset(
        croma_params.lores_dataset_path,
        transform=Compose([
            RemoveChannels([3]), # remove alpha
            ChannelsFirstImageOrder()
        ])
    )
    radar_dset = PMTileDataset(
        croma_params.radar_dataset_path,
        transform=Compose([
            RemoveChannels([0, 3]), # remove alpha and empty red channel
            ChannelsFirstImageOrder()
        ])
    )

    dataset = LoresMultimodalDataset(
        radar_dset,
        lores_dset,
    )

    # sampler and loaders
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=croma_params.batch_size, sampler=sampler, num_workers=4, pin_memory=True)


    # amp maybe?
    scaler = None
    if train_params.amp:
        scaler = GradScaler()

    # create the model
    model = CROMA(
        # ... defaults should be okay
        # except for total_channels, we may need to remove channels from S2,
        total_channels=croma_params.total_channels, # 2 sar, 3 lores_visual
        num_patches=croma_params.num_patches,
        patch_size=croma_params.patch_size
    )
    model.to(device)

    # optimizer
    optimizer = Adam(model.parameters(), lr=croma_params.learning_rate)

    # now we begin!
    for epoch in range(croma_params.epochs):
        console.print(f'beginning epoch {epoch}/{croma_params.epochs} (rank {rank}/{world_size})')
        model.train()

        # make sure each process gets different data
        sampler.set_epoch(epoch)

        total_loss = 0
        for batch_idx, batch in enumerate(loader):
            # get the data masks for the MAE

            seq_len = croma_params.num_patches
            batch_size = batch.radar.shape[0]  # Use the actual batch size
            radar_mask = get_mask(
                batch_size,
                seq_len,
                device,
                croma_params.mask_ratio
            )
            optical_mask = get_mask(
                batch_size,
                seq_len,
                device,
                croma_params.mask_ratio
            )

            radar_imgs = batch.radar.float().to(device)
            lores_imgs = batch.lores.float().to(device)

            optimizer.zero_grad()

            if train_params.amp:
                # AMP pass
                with autocast():
                    contrastive_loss, mae_loss = model(
                        radar_imgs,
                        lores_imgs,
                        radar_mask,
                        optical_mask,
                        rank,
                        world_size
                    )
                    
                    # TODO: optionally balance the two loss components
                    loss = contrastive_loss + mae_loss
                
                assert scaler is not None # not necessary but keeps my editor from yelling at me

                # AMP backward pass
                scaler.scale(loss).backward()

                # AMP optimizer step
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
            else:
                # no AMP training
                contrastive_loss, mae_loss = model(
                    radar_imgs,
                    lores_imgs,
                    radar_mask,
                    optical_mask,
                    rank,
                    world_size
                )
                
                # TODO: optionally balance the two loss components
                loss = contrastive_loss + mae_loss

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if rank == 0 and batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{croma_params.epochs}], Step [{batch_idx+1}/{len(loader)}], Loss: {loss.item():.4f}")
                wandb.log({
                    'epoch': epoch + 1,
                    'batch_idx': batch_idx + 1,
                    "train_loss": loss.item(),
                    "contrastive_loss": contrastive_loss.item(),
                    "mae_loss": mae_loss.item()
                })

        # Save the model checkpoint after each epoch (only rank 0 saves to avoid duplication)
        if rank == 0:
            model_path = train_params.checkpoint_dir / f'croma_checkpoint_{train_params.run_name}_epoch_{epoch}.pth'
            torch.save(model.state_dict(), model_path)
