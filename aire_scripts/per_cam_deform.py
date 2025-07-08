from deform_dataloader import ViewDataset
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np
from deform_net import DeltaMLP
import sys
import torch
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
import wandb
from tqdm import tqdm

VALID_SIZE = 0.2
BATCH_SIZE = 1
SEED = 0

def main():
    
    wandb.init(project="deform_per_view", name="lr_1e-4", notes="Init LR: 1e-4, Arch=MLP, Losses:MSE, Split Vids")
    device='cuda'

    data = ViewDataset("/mnt/scratch/scasag/input/")

    _, _, train_vids, test_vids= train_test_split(
      range(len(set(data.videos))),
      list(set(data.videos)),
      test_size=VALID_SIZE,
      random_state=SEED
    )
    print("Test Vids: ", test_vids)
    
    train_indices = [idx for idx, v in enumerate(data.videos) if v not in test_vids]
    valid_indices = [idx for idx, v in enumerate(data.videos) if v in test_vids]
    
    train_split = Subset(data, train_indices)
    valid_split = Subset(data, valid_indices)
    train_batches = DataLoader(train_split, batch_size=BATCH_SIZE, shuffle=True)
    valid_batches = DataLoader(valid_split, batch_size=BATCH_SIZE)
    print("Valid and Train Sizes: ", len(valid_batches), len(train_batches))

    init_lr = 1e-4
    model = DeltaMLP(aud_in=1707, emo_in=1280).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=init_lr)
    loss_fn = torch.nn.MSELoss()
    
    scheduler = ReduceLROnPlateau(
      opt, mode='min', factor=0.8,
      patience=5, threshold=1e-5,
    )
    
    wandb.watch(model, log='all')

    n_epochs = 10
    epoch_progress_bar = tqdm(range(n_epochs), desc="Epoch progress")

    for epoch in range(n_epochs):

        losses = []
        model.train()
        batch_progress_bar = tqdm(range(len(train_batches)), desc="Batch progress")
        
        for batch_num, input_data in enumerate(train_batches):

            outputs = model(input_data['aud_feats'].cuda(), input_data['emo_feats'].cuda())
            loss = loss_fn(input_data['delta_xyz'], outputs['deform_xyz']) + loss_fn(input_data['delta_sca'], outputs['deform_sca']) \
                   + loss_fn(input_data['delta_opa'], outputs['deform_opa']) + loss_fn(input_data['delta_rot'], outputs['deform_rot']) \
                   + loss_fn(input_data['delta_ftr'], outputs['deform_ftr'])
            losses.append(loss.item())
            loss.backward()
            opt.step()

            batch_progress_bar.set_postfix({
                  'Epoch': epoch,
                  'Batch': batch_num,
                  'Loss': loss.item(),
            })
            wandb.log({
                  'batch/batch_num': batch_num,
                  'batch/loss': loss.item(),
            })
            batch_progress_bar.update(1)
            opt.zero_grad()
            torch.cuda.empty_cache()
        
        batch_progress_bar.close()

        valid_losses = []
        model.eval()
        with torch.no_grad():
            for batch_num, input_data in enumerate(valid_batches):

                outputs = model(input_data['aud_feats'].cuda(), input_data['emo_feats'].cuda())
                loss = loss_fn(input_data['delta_xyz'], outputs['deform_xyz']) + loss_fn(input_data['delta_sca'], outputs['deform_sca']) \
                    + loss_fn(input_data['delta_opa'], outputs['deform_opa']) + loss_fn(input_data['delta_rot'], outputs['deform_rot']) \
                    + loss_fn(input_data['delta_ftr'], outputs['deform_ftr'])
                valid_losses.append(loss.item())
        
        epoch_progress_bar.set_postfix({
          'Epoch': epoch,
          'Train Loss': sum(losses)/len(losses),
          'Valid Loss': sum(valid_losses)/len(valid_losses),
        })
        wandb.log({
          'epoch/epoch': epoch,
          'epoch/train_loss': sum(losses)/len(losses),
          'epoch/val_loss': sum(valid_losses)/len(valid_losses),
          'epoch/learning_rate': scheduler.get_last_lr()[0],
        })

        scheduler.step(sum(valid_losses)/len(valid_losses))
        epoch_progress_bar.update(1)
        
        if epoch == n_epochs-1:
          epoch_progress_bar.close()
        
        '''
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': loss,
            'scheduler': scheduler.state_dict(),}, 
            f"/mnt/scratch/scasag/deform_model/per_view_model.pth", _use_new_zipfile_serialization=False)
        '''

if __name__ == "__main__":
    main()