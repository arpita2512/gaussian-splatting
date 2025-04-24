from deform_dataloader import CustomDataset
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np
from deform_net import MLP
from scene import GaussianModel, Scene
from gaussian_renderer import render
from arguments import PipelineParams, OptimizationParams
from arguments import ModelParams
from argparse import ArgumentParser
import sys
import torch
from utils.loss_utils import l1_loss, ssim

VALID_SIZE = 0.1
BATCH_SIZE = 1
SEED = 0

def main():
    np.random.seed(SEED)
    device = 'cuda'

    data = CustomDataset("C:/Users/arpit/Desktop/M005_front")

    test_vids = ['happy_level_3_014', 'happy_level_2_005', 'happy_level_1_029',
       'happy_level_1_009', 'happy_level_1_028', 'happy_level_3_008',
       'neutral_level_1_024', 'neutral_level_1_026',
       'neutral_level_1_001', 'happy_level_2_002', 'happy_level_3_027',
       'neutral_level_1_023', 'neutral_level_1_016']
    
    train_valid_vids = [v for v in data.videos if v not in test_vids]
    train_indices, valid_indices, _, _ = train_test_split(
        range(len(train_valid_vids)),
        train_valid_vids,
        stratify=train_valid_vids,
        test_size=VALID_SIZE,
        random_state=SEED
    )

    train_split = Subset(data, train_indices)
    valid_split = Subset(data, valid_indices)
    train_batches = DataLoader(train_split, batch_size=BATCH_SIZE, shuffle=True)
    valid_batches = DataLoader(valid_split, batch_size=BATCH_SIZE)

    colmap_path = "C:/Users/arpit/Desktop/gs_data/all_videos/M005"
    sys.argv = f"x --source_path {colmap_path} -r 2".split()
    parser = ArgumentParser(description="Training script parameters")
    dataset = ModelParams(parser)
    args = parser.parse_args(sys.argv[1:])
    temp = dataset.extract(args)
    gaussians = GaussianModel(sh_degree=3, optimizer_type='default')
    scene = Scene(temp, gaussians)
    front_cam = [cam for cam in scene.getTrainCameras() if cam.image_name.startswith('front')]
    pipe = PipelineParams(parser)
    gaussians.load_ply("C:/Users/arpit/Desktop/M005_no_densify/point_cloud/iteration_30000/point_cloud.ply")

    model = MLP(n_gaussians=4085, aud_in=1707, emo_in=1280).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)

    g_xyz = gaussians.get_xyz
    g_rot = gaussians.get_rotation
    g_scaling = gaussians.get_scaling
    g_opacity = gaussians.get_opacity
    g_input = torch.cat((g_xyz, g_rot, g_scaling, g_opacity), dim=1).flatten() 
    g_input = g_input.detach()

    n_epochs = 3
    simulated_batch_size = 100
    bg_color = [1, 1, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    for epoch in range(n_epochs):
        losses = []
        model.train()
        for batch_num, input_data in enumerate(train_batches):
            
            gt_image, aud, emo, _ = input_data
            gt_image = gt_image[0].to(device)
            aud = aud.to(device)
            emo = emo.to(device)
            
            output = model(g_input, aud, emo)

            gaussians._xyz = output.reshape(4085,-1)[:,:3]
            gaussians._rotation = output.reshape(4085,-1)[:,3:7]
            gaussians._scaling = output.reshape(4085,-1)[:,7:10]
            gaussians._opacity = output.reshape(4085,-1)[:,10:]
            output_image = render(front_cam[0],
                    gaussians, pipe, background, 
                    scaling_modifier=1.0, 
                    use_trained_exp=temp.train_test_exp, 
                    separate_sh=False)["render"]
            
            Ll1 = l1_loss(output_image, gt_image)
            ssim_value = ssim(output_image, gt_image)
            loss = (1.0 - 0.2) * Ll1 + 0.2 * (1.0 - ssim_value)
            losses.append(loss.item())

            if batch_num % simulated_batch_size == 0:
                loss.backward()
                opt.step()
                print('\tEpoch %d | Batch %d | Loss %6.2f' % (epoch, batch_num, sum(losses[-simulated_batch_size:])/simulated_batch_size))
                opt.zero_grad()
        
        v_losses = []
        model.eval()
        with torch.no_grad():
            for batch_num, input_data in enumerate(valid_batches):
                gt_image, aud, emo, _ = input_data
                gt_image = gt_image[0].to(device)
                aud = aud.to(device)
                emo = emo.to(device)
                output = model(g_input, aud, emo)
                
                gaussians._xyz = output.reshape(4085,-1)[:,:3]
                gaussians._rotation = output.reshape(4085,-1)[:,3:7]
                gaussians._scaling = output.reshape(4085,-1)[:,7:10]
                gaussians._opacity = output.reshape(4085,-1)[:,10:]
                output_image = render(front_cam[0],
                    gaussians, pipe, background, 
                    scaling_modifier=1.0, 
                    use_trained_exp=temp.train_test_exp, 
                    separate_sh=False)["render"]
                Ll1 = l1_loss(output_image, gt_image)
                ssim_value = ssim(output_image, gt_image)
                loss = (1.0 - 0.2) * Ll1 + 0.2 * (1.0 - ssim_value)
                v_losses.append(loss.item())
            
        print('Epoch %d | Train Loss %6.2f | Val Loss %6.2f' % (epoch, sum(losses)/len(losses), sum(v_losses)/len(v_losses)))
    
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': loss,}, "PATH")
    print("saved model")
    
if __name__ == "__main__":
    main()