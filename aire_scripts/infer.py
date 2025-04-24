from deform_dataloader import CustomDataset
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
from scene import GaussianModel, Scene
from gaussian_renderer import render
from arguments import PipelineParams
from arguments import ModelParams
from argparse import ArgumentParser
import sys
import torch
from deform_net import MLP
from utils.loss_utils import l1_loss
from utils.image_utils import psnr
from lpipsPyTorch.modules.lpips import LPIPS
from gaussian_renderer import render_motion

BATCH_SIZE = 1
SEED = 0

def main():
    np.random.seed(SEED)
    device = 'cuda'

    data = CustomDataset()

    gpath = "C:/Users/arpit/Downloads/M005/M005"
    colmap_path = "C:/Users/arpit/Desktop/gs_data/all_videos/M005"
    sys.argv = f"x --source_path {colmap_path} -r 2".split()
    parser = ArgumentParser(description="Training script parameters")
    dataset = ModelParams(parser)
    args = parser.parse_args(sys.argv[1:])
    temp = dataset.extract(args)
    gaussians = GaussianModel(sh_degree=3)
    scene = Scene(temp, gaussians)
    front_cam = [cam for cam in scene.getTrainCameras() if cam.image_name.startswith('front')]
    pipe = PipelineParams(parser)
    gaussians.load_ply("C:/Users/arpit/Desktop/M005_no_densify/point_cloud/iteration_30000/point_cloud.ply")

    model = MLP(n_gaussians=4085, n_features=11, aud_in=1707, emo_in=1280).to('cuda')
    checkpoint = torch.load("", weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)
    lpips_obj = LPIPS(net_type='vgg').to(device)

    for video in tqdm(data.videos):
        test_indices = [v for v in data.videos if v in video]
        test_indices = [i for i, x in enumerate(data.videos) if x == video]
        
        test_split = Subset(data, test_indices)  
        test_batches = DataLoader(test_split, batch_size=BATCH_SIZE)

        t_psnrs, t_images, t_l1s, t_lpips = [], [], [], []

        with torch.no_grad():
            for batch_num, input_data in enumerate(tqdm(test_batches)):
                gt_image, aud, emo, _ = input_data
                gt_image = gt_image[0].to(device)
                aud = aud.to(device)
                emo = emo.to(device)
                output = model(aud, emo)
                
                output_image = render_motion(front_cam[0],
                   gaussians, pipe, background,
                   deform_output=output.reshape(4085, -1), 
                   scaling_modifier=1.0, 
                   use_trained_exp=temp.train_test_exp)["render"]
                
                t_images.append(output_image)
                t_psnr = psnr(output_image, gt_image).mean().double()
                t_psnrs.append(t_psnr.item())
                t_l1s.append(l1_loss(output_image, gt_image).item())
                t_lpips.append(lpips_obj(output_image, gt_image).item())
        
        print(video, round(sum(t_psnrs)/len(t_psnrs), 2))
        
        if sum(t_psnrs)/len(t_psnrs) > 37:
            t_images = [x.permute(1, 2, 0).cpu().detach().numpy() for x in t_images]
            np.save("", np.array(t_images))

if __name__ == "__main__":
    main()