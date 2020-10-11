import os
import sys
import cv2
import subprocess

import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor, Normalize

from pathlib import Path
from tempfile import TemporaryDirectory
from natsort import natsorted
from PIL import Image

from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode

# local src code
from logger import setup_logger
from real_time_video_mosaic import VideMosaic


logger = setup_logger(__name__)

video_paths = list(Path('./data/raw_videos').glob('*.m4v'))

class ImageDataset(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image, np.array(image)


# rename all files with spaces to '_'
for i, video_path in enumerate(video_paths):
    new_name = video_path.parent.joinpath(video_path.name.replace(' ', '_'))
    video_path.rename(new_name)

with TemporaryDirectory() as td:
    
    for video_path in video_paths:
        logger.info(video_path)
        outpath = Path(td).joinpath(video_path.name)
        outpath.mkdir()

        #-----------------
        # Video to Frames TODO: To Function?
        #-----------------
        cmd = f"ffmpeg -i {str(video_path)} {str(outpath)+'/%06d.png'}"
        logger.info(cmd)
        subprocess.run(cmd.split())
        frame_paths = sorted(list(outpath.glob("*.png")))

        #-----------------------------
        # Human Segmentation / Masking
        #-----------------------------
        # Load models
        net_encoder = ModelBuilder.build_encoder(
            arch='hrnetv2',
            fc_dim=720,
            weights='ckpt/ade20k-hrnetv2-c1/encoder_epoch_30.pth')
        net_decoder = ModelBuilder.build_decoder(
            arch='c1',
            fc_dim=720,
            num_class=150,
            weights='ckpt/ade20k-hrnetv2-c1/decoder_epoch_30.pth',
            use_softmax=True)
        crit = torch.nn.NLLLoss(ignore_index=-1)
        segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
        segmentation_module.eval()
        segmentation_module.cuda()

        # Create image dataset
        transform= Compose([
            ToTensor(),
            Normalize(
                mean=[0.485, 0.456, 0.406], # These are RGB mean+std values
                std=[0.229, 0.224, 0.225])  # across a large photo dataset.
            ])
        dataset = ImageDataset(outpath, transform=transform)
        batch_size = 1
        dataloader = DataLoader(dataset, batch_size=batch_size)

        # save dir
        seg_dir = Path(str(outpath)+"_seg")
        seg_dir.mkdir()

        for batch_idx, batch in enumerate(dataloader):
            tensor_batch, orig_batch = batch
            output_size = tensor_batch.shape[2:]
            with torch.no_grad():
                scores = segmentation_module({'img_data':tensor_batch.cuda()}, segSize=output_size)
            
            # Get the predicted scores for each pixel
            _, pred = torch.max(scores, dim=1)
            for im_idx, (im, pd) in enumerate(zip(orig_batch, pred)):
                pd = pd.cpu().numpy()
                im = im.cpu().numpy()

                mask_idx = (pd==12) # 12 is human class label
                im[mask_idx] = 0
                logger.debug(pd.shape,im.shape, mask_idx.shape)
                save_path = str(seg_dir.joinpath(f"{batch_idx*batch_size+im_idx:06}.png"))
                logger.debug(save_path)
                Image.fromarray(im).save(save_path)



        #-------------------
        # Build Video Mosaic
        #-------------------
        # Build full mosaic on first loop
        mosaic_dir = Path(str(outpath)+"_mosaic")
        mosaic_dir.mkdir()
        for loop_idx in range(2):
            for frame_idx, frame_path in enumerate(frame_paths):
                frame = cv2.imread(str(frame_path))
                if frame_idx==0 and loop_idx==0:
                    video_mosaic = VideMosaic(frame, detector_type="sift")

                # process each frame
                video_mosaic.process_frame(frame)

                # if loop_idx == 1:
                save_path = str(mosaic_dir.joinpath(f"{frame_idx:06}.png"))
                logger.debug(save_path)
                cv2.imwrite(save_path, video_mosaic.output_img)

        #-----------------
        # Frames to Video TODO: To Function?
        #-----------------
        cmd = f"ffmpeg -framerate 60 -i {str(mosaic_dir)+'/%06d.png'}  {'./data/'+video_path.name+'_mosaic.mp4'}"
        logger.info(cmd)
        subprocess.run(cmd.split())
        
        break
