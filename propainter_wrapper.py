import os
import sys
import torch
import torchvision
import numpy as np
import scipy.ndimage
import cv2
import imageio
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import tempfile
import platform
from loguru import logger

# Add ProPainter to sys.path so we can import its modules
propainter_path = os.path.join(os.path.dirname(__file__), "third_party", "ProPainter")
if propainter_path not in sys.path:
    sys.path.insert(0, propainter_path)

from model.modules.flow_comp_raft import RAFT_bi
from model.recurrent_flow_completion import RecurrentFlowCompleteNet
from model.propainter import InpaintGenerator
from utils.download_util import load_file_from_url
from core.utils import to_tensors
from model.misc import get_device

# OS Detection and Hardware Setup
# Force MPS fallback if on Apple Silicon to handle unsupported correlation layers
if platform.system() == "Darwin" and torch.backends.mps.is_available():
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    logger.info("Apple Silicon detected: Enabling MPS fallback for ProPainter.")

def load_propainter(device):
    """Downloads weights and loads the ProPainter model components."""
    pretrain_model_url = 'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/'
    weights_dir = os.path.join(propainter_path, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    
    logger.info("Loading ProPainter models (downloading weights if missing)...")
    
    # 1. RAFT Flow model
    ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'raft-things.pth'), 
                                    model_dir=weights_dir, progress=True, file_name=None)
    fix_raft = RAFT_bi(ckpt_path, device)
    
    # 2. Flow Completion Network
    ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'recurrent_flow_completion.pth'), 
                                    model_dir=weights_dir, progress=True, file_name=None)
    fix_flow_complete = RecurrentFlowCompleteNet(ckpt_path)
    for p in fix_flow_complete.parameters():
        p.requires_grad = False
    fix_flow_complete.to(device)
    fix_flow_complete.eval()
    
    # 3. ProPainter Network
    ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'ProPainter.pth'), 
                                    model_dir=weights_dir, progress=True, file_name=None)
    model = InpaintGenerator(model_path=ckpt_path).to(device)
    model.eval()
    
    return fix_raft, fix_flow_complete, model

def get_ref_index(mid_neighbor_id, neighbor_ids, length, ref_stride=10, ref_num=-1):
    ref_index = []
    if ref_num == -1:
        for i in range(0, length, ref_stride):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
        end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
        for i in range(start_idx, end_idx, ref_stride):
            if i not in neighbor_ids:
                if len(ref_index) > ref_num:
                    break
                ref_index.append(i)
    return ref_index

def binary_mask(mask, th=0.1):
    mask[mask>th] = 1
    mask[mask<=th] = 0
    return mask

def process_video_propainter(frames_pil, masks_pil, device):
    """
    Main logic to pass frames and masks through ProPainter.
    frames_pil: List of PIL images.
    masks_pil: List of PIL masks (L mode).
    """
    use_half = True if device.type == 'cuda' else False
    
    size = frames_pil[0].size
    process_size = (size[0]-size[0]%8, size[1]-size[1]%8)
    if not size == process_size:
        frames_pil = [f.resize(process_size) for f in frames_pil]
        masks_pil = [m.resize(process_size, Image.NEAREST) for m in masks_pil]
    else:
        out_size = size
        
    flow_masks = []
    masks_dilated = []
    
    for mask_img in masks_pil:
        mask_arr = np.array(mask_img.convert('L'))
        
        # Dilate 4 pixels
        flow_mask_img = scipy.ndimage.binary_dilation(mask_arr, iterations=4).astype(np.uint8)
        flow_masks.append(Image.fromarray(flow_mask_img * 255))
        
        mask_dilated = scipy.ndimage.binary_dilation(mask_arr, iterations=4).astype(np.uint8)
        masks_dilated.append(Image.fromarray(mask_dilated * 255))
        
    frames_inp = [np.array(f).astype(np.uint8) for f in frames_pil]
    frames = to_tensors()(frames_pil).unsqueeze(0) * 2 - 1    
    flow_masks = to_tensors()(flow_masks).unsqueeze(0)
    masks_dilated = to_tensors()(masks_dilated).unsqueeze(0)
    frames, flow_masks, masks_dilated = frames.to(device), flow_masks.to(device), masks_dilated.to(device)
    
    fix_raft, fix_flow_complete, model = load_propainter(device)
    
    video_length = frames.size(1)
    
    with torch.no_grad():
        if frames.size(-1) <= 640: 
            short_clip_len = 12
        elif frames.size(-1) <= 720: 
            short_clip_len = 8
        elif frames.size(-1) <= 1280:
            short_clip_len = 4
        else:
            short_clip_len = 2
            
        if frames.size(1) > short_clip_len:
            gt_flows_f_list, gt_flows_b_list = [], []
            for f in range(0, video_length, short_clip_len):
                end_f = min(video_length, f + short_clip_len)
                if f == 0:
                    flows_f, flows_b = fix_raft(frames[:,f:end_f], iters=20)
                else:
                    flows_f, flows_b = fix_raft(frames[:,f-1:end_f], iters=20)
                
                gt_flows_f_list.append(flows_f)
                gt_flows_b_list.append(flows_b)
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                
            gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
            gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
            gt_flows_bi = (gt_flows_f, gt_flows_b)
        else:
            gt_flows_bi = fix_raft(frames, iters=20)
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        if use_half:
            frames, flow_masks, masks_dilated = frames.half(), flow_masks.half(), masks_dilated.half()
            gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())
            fix_flow_complete = fix_flow_complete.half()
            model = model.half()

        subvideo_length = 80
        flow_length = gt_flows_bi[0].size(1)
        if flow_length > subvideo_length:
            pred_flows_f, pred_flows_b = [], []
            pad_len = 5
            for f in range(0, flow_length, subvideo_length):
                s_f = max(0, f - pad_len)
                e_f = min(flow_length, f + subvideo_length + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(flow_length, f + subvideo_length)
                pred_flows_bi_sub, _ = fix_flow_complete.forward_bidirect_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]), 
                    flow_masks[:, s_f:e_f+1])
                pred_flows_bi_sub = fix_flow_complete.combine_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]), 
                    pred_flows_bi_sub, 
                    flow_masks[:, s_f:e_f+1])

                pred_flows_f.append(pred_flows_bi_sub[0][:, pad_len_s:e_f-s_f-pad_len_e])
                pred_flows_b.append(pred_flows_bi_sub[1][:, pad_len_s:e_f-s_f-pad_len_e])
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                
            pred_flows_f = torch.cat(pred_flows_f, dim=1)
            pred_flows_b = torch.cat(pred_flows_b, dim=1)
            pred_flows_bi = (pred_flows_f, pred_flows_b)
        else:
            pred_flows_bi, _ = fix_flow_complete.forward_bidirect_flow(gt_flows_bi, flow_masks)
            pred_flows_bi = fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks)
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        masked_frames = frames * (1 - masks_dilated)
        subvideo_length_img_prop = min(100, subvideo_length)
        b, t, _, h, w = masks_dilated.size()
        
        if video_length > subvideo_length_img_prop:
            updated_frames, updated_masks = [], []
            pad_len = 10
            for f in range(0, video_length, subvideo_length_img_prop):
                s_f = max(0, f - pad_len)
                e_f = min(video_length, f + subvideo_length_img_prop + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(video_length, f + subvideo_length_img_prop)

                pred_flows_bi_sub = (pred_flows_bi[0][:, s_f:e_f-1], pred_flows_bi[1][:, s_f:e_f-1])
                prop_imgs_sub, updated_local_masks_sub = model.img_propagation(masked_frames[:, s_f:e_f], 
                                                                       pred_flows_bi_sub, 
                                                                       masks_dilated[:, s_f:e_f], 
                                                                       'nearest')
                updated_frames_sub = frames[:, s_f:e_f] * (1 - masks_dilated[:, s_f:e_f]) + \
                                    prop_imgs_sub.view(b, e_f-s_f, 3, h, w) * masks_dilated[:, s_f:e_f]
                updated_masks_sub = updated_local_masks_sub.view(b, e_f-s_f, 1, h, w)
                
                updated_frames.append(updated_frames_sub[:, pad_len_s:e_f-s_f-pad_len_e])
                updated_masks.append(updated_masks_sub[:, pad_len_s:e_f-s_f-pad_len_e])
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                
            updated_frames = torch.cat(updated_frames, dim=1)
            updated_masks = torch.cat(updated_masks, dim=1)
        else:
            prop_imgs, updated_local_masks = model.img_propagation(masked_frames, pred_flows_bi, masks_dilated, 'nearest')
            updated_frames = frames * (1 - masks_dilated) + prop_imgs.view(b, t, 3, h, w) * masks_dilated
            updated_masks = updated_local_masks.view(b, t, 1, h, w)
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    ori_frames = frames_inp
    comp_frames = [None] * video_length

    neighbor_stride = 10 // 2
    if video_length > subvideo_length:
        ref_num = subvideo_length // 10
    else:
        ref_num = -1
    
    for f in tqdm(range(0, video_length, neighbor_stride), desc="HQ Inpainting"):
        neighbor_ids = [
            i for i in range(max(0, f - neighbor_stride),
                                min(video_length, f + neighbor_stride + 1))
        ]
        ref_ids = get_ref_index(f, neighbor_ids, video_length, 10, ref_num)
        selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :]
        selected_masks = masks_dilated[:, neighbor_ids + ref_ids, :, :, :]
        selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :]
        selected_pred_flows_bi = (pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :], pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :])
        
        with torch.no_grad():
            l_t = len(neighbor_ids)
            pred_img = model(selected_imgs, selected_pred_flows_bi, selected_masks, selected_update_masks, l_t)
            pred_img = pred_img.view(-1, 3, h, w)
            pred_img = (pred_img + 1) / 2
            pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
            binary_masks = masks_dilated[0, neighbor_ids, :, :, :].cpu().permute(
                0, 2, 3, 1).numpy().astype(np.uint8)
            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                img = np.array(pred_img[i]).astype(np.uint8) * binary_masks[i] \
                    + ori_frames[idx] * (1 - binary_masks[i])
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else: 
                    comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
                comp_frames[idx] = comp_frames[idx].astype(np.uint8)
        
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Resize back if needed
    if process_size != size:
        comp_frames = [cv2.resize(f, size, interpolation=cv2.INTER_CUBIC) for f in comp_frames]
        
    return comp_frames
