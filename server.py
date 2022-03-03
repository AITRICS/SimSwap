import os
from fastapi import *
from fastapi.responses import *

from dotmap import DotMap
import secrets

import cv2
import torch
import fractions
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
from insightface_func.face_detect_crop_single import Face_detect_crop as Face_detect_crop_single
from insightface_func.face_detect_crop_multi import Face_detect_crop as Face_detect_crop_multi
from util.videoswap import video_swap
from glob import glob
import os
import copy

# opt
base_opt = DotMap()
base_opt.name = 'people'
base_opt.gpu_ids = '0'
base_opt.checkpoints_dir = './checkpoints'
base_opt.model = 'pix2pixHD'
base_opt.norm = 'batch'
base_opt.data_type = 32
base_opt.local_rank = 0
base_opt.isTrain = False

base_opt.loadSize = 1024
base_opt.fineSize = 512
base_opt.label_nc = 0
base_opt.input_nc = 3
base_opt.otput_nc = 3

base_opt.netG = 'global'
base_opt.latent_size = 512
base_opt.ngf = 64
base_opt.n_downsample_global = 3
base_opt.n_blocks_local = 3
base_opt.n_blocks_global = 6
base_opt.n_local_enhancers = 1
base_opt.niter_fix_global = 0

base_opt.ntest = float('inf')
base_opt.results_dir = './results/'
base_opt.aspect_ratio = 1.0
base_opt.phase = 'test'
base_opt.which_epoch = 'latest'
base_opt.how_many = 50
base_opt.cluster_path = 'features_clustered_010.npy'
base_opt.Arc_path = 'arcface_model/arcface_checkpoint.tar'
base_opt.multisepcific_dir = './demo_file/multispecific'
base_opt.output_path = './output/'
base_opt.id_thres = 0.03
base_opt.crop_size = 224
base_opt.use_mask = True

torch.nn.Module.dump_patches = True

model = create_model(base_opt)
model.eval()

insightface_app_single = Face_detect_crop_single(name='antelope', root='./insightface_func/models')
insightface_app_single.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640),mode = 'None')

insightface_app_multi = Face_detect_crop_multi(name='antelope', root='./insightface_func/models')
insightface_app_multi.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640),mode = 'None')

app = FastAPI()

transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def preprocess(data):
    # copy from base option
    opt = copy.deepcopy(base_opt)
    infer_id = secrets.token_hex(16)
    os.makedirs(f'temp/{infer_id}/', exist_ok=True)
    os.makedirs(f'temp/{infer_id}/frames', exist_ok=True)
    # write files to temp dir
    for filename in data:
        with open(os.path.join(f'temp/{infer_id}/{filename}'), 'wb') as f:
            f.write(data[filename].file.read())
    # opt setting
    opt.output_path = f'temp/{infer_id}/result.mp4'
    opt.temp_path = f'temp/{infer_id}/frames'
    opt.pic_a_path = f'temp/{infer_id}/source'
    opt.video_path = f'temp/{infer_id}/target'

    return infer_id, opt

@app.post('/test_video_swapmulti')
def test_video_swapmulti(target: UploadFile = File(...), source: UploadFile = File(...)):
    infer_id, opt = preprocess({'target': target, 'source': source})
    pic_a = opt.pic_a_path
    img_a_whole = cv2.imread(pic_a)
    img_a_align_crop, _ = insightface_app_multi.get(img_a_whole, opt.crop_size)
    img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0],cv2.COLOR_BGR2RGB))
    img_a = transformer_Arcface(img_a_align_crop_pil)
    img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2]).cuda()
    img_id_downsample = F.interpolate(img_id, size=(112,112))
    latend_id = model.netArc(img_id_downsample)
    latend_id = F.normalize(latend_id, p=2, dim=1)

    video_swap(opt.video_path, latend_id, model, insightface_app_multi, opt.output_path,temp_results_dir=opt.temp_path,no_simswaplogo=True,use_mask=opt.use_mask,crop_size=opt.crop_size)

    return StreamingResponse(open(f'temp/{infer_id}/result.mp4', mode='rb'), media_type='video/mp4')

@app.post('/test_video_swapsingle')
def test_video_swapsingle(target: UploadFile = File(...), source: UploadFile = File(...)):
    infer_id, opt = preprocess({'target': target, 'source': source})
    pic_a = opt.pic_a_path
    img_a_whole = cv2.imread(pic_a)
    img_a_align_crop, _ = insightface_app_single.get(img_a_whole, opt.crop_size)
    img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0],cv2.COLOR_BGR2RGB))
    img_a = transformer_Arcface(img_a_align_crop_pil)
    img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2]).cuda()
    img_id_downsample = F.interpolate(img_id, size=(112,112))
    latend_id = model.netArc(img_id_downsample)
    latend_id = F.normalize(latend_id, p=2, dim=1)

    video_swap(opt.video_path, latend_id, model, insightface_app_single, opt.output_path,temp_results_dir=opt.temp_path,no_simswaplogo=True,use_mask=opt.use_mask,crop_size=opt.crop_size)

    return StreamingResponse(open(f'temp/{infer_id}/result.mp4', mode='rb'), media_type='video/mp4')
