"""Final test evaluator for original TransUNet and Soft ROI Refiner."""
from __future__ import annotations
import argparse, json, os
import torch
from torch.utils.data import DataLoader
from datasets.dataset_fetalheart import FetalHeartNPZDataset
from trainer_fetalheart import evaluate_present_case_class

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root_path',required=True); p.add_argument('--list_dir',required=True)
    p.add_argument('--checkpoint',required=True); p.add_argument('--output',required=True)
    p.add_argument('--model_type',choices=['original','roi_refiner',
        'baseline_unet','baseline_attunet','baseline_unetpp','baseline_swinunet'],required=True)
    p.add_argument('--arch',default=None,help='baseline arch override; defaults to model_type suffix')
    p.add_argument('--split',choices=['test'],default='test'); p.add_argument('--img_size',type=int,default=384)
    p.add_argument('--num_classes',type=int,default=12); p.add_argument('--vit_name',default='R50-ViT-B_16')
    p.add_argument('--vit_patches_size',type=int,default=16); p.add_argument('--n_skip',type=int,default=3)
    args=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required')
    from networks.vit_seg_modeling import CONFIGS
    config=CONFIGS[args.vit_name]; config.n_classes=args.num_classes; config.n_skip=args.n_skip
    config.patches.grid=(args.img_size//args.vit_patches_size,args.img_size//args.vit_patches_size)
    if args.model_type == 'original':
        from networks.vit_seg_modeling import VisionTransformer
        model=VisionTransformer(config,img_size=args.img_size,num_classes=args.num_classes).cuda()
    elif args.model_type == 'roi_refiner':
        from networks.vit_seg_modeling_roi_refiner import SoftROIRefinerVisionTransformer
        model=SoftROIRefinerVisionTransformer(config,img_size=args.img_size,num_classes=args.num_classes).cuda()
    else:
        from networks.baselines_fetalheart import build_baseline
        arch=args.arch or args.model_type.replace('baseline_','')
        model=build_baseline(arch,num_classes=args.num_classes,img_size=args.img_size,pretrained=False).cuda()
    state=torch.load(args.checkpoint,map_location='cuda'); model.load_state_dict(state); model.eval()
    dataset=FetalHeartNPZDataset(args.root_path,args.list_dir,args.split)
    loader=DataLoader(dataset,batch_size=1,shuffle=False,num_workers=4)
    metrics=evaluate_present_case_class(model,loader,args.num_classes,args.img_size,'cuda')
    metrics.update({'split':args.split,'checkpoint':args.checkpoint,'model_type':args.model_type,'cases':len(dataset)})
    os.makedirs(os.path.dirname(args.output) or '.',exist_ok=True)
    with open(args.output,'w',encoding='utf-8') as h: json.dump(metrics,h,indent=2,ensure_ascii=False)
    print(json.dumps(metrics,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
