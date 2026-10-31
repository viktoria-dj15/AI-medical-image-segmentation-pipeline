"""Planning-only memory proxy. No forward pass, GPU allocation, data loading or training."""
import json
from pathlib import Path
import torch
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

def main():
    path = Path.home()/'Documents/medical/nnunet_workspace/nnUNet_preprocessed/Dataset501_COVID19Lesions/nnUNetPlans.json'
    plans=json.loads(path.read_text())
    config=plans['configurations']['3d_fullres']
    a=config['architecture']
    # Meta tensors represent shapes only; model weights are not allocated.
    with torch.device('meta'):
        model=get_network_from_plans(a['network_class_name'],a['arch_kwargs'],a['_kw_requires_import'],1,2,allow_init=False,deep_supervision=True)
    parameters=sum(p.numel() for p in model.parameters())
    batch=int(config['batch_size'])
    features=int(model.compute_conv_feature_map_size(config['patch_size']))
    gib=1024**3
    print('PLAN-ONLY MEMORY CHECK — no training performed')
    print('Configuration: 3d_fullres')
    print('Patch:',config['patch_size'],'batch:',batch)
    print('Parameters:',f'{parameters:,}')
    print('FP32 parameters only:',round(parameters*4/gib,3),'GiB')
    print('FP32 weights + gradients + SGD momentum:',round(parameters*12/gib,3),'GiB')
    print('Summed convolution feature-map storage proxy:')
    for n in [1,batch]:
        print(f'  Batch {n}: {features*n*4/gib:.2f} GiB (FP32)')
    print('This feature-map proxy is NOT measured peak memory, a bound, or a fit guarantee.')
    print('Actual training also needs saved activations, backward buffers, loss, optimizer workspaces, data and macOS memory.')
    print('Next: a bounded MPS forward/backward benchmark, initially batch 1, before full-batch training.')
if __name__=='__main__':main()
