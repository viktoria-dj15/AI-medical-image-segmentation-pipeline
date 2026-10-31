"""Bounded synthetic MPS feasibility test, not a segmentation-quality experiment."""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
os.environ['nnUNet_compile'] = 'false'
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size',type=int,default=1,choices=[1,2])
    parser.add_argument('--steps',type=int,default=3,choices=range(2,6))
    args=parser.parse_args()
    project=Path(__file__).resolve().parents[1]
    output=project/'outputs'/f'nnunet_mps_benchmark_{datetime.now():%Y%m%d_%H%M%S_%f}'
    output.mkdir(parents=True,exist_ok=False)
    plan_path=Path.home()/'Documents/medical/nnunet_workspace/nnUNet_preprocessed/Dataset501_COVID19Lesions/nnUNetPlans.json'
    plans=json.loads(plan_path.read_text())
    config=plans['configurations']['3d_fullres']
    report={'status':'started','batch_size':args.batch_size,'patch_size':config['patch_size'],
            'planned_batch_size':config['batch_size'],'precision':'float32','compile':False,
            'cpu_fallback':False,'steps':[],
            'memory_note':'Stage-sampled current allocations, NOT true peak memory. Synthetic data; no loading/augmentation overhead.'}
    gib=1024**3
    def memory(stage):
        torch.mps.synchronize()
        return {'stage':stage,'tensor_gib':torch.mps.current_allocated_memory()/gib,
                'driver_gib':torch.mps.driver_allocated_memory()/gib}
    def save():
        (output/'summary.json').write_text(json.dumps(report,indent=2))
    save()
    try:
        if not torch.backends.mps.is_available():
            raise RuntimeError('MPS unavailable; use the nnU-Net interpreter on your Mac.')
        recommended=torch.mps.recommended_max_memory()
        # Keep allocator limits enabled; do not consume the entire unified-memory budget.
        torch.mps.set_per_process_memory_fraction(0.65)
        report['mps_recommended_max_gib']=recommended/gib
        report['allocator_limit_gib']=0.65*recommended/gib
        print('MPS synthetic benchmark: full-resolution plan',flush=True)
        print('Patch:',config['patch_size'],'batch:',args.batch_size,flush=True)
        print('Allocator cap:',round(report['allocator_limit_gib'],2),'GiB',flush=True)
        print('Output:',output,flush=True)
        torch.manual_seed(42)
        a=config['architecture']
        model=get_network_from_plans(a['network_class_name'],a['arch_kwargs'],a['_kw_requires_import'],1,2,deep_supervision=True).to('mps')
        model.train()
        report['parameters']=sum(p.numel() for p in model.parameters())
        optimizer=torch.optim.SGD(model.parameters(),lr=1e-2,momentum=0.99,nesterov=True,weight_decay=3e-5)
        image=torch.randn((args.batch_size,1,*config['patch_size']),device='mps')
        target=torch.zeros((args.batch_size,1,*config['patch_size']),dtype=torch.float32)
        z,y,x=config['patch_size']
        target[:,:,z//3:2*z//3,y//3:2*y//3,x//3:2*x//3]=1
        strides=np.array(a['arch_kwargs']['strides'])
        scales=(1/np.cumprod(strides,axis=0))[:-1]
        targets=[torch.nn.functional.interpolate(target,size=tuple(int(round(d*s)) for d,s in zip(config['patch_size'],scale)),mode='nearest').long().to('mps') for scale in scales]
        del target
        weights=np.array([1/(2**i) for i in range(len(targets))]);weights[-1]=0;weights/=weights.sum()
        loss_function=DeepSupervisionWrapper(DC_and_CE_loss({'batch_dice':config['batch_dice'],'smooth':1e-5,'do_bg':False,'ddp':False},{},dice_class=MemoryEfficientSoftDiceLoss),weights)
        report['initial_memory']=memory('initialized')
        overall=time.perf_counter()
        for step in range(1,args.steps+1):
            optimizer.zero_grad(set_to_none=True)
            torch.mps.synchronize();start=time.perf_counter()
            print(f'Step {step}/{args.steps}: forward...',flush=True)
            outputs=model(image)
            samples=[memory('after_forward')]
            loss=loss_function(outputs,targets)
            samples.append(memory('after_loss'))
            if not torch.isfinite(loss).item():raise RuntimeError('Non-finite loss')
            print('  backward...',flush=True)
            loss.backward()
            samples.append(memory('after_backward'))
            torch.nn.utils.clip_grad_norm_(model.parameters(),12)
            optimizer.step()
            samples.append(memory('after_optimizer'))
            elapsed=time.perf_counter()-start
            result={'step':step,'warmup':step==1,'loss':float(loss.item()),'seconds':elapsed,'memory_samples':samples}
            report['steps'].append(result);save()
            print(f'  {elapsed:.2f}s; loss={result["loss"]:.4f}; driver={samples[-1]["driver_gib"]:.2f} GiB',flush=True)
            del outputs,loss
            if time.perf_counter()-overall>180 and step<args.steps:
                report['status']='stopped_after_time_budget';break
        else:report['status']='passed'
        timings=[s['seconds'] for s in report['steps'] if not s['warmup']]
        report['mean_measured_step_seconds']=float(np.mean(timings)) if timings else None
        report['maximum_sampled_driver_gib']=max(m['driver_gib'] for s in report['steps'] for m in s['memory_samples'])
        report['maximum_sampled_tensor_gib']=max(m['tensor_gib'] for s in report['steps'] for m in s['memory_samples'])
    except (RuntimeError,NotImplementedError) as error:
        report['status']='failed_or_memory_limited'
        report['error']=str(error)
        print('Benchmark stopped:',error,flush=True)
    finally:
        save()
    print('\nBENCHMARK SUMMARY',flush=True)
    print(json.dumps({k:v for k,v in report.items() if k!='steps'},indent=2))
    print('Saved:',output/'summary.json')
    print('Batch 1 success does not establish batch 2 fits. No dataset training was performed.')
    return 0 if report['status']=='passed' else 1
if __name__=='__main__':raise SystemExit(main())
