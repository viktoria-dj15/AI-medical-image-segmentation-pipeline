"""Prepare renamed copies and fixed fold 0; no test data, planning or training."""
from pathlib import Path
import csv, hashlib, json, shutil
import nibabel as nib
import numpy as np

PROJECT = Path.home() / '3D segmentation + slicer'
ROOT = Path.home() / 'Documents/medical/nnunet_workspace'
NAME = 'Dataset501_COVID19Lesions'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()

def main():
    manifest_path = PROJECT / 'splits.json'
    m = json.loads(manifest_path.read_text())
    assert [len(m[k]) for k in ('train','validation','test')] == [139,30,30]
    ids = {k: [c['case_id'] for c in m[k]] for k in ('train','validation','test')}
    groups = {k: {c['group_id'] for c in m[k]} for k in ids}
    assert len(set(sum(ids.values(), []))) == 199
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        assert not groups[a] & groups[b]
    raw = ROOT/'nnUNet_raw'/NAME
    prep = ROOT/'nnUNet_preprocessed'/NAME
    for folder in [raw/'imagesTr',raw/'labelsTr',prep,ROOT/'nnUNet_results',ROOT/'manifests']:
        folder.mkdir(parents=True,exist_ok=True)
    expected_images = {f'{cid}_0000.nii.gz' for cid in ids['train']+ids['validation']}
    expected_labels = {f'{cid}.nii.gz' for cid in ids['train']+ids['validation']}
    if {p.name for p in (raw/'imagesTr').iterdir()} - expected_images:
        raise RuntimeError('Unexpected files in imagesTr; refusing to mix datasets.')
    if {p.name for p in (raw/'labelsTr').iterdir()} - expected_labels:
        raise RuntimeError('Unexpected files in labelsTr.')
    mapping=[]
    for subset in ('train','validation'):
        for case in m[subset]:
            cid=case['case_id']
            image=Path(case['image']); label=Path(case['label'])
            ct=nib.load(image); mask=nib.load(label)
            assert len(ct.shape)==3 and ct.shape==mask.shape, cid
            assert np.allclose(ct.affine,mask.affine,atol=1e-3,rtol=0), cid
            assert np.isfinite(ct.affine).all(),cid
            values=np.unique(np.asanyarray(mask.dataobj))
            assert set(values.tolist()) <= {0,1},(cid,values)
            row={'case_id':cid,'group_id':case['group_id'],'split':subset}
            for kind,src,dst in [
                ('image',image,raw/'imagesTr'/f'{cid}_0000.nii.gz'),
                ('label',label,raw/'labelsTr'/f'{cid}.nii.gz')]:
                checksum=digest(src)
                if not dst.exists():
                    tmp=dst.with_name(dst.name+'.partial')
                    shutil.copy2(src,tmp)
                    tmp.replace(dst)
                if digest(dst)!=checksum:
                    raise RuntimeError(f'Copy differs from source: {dst}')
                row[kind+'_source']=str(src)
                row[kind+'_destination']=str(dst)
                row[kind+'_sha256']=checksum
            mapping.append(row)
            if len(mapping)%20==0:print('Verified copies:',len(mapping),flush=True)
    assert {p.name for p in (raw/'imagesTr').iterdir()}==expected_images
    assert {p.name for p in (raw/'labelsTr').iterdir()}==expected_labels
    metadata={
        'name':'COVID19Lesions',
        'description':'139 training + 30 validation cases; custom fold 0; 30 held-out tests excluded.',
        'channel_names':{'0':'CT'},'labels':{'background':0,'lesion':1},
        'numTraining':169,'file_ending':'.nii.gz',
        'overwrite_image_reader_writer':'NibabelIO',
    }
    (raw/'dataset.json').write_text(json.dumps(metadata,indent=2)+'\n')
    split=[{'train':ids['train'],'val':ids['validation']}]
    (prep/'splits_final.json').write_text(json.dumps(split,indent=2)+'\n')
    (ROOT/'manifests'/'splits_final.json').write_text(json.dumps(split,indent=2)+'\n')
    (ROOT/'manifests'/'held_out_test.json').write_text(json.dumps({
        'case_ids':ids['test'],'group_ids':[c['group_id'] for c in m['test']],
        'policy':'Never copied into nnUNet_raw or nnUNet_preprocessed; do not use for planning/training/tuning.'
    },indent=2)+'\n')
    with (ROOT/'manifests'/'conversion_manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=mapping[0].keys());w.writeheader();w.writerows(mapping)
    report={'dataset':NAME,'training':139,'validation':30,'held_out_test':30,
        'copied_images':169,'copied_labels':169,'source_splits_sha256':digest(manifest_path),
        'copy_hash_checks':'PASS','geometry_and_binary_label_checks':'PASS',
        'test_cases_copied':0,'planning_started':False,'training_started':False}
    (ROOT/'manifests'/'preparation_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
