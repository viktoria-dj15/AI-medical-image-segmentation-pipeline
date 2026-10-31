"""Run inside 3D Slicer's Python Console, not the PyCharm interpreter."""
from pathlib import Path
import os
import slicer

folder = Path(os.environ.get(
    "COVID19_TRAIN_DIR",
    str(Path.home() / "Documents/medical/covid19-20/COVID-19-20_v2/Train"),
)).expanduser()
case = 'volume-covid19-A-0256_1'
ct_path = folder / f'{case}_ct.nii.gz'
mask_path = folder / f'{case}_seg.nii.gz'
for path in (ct_path, mask_path):
    if not path.is_file():
        raise FileNotFoundError(path)

# Hide existing segmentations so other cases do not overlap this CT.
for node in slicer.util.getNodesByClass('vtkMRMLSegmentationNode'):
    if node.GetDisplayNode():
        node.GetDisplayNode().SetVisibility(False)

ct = slicer.util.loadVolume(str(ct_path))
mask = slicer.util.loadLabelVolume(str(mask_path))
segmentation = slicer.mrmlScene.AddNewNodeByClass(
    'vtkMRMLSegmentationNode', 'Case 0256_1 — ground truth'
)
segmentation.CreateDefaultDisplayNodes()
segmentation.SetReferenceImageGeometryParameterFromVolumeNode(ct)
slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(mask, segmentation)
segment_id = segmentation.GetSegmentation().GetNthSegmentID(0)
segment = segmentation.GetSegmentation().GetSegment(segment_id)
segment.SetName('COVID-19 lesions')
segment.SetColor(1.0, 0.25, 0.1)
display = segmentation.GetDisplayNode()
display.SetOpacity2DFill(0.25)
display.SetOpacity2DOutline(1.0)
ct.GetDisplayNode().AutoWindowLevelOff()
ct.GetDisplayNode().SetWindowLevel(1400, -300)
slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
slicer.util.setSliceViewerLayers(background=ct, foreground=None, label=None)
slicer.util.resetSliceViews()
segmentation.CreateClosedSurfaceRepresentation()
slicer.util.resetThreeDViews()
print('Loaded case 0256_1. Inspect alignment in all three slice views.')
