# Low Light Vision Encoder
Training to build a generalized low-light image feature extractor.

## Theory
If a feature extractor is trained on multiple tasks simultaneously, then the features extractor should be able to provide good features generalized to all low light related tasks.

## Training
A single feature extractor (ResNet-50) is taken and a U-Net is attached onto it which performs two tasks - Low-Light Image Enhancement & Semantic Segmentation.<br>


## Results
### Phase 1
Linear Evaluation :-
| Metrics | Pretrained Resnet50 | Phase-1 trained Resnet50 |
|:----:|-----:|-----:|
|5-Fold Top-1 Accuracy|65.04%|65.64%|
|5-Fold Top-5 Accuracy|94.08%|94.65%|

## Files
### Phase 1
Download `phase1_final_encoder.pt` through the following link-<br>
[Google Drive](https://drive.google.com/file/d/1zNgsu2sn964O54Keq_Op0Xsi3joelJUc/view?usp=drivesdk)
<br><br>
Saved weights for Resnet50 can be loaded from `phase1_final_encoder.pt` file through the following code and replacing `root` with the directory where the `.pt` file is saved, and `device` with either `'cuda'` or `'cpu'`.
```python
from torchvision import models
resnet = models.resnet50(weights='DEFAULT')
resnet.fc = nn.Identity()
resnet.load_state_dict(torch.load('root/phase1_final_encoder.pt', map_location = device, weights_only = False))
```
