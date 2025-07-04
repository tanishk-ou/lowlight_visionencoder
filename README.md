# Low Light Vision Encoder
A 3-phase training plan for a generalized low light vision encoder.

## Theory
If a feature extractor is trained on multiple tasks simultaneously, then the features extractor should be able to provide good features generalized to all low light related tasks.

## Training Plan
### 1st Phase
Training a single encoder on 3 different computer vision, low light related tasks on a synthetically darkened low light image dataset.
### 2nd Phase
Training the encoder from previous phase on 2 different tasks on a synthetically simulated low light image dataset.
### 3rd Phase
Training the encoder from previous phase using Self-Supervised Learning on a real low light image dataset.
### Evaluation
Evaluating the encoder on a classification task on a low light image dataset, provided the encoder has never been trained on any classification-related task nor has it seen the dataset used for evaluation.

## Results
### Phase 1
Linear Evaluation :-
| Metrics | Pretrained Resnet50 | Phase-1 trained Resnet50 |
|:----:|-----:|-----:|
|5-Fold Top-1 Accuracy|65.04%|65.64%|
|5-Fold Top-5 Accuracy|94.08%|94.65%|

## Files
### Phase 1
Download `phase1_final_encoder.pt` through the following link-
[Google Drive](https://drive.google.com/file/d/1zNgsu2sn964O54Keq_Op0Xsi3joelJUc/view?usp=drivesdk)
<br>
Saved weights for Resnet50 can be loaded from `phase1_final_encoder.pt` file through the following code and replacing `root` with the directory where the `.pt` file is saved, and `device` with either `'cuda'` or `'cpu'`.
```python
from torchvision import models
resnet = models.resnet50(weights='DEFAULT')
resnet.fc = nn.Identity()
resnet.load_state_dict(torch.load('root/phase1_final_encoder.pt', map_location = device, weights_only = False))
```
