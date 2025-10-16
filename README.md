# Low Light Vision Encoder
Building a **Low-Light Image Feature Extractor**.<br>
The feature extracting will be done directly on Low-Light images **without enhancing them**.

## Theory
If a feature extractor is trained on **multiple tasks simultaneously**, then the features extractor should be able to provide good features generalized to all low light related tasks.

## Training
A single feature extractor (ResNet-50) is taken and a U-Net is attached onto it which performs two tasks - **Low-Light Image Enhancement** & **Semantic Segmentation**.<br>
**Specifics:-** <br>
- ResNet-50 backbone is **unfrozen**. The backbone is wrapped into an Encoder which outputs feature maps from each layer to be used in skip connections.
- A combined U-Net model is used with custom decoder blocks. Out of the 6 decoder blocks (including final head), 3 are **shared** between the tasks, and 3 are **task-specific**.

## Data
**Mini MS-COCO** dataset (25k subset) is used and Segmentation masks are generated.<br>A **synthetically darkening pipeline using statistical methods** is used to darken the images. Refer to my other repository for the pipeline.<br>The output targets are the **masks** (Semantic Segmentation) and the **normal light images** (Enhancement) for the **darkened images** (Input).

## Results
### EX-DARK DATASET
| Metrics | Pretrained ResNet-50 | Fine-Tuned ResNet-50 |
|:----:|-----:|-----:|
|Top-1 Accuracy|65.04%|72.69%<span style="color:green"> (+7.65%)</span>|
|Top-5 Accuracy|94.08%|95.79%<span style="color:green"> (+1.71%)</span>|

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
