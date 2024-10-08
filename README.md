# MOL
MOL: Joint Estimation of Micro-Expression, Optical Flow, and Landmark via Transformer-Graph-Style Convolution

### Overview
<img src="figures/overview.png" style="zoom:45%;" />

### Datasets
Data preparation: The dataset should follow such folder structure.

```
│data/
├──CASME2_data_5/
│  ├── disgust
│  │   ├── 01_EP19_05f
│  │   │   ├── img1.jpg
│  │   │   ├── img2.jpg
│  │   │   ├── ......
│  ├── surprise
│  │   ├── ......

├──SMIC_data_3/
│  ├── surprise
│  │   ├── s9_sur_03
│  │   │   ├── image090823.jpg
│  │   │   ├── image090824.jpg
│  │   │   ├── ......
│  ├── negative
│  │   ├── ......

├──SAMM_data_5/
│  ├── anger
│  │   ├── 006_1_2
│  │   │   ├── 006_05562.jpg
│  │   │   ├── 006_05563.jpg
│  │   │   ├── ......
│  ├── contempt
│  │   ├── ......
|......
```
### Requirement
```
dlib==19.24.1
numpy==1.23.5
opencv_contrib_python_headless==4.7.0.72
opencv_python==4.7.0.72
opencv_python_headless==4.7.0.72
Pillow==9.5.0
scikit_learn==1.2.2
spatial_correlation_sampler==0.4.0
torch==1.13.0+cu116
torchsummary==1.5.1
torchvision==0.14.0+cu116
```

### Train
Run 'train.py' to train the model.
```

```

### Validate
Run 'eval.py' to validate.
```

```
