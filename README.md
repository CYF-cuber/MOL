# MOL
MOL: Joint Estimation of Micro-Expression, Optical Flow, and Landmark via Transformer-Graph-Style Convolution

### Overview
<img src="figures/overview.png" style="zoom:45%;" />

### Datasets
Data preparation: The dataset should follow the structure.

```
│data/
├──CASME2/
│  ├── disgust
│  │   ├── 01_EP19_05f
│  │   │   ├── img1.jpg
│  │   │   ├── img2.jpg
│  │   │   ├── ......
│  ├── surprise
│  │   ├── ......

├──SMIC/
│  ├── surprise
│  │   ├── s9_sur_03
│  │   │   ├── image090823.jpg
│  │   │   ├── image090824.jpg
│  │   │   ├── ......
│  ├── negative
│  │   ├── ......

├──SAMM/
│  ├── anger
│  │   ├── 006_1_2
│  │   │   ├── 006_05562.jpg
│  │   │   ├── 006_05563.jpg
│  │   │   ├── ......
│  ├── contempt
│  │   ├── ......
```
### Requirement
```

```

### Train
Run 'train.py' to train the model.
```

```

### Validate
Run 'eval.py' to validate.
```

```
