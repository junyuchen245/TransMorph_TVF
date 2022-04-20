# Unsupervised Learning of Diffeomorphic Image Registration via TransMorph

We propose a learning-based framework for unsupervised and end-to-end learning of diffeomorphic image registration. Specifically, the proposed network learns to produce and integrate ***time-dependent velocity fields*** in an LDDMM setting.

This repository contains the source code for two models, **TM-TVF<sub>LDDMM</sub>** and **TM-TVF**, from our paper: "Unsupervised Learning of Diffeomorphic Image Registration via TransMorph"

- Paper link: TBA
- This method is currently ranked **No. 1** @ [2021 MICCAI Learn2Reg challenge Task 03](https://learn2reg.grand-challenge.org/) (OASIS brain MR dataset)
- Please also check out our base network, [**TransMorph**](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration), a top-ranked Transformer-based network for image registration.
- Preprocessed datasets were made available @ [**TransMorph**](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration)

## Modeling time-dependent velocity fields using TransMorph:
<img src="https://github.com/junyuchen245/TransMorph_TVF/blob/main/example_imgs/net_arch.jpg" width="700"/>

## Smoother transformation without imposing a diffeomorphism:
<img src="https://github.com/junyuchen245/TransMorph_TVF/blob/main/example_imgs/field_visual.jpg" width="900"/>

## Diffeomorphic registration
***Forward:***
<img src="https://github.com/junyuchen245/TransMorph_TVF/blob/main/example_imgs/forward.gif" width="100"/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
***Inverse:***
<img src="https://github.com/junyuchen245/TransMorph_TVF/blob/main/example_imgs/inverse.gif" width="100"/>

***Inversion and composition:***\
<img src="https://github.com/junyuchen245/TransMorph_TVF/blob/main/example_imgs/forward_inverse.jpg" width="600"/>

## State-of-the-art performance:
Click on the `Model Weights` to start downloading the pre-trained weights.\
We also provided the Tensorboard training log for each model. To visualize loss and validation curves, run: \
```Tensorboard --logdir=*training log file name*``` in terminal. *Note: This requires Tensorboard installation (`pip install tensorboard`).*
### 2021 MICCAI Learn2Reg challenge Task 03:
#### Validation set results
|Ranking|Model|Dice|SDlogJ|HdDist95|Pretrained Weights|Tensorboard Log|
|---|---|---|---|---|---|---|
|[1](https://learn2reg.grand-challenge.org/evaluation/task-3-validation/leaderboard/)|TM-TVF|0.8691 ± 0.0145|0.0945|1.3969|N/A| N/A|
|[2](https://learn2reg.grand-challenge.org/evaluation/task-3-validation/leaderboard/)|[TM-Large](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/OASIS/TransMorph_on_OASIS.md)|0.8623 ± 0.0144|0.1276|1.4315|-|-|
|3|[TransMorph (TM)](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/OASIS/TransMorph_on_OASIS.md)|0.8575 ± 0.0145|0.1253|1.4594| - | - |

#### Test set results (*results obtained from Learn2Reg challenge organizers*)
|Ranking|Model|Dice|SDlogJ|HdDist95|
|---|---|---|---|---|
|1|TM-TVF|**0.8241 ± 0.1516**|0.0905 ± 0.0054|**1.6329 ± 0.4358**|
|2|[TM-Large](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/OASIS/TransMorph_on_OASIS.md)|*0.8196 ± 0.1497*|0.1244 ± 0.0148|1.6564 ± 1.7368|
|3|[TM](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/OASIS/TransMorph_on_OASIS.md)|0.8162 ± 0.1541| 0.1242 ± 0.0136|1.6920 ± 1.7587|
|4|[LapIRN](https://github.com/cwmok/LapIRN)|0.82| 0.07 |1.67|
|5|[ConvexAdam](https://github.com/multimodallearning/convexAdam)|0.81| 0.07 |1.63|
...

