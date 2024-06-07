# Text-to-Image Rectified Flow as Plug-and-Play Priors
<div align="left">
<a href='https://arxiv.org/abs/2406.03293'><img src='https://img.shields.io/badge/arXiv-2406.03293-b31b1b.svg'></a> 
</div>

<p style='text-align: justify;'> 
<em>by Xiaofeng Yang, Cheng Chen, Xulei Yang, Fayao Liu, Guosheng Lin.</em>
</p><p style='text-align: justify;'> 
Large-scale diffusion models have achieved remarkable performance in generative tasks. Beyond their initial training applications, these models have proven their ability to function as versatile plug-and-play priors. For instance, 2D diffusion models can serve as loss functions to optimize 3D implicit models. Rectified flow, a novel class of generative models, enforces a linear progression from the source to the target distribution and has demonstrated superior performance across various domains. Compared to diffusion-based methods, rectified flow approaches surpass in terms of generation quality and efficiency, requiring fewer inference steps. In this work, we present theoretical and experimental evidence demonstrating that rectified flow based methods offer similar functionalities to diffusion models — they can also serve as effective priors. Besides the generative capabilities of diffusion priors, motivated by the unique time-symmetry properties of rectified flow models, a variant of our method can additionally perform image inversion. Experimentally, our rectified flow-based priors outperform their diffusion counterparts — the SDS and VSD losses — in text-to-3D generation. Our method also displays competitive performance in image inversion and editing.</p>

## Updates
- 05/06/2024: Code Released.

## ToDo

- [x] Code release. Currently, the base text-to-image is based on **[InstaFlow](https://github.com/gnobitab/InstaFlow)**.
- [ ] Add support for Stable Diffusion 3 after the model is released.



## Installation


Our codes are based on the implementations of [ThreeStudio](https://github.com/threestudio-project/threestudio).
Please follow the instructions in ThreeStudio to install the dependencies.

## Quickstart
### 2D Playground
```
# run RFDS in 2D space for image generation
python 2dplayground_RFDS.py

# run RFDS-Rev in 2D space for image generation
python 2dplayground_RFDS_Rev.py

# run iRFDS in 2D space for image editing
python 2dplayground_iRFDS.py
```

### Text-to-3D with RFDS
```
python launch.py --config configs/rfds.yaml --train --gpu 0 system.prompt_processor.prompt="A DSLR photo of a hamburger" 
```

### Text-to-3D with RFDS-Rev
```
python launch.py --config configs/rfds-rev.yaml --train --gpu 0 system.prompt_processor.prompt="A DSLR photo of a hamburger" 
```

## Results

### Optimization in 2D space
Caption: an astronaut is riding a horse
<table style="border: none;">
  <tr>
    <td style="text-align: center;">
      <img src="images/RFDS_2d.png" alt="RFDS" width="300" height="300"/>
      <br><em>RFDS</em>
    </td>
    <td style="text-align: center;">
      <img src="images/RFDS_Rev_2d.png" alt="RFDS-Rev" width="300" height="300"/>
      <br><em>RFDS-Rev</em>
    </td>
  </tr>
</table>

### Text-to-3D with RFDS-Rev (NeRF backbone)

<table style="border: none;">
  <tr>
    <td style="text-align: center;">
      <img src="images/A_DSLR_image_of_a_hamburger.gif" alt="A DSLR image of a hamburger" width="300" height="300"/>
      <br><em>A DSLR image of a hamburger</em>
    </td>
    <td style="text-align: center;">
      <img src="images/An_intricate_ceramic_vase_with_peonies_painted_on_it.gif" alt="An intricate ceramic vase with peonies painted on it" width="300" height="300"/>
      <br><em>An intricate ceramic vase with peonies painted on it</em>
    </td>
  </tr>
</table>

### Text guided editing with iRFDS
![Editing Results](images/editing.PNG)



## Credits

RFDS is built on the following open-source projects:
- **[ThreeStudio](https://github.com/threestudio-project/threestudio)** Main Framework
- **[InstaFlow](https://github.com/gnobitab/InstaFlow)** Large-scale text-to-image Rectified Flow model


## Citation
```
@article{yang2024rfds,
  title={Text-to-Image Rectified Flow as Plug-and-Play Priors},
  author={Xiaofeng Yang and Cheng Chen and Xulei Yang and Fayao Liu and Guosheng Lin},
  journal={arXiv-2406.03293},
  year={2024}
}
```
 
