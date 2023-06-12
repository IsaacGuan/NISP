# NISP: Neural Implicit Surface Parameterization

This repository contains the experimental code and data for training a neural network that learns to implicitly represent the parameterization charts of the object surface, as described in the paper ["Learning Neural Implicit Representations with Surface Signal Parameterizations"](https://arxiv.org/abs/2211.00519).

## Data

The experimental data is in the `data` folder, where `data/shapes` stores the OBJ files for the experimental 3D objects containing the triangle meshes and UV maps, `data/diffuse-maps` stores the diffuse maps, `data/normal-maps` stores the normal maps, and `sdf-models` stores the pre-trained neural implicit surfaces of the 3D objects learned by [OverfitSDF](https://github.com/daviesthomas/overfitSDF).

## Training

Use `main.py` for training our model, with the argument `--train` turned on. For example,
```bash
python main.py --train --model_name apple_strudel --fourier_max_freq 10 --use_siren
```
where `--model_name` refers to the name of the object, such as `apple`, `banana`, and so on (see the files in `data/shapes`); `--fourier_max_freq` controls the number of Fourier series the input is encoded into; and `--use_siren` controls whether the SIREN layer is implemented for the hidden layers. For running other comparative baseline models, please use the argument `--texture_model_type`, and run with value `color` or `uv`.

The trained model and the output files generated during training will be saved in `decomposed-uv-mapper` (or in `color-mapper` or `uv-mapper`) in the `results` folder, depending on the type of model being trained.

## Rendering

Use `main.py` to render the implicit surface with texture mapping enabled by our model, with `--train` off. For example,
```bash
python main.py --model_name apple_strudel --use_normal_map
```
where `--use_normal_map` is optional and controls whether to enable normal mapping or not. Texture mapping results by the comparative baseline models can also be rendered through the argument `--texture_model_type`. The rendered images will appear in `decomposed-uv-mapper` (or in `color-mapper` or `uv-mapper`) in the `results` folder.
