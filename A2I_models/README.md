# A2I_models

Baseline audio-to-image models used for comparison. Each folder is a copy of the
upstream project at the commit listed below, with local modifications for our
experiments. All credit for the original code goes to the upstream authors; see
each folder's own README and LICENSE.

| Folder | Upstream | Based on commit |
|---|---|---|
| `MACS/` | https://github.com/alxzzhou/MACS | `becd587` |
| `SonicDiffusion/` | https://github.com/BurakCanBiner/SonicDiffusion | `36be618` |
| `Sound2Scene/` | https://github.com/kaist-ami/Sound2Scene | `19ecc2d` |
| `Official/Attend-and-Excite/` | https://github.com/AnuragSingh0000/Attend-and-Excite | `163efdf` |
| `Official/AudioToken/` | https://github.com/guyyariv/AudioToken | `0cdd591` |
| `Official/stable-diffusion/` | https://github.com/CompVis/stable-diffusion | `21f890f` |

## Not included in this repository

Pretrained weights and generated outputs are too large for git. Download the
weights by following each upstream README and place them here:

| Path | Contents |
|---|---|
| `MACS/weights/` | MACS UNet, MLP and separator checkpoints |
| `MACS/laion_clap/630k-audioset-best.pt` | LAION-CLAP checkpoint |
| `SonicDiffusion/ckpts/` | SonicDiffusion and CLAP checkpoints |
| `Sound2Scene/checkpoints/` | Sound2Scene, IC-GAN and SwAV checkpoints |
| `Official/output/` | Generated images |

## Submodules

CLIP and taming-transformers are linked as submodules. Clone with:

```
git clone --recurse-submodules https://github.com/AnuragSingh0000/Compositional_A2I_Generation.git
```
