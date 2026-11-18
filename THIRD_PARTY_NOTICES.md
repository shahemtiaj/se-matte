# Third-party notices

SE Matte's own code (the After Effects panel, the engine package `engine/semat/` and the native helper
`engine/native/semfast/`) is in this repository. The AI models it uses are third-party work. Their code is
**not** stored here: `engine/setup_mac.sh` clones each project at the pinned commit below, and model weights are
downloaded from the authors' official release pages or Hugging Face.

| Component | Source (pinned) | License | Notes |
|---|---|---|---|
| SAM 2.1 | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) @ `2b90b9f` · weights [facebook/sam2.1-hiera-base-plus](https://huggingface.co/facebook/sam2.1-hiera-base-plus) | Apache-2.0 | Commercial use allowed with notice. |
| MatAnyone | [pq-yang/MatAnyone](https://github.com/pq-yang/MatAnyone) @ `e5ddc53` · weights [PeiqingYang/MatAnyone](https://huggingface.co/PeiqingYang/MatAnyone) | S-Lab License 1.0 | **Non-commercial use only.** Must be replaced or licensed before any commercial distribution. |
| Robust Video Matting | [PeterL1n/RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) @ `53d74c6` · weights from its v1.0.0 release | GPL-3.0 | Copyleft: distributing software that includes or links it (including converted Core ML models) brings GPL-3.0 obligations. |
| ViTMatte | [hustvl/vitmatte-small-composition-1k](https://huggingface.co/hustvl/vitmatte-small-composition-1k) (via 🤗 Transformers) | MIT | |
| BiRefNet | [ZhengPeng7/BiRefNet](https://github.com/ZhengPeng7/BiRefNet) @ `f707963` | MIT | Used for experiments. |

Python dependencies (PyTorch, FastAPI, coremltools, Transformers and others in `engine/requirements.txt`) are
installed from PyPI under their own licenses. FFmpeg is installed with Homebrew and used as a separate program.

**Current status:** SE Matte is built for internal team use, which all of the licenses above allow. The model
registry keeps every model swappable, so MatAnyone and Robust Video Matting can be replaced before any
commercial release.

Adobe, After Effects and the After Effects logo are trademarks of Adobe Inc. SE Matte is not affiliated with or
endorsed by Adobe.
