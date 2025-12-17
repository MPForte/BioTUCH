## Contact-Aware Refinement of Human Pose Pseudo-Ground Truth via Bioimpedance Sensing

[[Project Page](https://biotuch.is.tue.mpg.de/)] 

## Comment
The test code is ready to use!  
The full dataset will be released soon!

## Table of Contents
  * [Setup](#setup)
  * [Testing](#testing)
  * [Usage](#usage)
  * [License](#license)
  * [Description](#description)
  * [Citation](#citation)
  * [Acknowledgments](#acknowledgments)
  * [Contact](#contact)


## Setup

### Prerequisites
- Latest `conda` or `mamba` installation
- CUDA-capable GPU 

### Installation

1. **Create an account at [https://biotuch.is.tue.mpg.de/](https://biotuch.is.tue.mpg.de/)**


2. **Clone this repository:**
   ```bash
   git clone https://github.com/MPForte/BioTUCH.git
   cd BioTUCH
   ```

3. **Run the installation script:**
   ```bash
   ./install.sh
   ```

   The script will:
   - Prompt for your BioTUCH credentials
   - Download required models and demo data
   - Create the `biotuch` conda environment
   - Install all dependencies
   - Set up directory structure

## Testing

To test that the setup worked, run:

```bash
conda activate biotuch
python biotuch.py \
    --cfg_file configs/demo.yaml 
    --output output_test
```

We provide the expected output; check that `data/demo/output_test` and `data/demo/output` have the same results.

## Usage

### Running BioTUCH

Execute the fitting pipeline:

```bash
python biotuch.py --cfg_file cfg_files/demo.yaml
```

### Running on Your Own Data

To run BioTUCH on your own data, organize your input in the following structure:

```
data/demo/input/
├── frames/           # Extracted video frames (e.g., frame_0001.png, frame_0002.png, ...)
├── contact.json      # List of frame numbers with contact events: [5, 12, 23, ...]
└── initialization/   # initialization files (*.pkl)
```

**Run initialization method:**
Run initialization method (e.g., Multi-HMR) on your video first to generate initialization files.

### Run BioTUCH

Execute the fitting pipeline:

```bash
python biotuch.py --cfg_file cfg_files/demo.yaml
```

**Command line options:**
- `--cfg_file`: Path to configuration file (required)
- `--input_folder`: Input data folder (default: data/demo/input)
- `--output_folder`: Output folder (default: data/demo/output)
- `--skip_preprocessing`: Skip preprocessing steps (i.e., keypoints extraction) if already done
- `--interactive`: Enable verbose debug output

### Output

Results are saved to `data/demo/output/`:

```
output/
├── images/          # Rendered overlays
├── meshes/          # 3D mesh files (.obj)
├── results/         # Optimization results (.pkl)
└── output.mp4       # Final visualization video
```

**Note on rendering:**
This code produces the unsmoothed results used for evaluation in our paper. For the supplementary video, we apply OneEuroFilter smoothing, anchored at the first contact frame, to improve visual quality.

## License

Software Copyright License for **non-commercial scientific research purposes**.
Please read carefully the [terms and conditions](https://github.com/MPForte/biotuch/blob/master/LICENSE) and any accompanying documentation before you download and/or use the BioTUCH model, data and software, (the "Model & Software"), including 3D meshes, blend weights, blend shapes, textures, software, scripts, and animations. By downloading and/or using the Model & Software (including downloading, cloning, installing, and any other use of this github repository), you acknowledge that you have read these terms and conditions, understand them, and agree to be bound by them. If you do not agree with these terms and conditions, you must not download and/or use the Model & Software. Any infringement of the terms of this agreement will automatically terminate your rights under this [License](./LICENSE).

## Description

BioTUCH provides code for contact-aware refinement of human pose pseudo-ground truth using bioimpedance sensing. This repository contains the fitting pipeline that integrates tactile contact information with visual observations to improve 3D human pose estimation accuracy.

## Citation

If you find this Model & Software useful in your research, we would kindly ask you to cite:

```
@inproceedings{Forte25-ICCV-BioTUCH,
  title = {Contact-Aware Refinement of Human Pose Pseudo-Ground Truth via Bioimpedance Sensing},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  address = {Honolulu, USA},
  month = oct,
  year = {2025},
  note = {Nikos Athanasiou and Giulia Ballardini contributed equally to this publication},
  slug = {forte25-iccv-biotuch},
  author = {Forte, Maria-Paola and Athanasiou*, Nikos and Ballardini*, Giulia and Bartels, Jan Ulrich and Kuchenbecker, Katherine J. and Black, Michael},
  month_numeric = {10}
}
```

## Acknowledgments

We thank Tsvetelina Alexiadis for trial coordination; Markus Höschle for the capture setup; Taylor McConnell for data-cleaning coordination; Florentin Doll, Arina Kuznetcova, Tomasz Niewiadomski, and Tithi Rakshit for data cleaning; Giorgio Becherini for fitting the ground truth with MoSh++.

## Contact

For questions, please contact [forte@tue.mpg.de](mailto:forte@tue.mpg.de). 

For commercial licensing (and all related questions for business applications), please contact [ps-licensing@tue.mpg.de](mailto:ps-licensing@tue.mpg.de).
