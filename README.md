### 1. Downloading + Preprocessing + Organizing TCGA Data

Using the [NIH Genomic Data Commons Data Portal](https://portal.gdc.cancer.gov/) and the [cBioPortal](https://www.cbioportal.org/), we downloaded diagnostic whole-slide images (WSIs) for 28 cancer types using the [GDC Data Transfer Tool](https://docs.gdc.cancer.gov/Data_Transfer_Tool/Users_Guide/Data_Download_and_Upload/), followed by using the publicly-available [CLAM library](https://github.com/mahmoodlab/CLAM) for tissue segmentation, tissue patching and feature extraction, which we modified for extracting both ResNet-50 features (pretrained on ImageNet) and ViT-16 features (pretrained on the TCGA). For patching at `[256 × 256]` resolution, we used default tissue segmentation parameters. For patching at `[4096 × 4096]` resolution, we additionally saved each `[4096 × 4096]` image region, which we used for ViT-16/256 pretraining. Extracted TCGA features are organized in the following example directory:
<details>
<summary>
Example Directory
</summary>
  
```bash
TCGA_ROOT_DIR/
    └──tcga_acc/
        ├── ...
    └──tcga_blca/
        ├── ...
    └──TCGA_brca/
        └── WSIs/
            ├── slide_1.svs
            ├── slide_2.svs
            └── ...
        └── extracted_mag20x_patch256_fp/
            └── masks/
                ├── slide_1.png
                ├── slide_2.png
                └── ...
            └── patches/
                ├── slide_1.h5
                ├── slide_2.h5
                └── ...
            └── stitches/
                ├── slide_1.png
                ├── slide_2.png
                └── ...
            └── resnet50_trunc_pt_patch_features/
                ├── slide_1.pt
                ├── slide_2.pt
                └── ...
            └── vits_tcga_pancancer_dino_pt_patch_features/
                ├── slide_1.pt
                ├── slide_2.pt
                └── ...
            └── process_list_autogen.csv
        └── extracted_mag20x_patch4096_fp/
            └── masks/
                ├── slide_1.png
                ├── slide_2.png
                └── ...
            └── patches/
                ├── slide_1.h5
                ├── slide_2.h5
                └── ...
            └── stitches/
                ├── slide_1.png
                ├── slide_2.png
                └── ...
            └── tar_patch_4096/
                ├── slide_1.tar
                ├── slide_2.tar
                └── ...
            └── vits_tcga_pancancer_dino_pt_patch_features/
                ├── slide_1.pt
                ├── slide_2.pt
                └── ...
            └── process_list_autogen.csv
    └──tcga_coadread/
        ├── ...
    ...
    └──tcga_ucec/
        ├── ...
```
  
</details>

Each cancer type is organized as its own folder in `TCGA_ROOT_DIR`, which additionally contains the following subfolders:
In extracting patches at level 0 with non-overlapping patch sizes of 256, we create a results directory called `extracted_level0_patch256` that will contain the following files / folders:
1. `WSIs/`: Raw `*.svs` WSIs for that cancer type
2. `extracted_mag20x_patch256_fp`: Extracted features at 20× magnification for `[256 × 256]` patches (performed only for BRCA, COADREAD, LUAD, LUSC, CCRCC, CHRCC, PRCC, and STAD studies in TCGA). The `_fp` suffix represents the use of 'fast patching" as performed in CLAM, in which coordinates instead of raw patches are saved. This folder contains the following subfolders:
    - `masks/`: Directory of segmented tissue-containing regions (one image per WSI).
    - `patches/`: Directory of extracted image patches (one .h5 file per WSI, where each entry corresponds to the coordinates of the top-left corner of a patch)
    - `stitches/`: Directory of downsampled visualizations of stitched tissue patches, used a sanity check to inspect whether we patched correctly (one image per WSI). 
    - `resnet50_trunc_pt_patch_features/`: Directory of pre-extracted ResNet-50 features (pretrained on ImageNet) for each patch within each WSI (with patches read via OpenSlide using coordinates in `patches/`, saved in a `*.pt` format. Each `*.pt` file is a `[M × 1024]`-sized Tensor containing extracted 1024-dim embeddings for `M` patches in the WSI.
    - `vits_tcga_pancancer_dino_pt_patch_features/`: Directory of pre-extracted ViT-16 features (pretrained on TCGA) for each patch within each WSI (with patches read via OpenSlide using coordinates in `patches/`, saved in a `*.pt` format. Each `*.pt` file is a `[M × 384]`-sized Tensor containing extracted 384-dim embeddings for `M` patches in the WSI.
    - `process_list_autogen.csv`: An auto-generated csv file that contains a list of all slides processed, along with their segmentation/patching parameters used.
3. `extracted_mag20x_patch4096_fp`: Extracted features at 20× magnification for `[4096 × 4096]` image regions, containing the following subfolders:
    - `masks/`: Same as `[256 × 256]` setting.
    - `patches/`: Same as `[256 × 256]` setting.
    - `stitches/`: Same as `[256 × 256]` setting.
    - `tar_patch_4096/`: Directory of saved `[4096 × 4096]` image regions for each WSI, stored in a `*.tar` format using [WebDataset](https://github.com/webdataset/webdataset) API.
    - `vits_tcga_pancancer_dino_pt_patch_features/`: Directory of pre-extracted ViT-16 features (pretrained on TCGA) for each `[4096 × 4096]` region within each WSI (with regions read via OpenSlide using coordinates in `patches/`, saved in a `*.pt` format. Each `*.pt` file is a `[M × 256 × 384]`-sized Tensor containing extracted 384-dim embeddings for `M` regions in the WSI, which each region represented as as a 256-length sequence of `[256 × 256]` patch embeddings.
    - `process_list_autogen.csv`: An auto-generated csv file that contains a list of all slides processed, along with their segmentation/patching parameters used. Note that in using a large image resolution for patching, not all WSIs are used in `[4096 × 4096]` evaluation.

### 2. ViT-16/256 Pretraining
<details>
<summary>
Example Directory
</summary>
  
```bash
TCGA_PRETRAINING_DIR/
  └──patch_256_pretraining/
      ├── patch_1.png
      ├── patch_2.png
      └── ...
  └──region_4096_pretraining/
      ├── slide_1_1.pt
      ├── slide_1_2.pt
      └── ...
  └──ckpts/
      └── pretrain/
          └── vit256_s_dino.pth
          └── vit4k_xs_dino.pth

  └──patch_256_pretraining/
      ├── patch_1.png
      ├── patch_2.png
      └── ...
  └──region_4096_pretraining/
      ├── slide_1_1.pt
      ├── slide_1_2.pt
      └── ...
 ```
 </details>
 
 We set up the following directories for ViT-16 and ViT-256 pretraining respectively:
  - `patch_256_pretraining/`: Directory of raw `[256 × 256]` patches (as `*.png` format) extracted from the `tar_patch_4096/` subdirectories of each cancer type, used to pretrain ViT-16.
  - `region_4096_pretraining/`: Directory of pre-extracted ViT-16 features for each `[4096 × 4096]` region across all WSIs (in total: 433779 regions). Each `*.pt` file is a `[256 × 384]`-sized Tensor, which is a 256-length sequence of pre-extracted ViT-16 features for each `[256 × 256]` patch. This folder is used to pretain ViT-256.
  - `ckpts/pretrain/`: Directory for holding the pretrained weights, which we use for feature extraction. The following commands below are used for pretraining.

```python
python -m torch.distributed.launch --nproc_per_node=8 main_dino256.py --arch vit_small --data_path /path/to/patch_256_pretraining/ --output_dir /path/to/ckpts/pretrain/ --epochs 100
python -m torch.distributed.launch --nproc_per_node=8 main_dino4k.py --arch vit_small --data_path /path/to/region_4k_pretraining/ --output_dir /path/to/ckpts/pretrain/ --epochs 100
```
