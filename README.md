# README

- ### Folder Description

  - SAM-guided DEADiff (experiment directory for SAM-guided DEADiff)
    - DEADiff: the DEADiff project, including the pipeline files for the entire SAM-guided DEADiff project and the interface scripts
    - groundingDINO+sam: the GroundingDINO-guided SAM semantic segmentation project
  - Overall project architecture:

![1767361036880](C:\Users\Yaqu\AppData\Roaming\Typora\typora-user-images\1767361036880.png)

### Key Files for SAM-guided DEADiff

- scripts: (core scripts)

  - app.py (Gradio version)

  - app_batch.py (batch processing)

  - app_batch_miniSet.py (for the miniset dataset)

  - app_batch_miniSet_mutimode (multiple modes with foreground/background control)

    These scripts run **plain DEADiff**. Just set the initial parameters in the scripts to run.

  - app_canny_control.py

  - app_canny_control_batch.py

  - app_depth_control.py

  - app_depth_control_batch.py

    These are **ControlNet-guided inference scripts**, including **canny** and **depth** modes, and each has a **single-test** version and a **batch-test** version.

  - app_canny_control_sam.py

  - app_deapth_control_sam.py

    These two scripts are for **SAM-guided local stylization experiments**. You need to modify parameters according to the comments in the scripts. Since **DEADiff** and **GroundingDINO + SAM** are two separate subprojects and may each require its own conda environment, you must correctly specify the conda environment for the **GroundingDINO + SAM** project for the scripts to run properly.

For detailed environment setup steps and usage instructions, see the `README.md` under the `DEADiff` folder.



### groundingDINO+sam中关键文件说明

- **code**: Source code for Project 1
  - **Pytorch-UNET**: UNet project source code, corresponding to basic task
    - **output-epoch5-lr0.00001**: Stores first training results
    - **output-epoch50-lr0.001**: Stores second training results
    - **car_test:** Test dataset
  - **segment-anything**: SAM project source code directory, corresponding to advance task 1
    - **mini_dataset**: Dataset containing 20 self-captured images
    - **output/mask**: Stores raw masks predicted by SAM's everything mode
    - **output/colored_mask**: Stores colored masks from output/mask
    - **output/multimask_results_single**: Stores results from SAM's multimask mode (just set one prompt dot)
    - **output/multimask_results**:  Stores results from SAM's multimask mode (set two prompt dot)
    - **scripts/amg.py:** Executable script for everything mode segmentation
    - **scripts/color_mask_map.py:** Script for applying colored mask overlays to segmented images
    - **scripts/multimask.py:** Script for multimask-out mode execution
  - **GroundingDINO**: GroundingDINO source code directory
  - **groundingdino_sam_pipline.py**: Pipeline script for advance task
  - **prompt.txt**: Prompt document for advance task 2
  - **outputs:**  Use SAM in combination with GroundingDINO for segmentation prediction results on a mini-dataset 

#### Install

- You can set up the environment according to the official installation guides:
  - UNet official project: https://github.com/milesial/Pytorch-UNet?tab=readme-ov-file
  - SAM official project: https://github.com/facebookresearch/segment-anything
  - GroundingDINO official project: https://github.com/IDEA-Research/GroundingDINO

#### Run

 This experiment is completed by executing the following commands. 

##### UNet

```bash
cd Pytorch-UNet

# train
python train.py -e 5 -l 0.00001
# predict
python predict.py --model <path/to/checkpoint>  -i <path/to/input/images>  -o <path/to/output/images>
```

##### SAM

```bash
cd segment-anything

# everything mode
python scripts/amg.py --checkpoint <path/to/checkpoint> --model-type <model_type> --input <image_or_folder> --output <path/to/output>

python color_mask_map.py # 处理成彩色蒙层重叠图

# multimask-output mode
python scripts/multimask.py
```

##### GroundingDINO+SAM

Before running the script, complete the configuration of the SAM and GroundingDINO projects and download the required models.

You can modify the initial parameters in groundingdino_sam_pipline.py according to your actual needs.

 If it is in offline mode, you need to pre-download bert-base-uncased, otherwise you won't be able to run GroundingDINO successfully. 

```python
config = 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
grounded_checkpoint = 'GroundingDINO/weights/groundingdino_swint_ogc.pth'
sam_version = 'vit_h'
sam_checkpoint = 'segment-anything/checkpoint/sam_vit_h_4b8939.pth'
input_folder = 'segment-anything/mini_dataset'
prompts_file = 'prompts.txt'
output_dir = 'outputs/groundingdino-sam'
device = 'cpu'
box_threshold = 0.3
text_threshold = 0.25
bert_base_uncased_path = 'bert-base-uncased'
```

```bash
#run
python groundingdino_sam_pipline.py
```



