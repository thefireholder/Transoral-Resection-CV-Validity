
4 models:
maskrcnn
yolo
sam3
monai - not in this computer

what I need to generate regarding these 4 models are following figures. 

Figure 1: training curve (precision and recall showing result for training and validation data only) and segmentation result in image (like surgery iamge) next to labeled image for all 4 models for comparison
Figure 2: PR curve for masking for each class for all 4 models. see /u/sl257/shared_data/train_script/yolov12_project/runs/segment/1200images/1200images_test/MaskPR_curve.png
Figure 3: map table. showing mAP value for per class and aggregaete
Figure 4: mAP Dice IoU score bar graph for aggregate across 4 models for showing difference.
Figure 5: Finally how much dataset do we actually need to train well: mAP result when trained with 75 150 300 600 900 1200 data

when I say map I only mean segementaion mask and not bounding box.

last figure is not needed now but might be in need later, so maybe consider writing code to obtain those data separately

I need the result -- numerical values to be in one spot so that I can always rerun matplotlib to change the format of the figure later. the numbers are scattered across and might be missing. if it is missing it needs to be generated ideally using the same seed





following is disorganized but roughly organized folder of what I have:

maskrcnn model
/projects/illinois/cimed/bts/gayed/data/train_script/maskrcnn_project/run_output/maskrcnn_resnet50_fpn_best.pth

for maskrcnn model script might want to study this folder:


yolo model script and path
/u/sl257/shared_data/train_script/yolov12_project/yolo11n_model_1200images.slurm

/u/sl257/shared_data/train_script/yolov12_project/yolo11n_model.py

yolo result path:
/u/sl257/shared_data/train_script/yolov12_project/runs/segment/1200images

sam3
/u/sl257/shared_data/train_script/SAM2_project/sam2/job_sam3.slurm

sam3 result path:
/u/sl257/shared_data/train_script/SAM2_project/sam2/result/sam3/1200images

I am aware that sam3 does not have training curve because it is not 


Dataset
/u/sl257/shared_data/processed_data/undergrad_data/JML619-split.json
/u/sl257/shared_data/processed_data/undergrad_data/JSR-split.json

JML619 split = train 430 val 92 test 93
JSR split = train 303 val 65 test 65
Total = train 733 val 157 test 158 = 1048

/u/sl257/shared_data/processed_data

it might require re running some of the code to generate things like training curve. use seed please in that case. have the code ready to run in this folder somewhere so I can run it myself. don't run it and leave that value as n/p for now for generating curves and tables and so on

so some of these contain result such as 


so following is what I need:
in this readme folder I need the result in one space with same formatting across each model, with also another note that points to where each result was found from which directory,

and then code that uses these unified data to plot the result. 

the plotted figure in one spot

it must be easy to modify the code 

please ask clarification question if you have

and if you could just organize only the relevant code in to one place in this directory so I can run it, it would be appreaciated. I cannot run it in this.

I also put the result I gathered from different places (original place they are found can be found in this computer save for monai-result) here: /u/sl257/shared_data/result/some disorganized results/1200 images

actually let's move this content and rewrite readme and have me look at it first before running it to confirm that it has every instruction i need


