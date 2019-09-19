# MGA: Motion Guided Attention for Video Salient Object Detection, ICCV 2019

0. If you want to compare with our method, a simple way is to download the \*.tar.gz files. These tar.gz files contain saliency maps predicted by our method without any post-processing like CRF. You could evaluate these saliency maps with your own evaluation code. It is suggested to uncompress them on Ubuntu:
```
cat DAVIS-SaliencyMap.tar.gz* | tar -xzf -
tar -zxvf FBMS-SaliencyMap.tar.gz 
tar -zxvf ViSal-SaliencyMap.tar.gz
```

1. If you want to run our trained MGA network, first, you need to correctly install FlowNet 2.0. The performance of our method could be affected by the quality of optical flow images. Notice that FlowNet 2.0 contains many variants, please choose the one with the highest accuracy. It is suggested to use the implementation by NVIDIA: https://github.com/NVIDIA/flownet2-pytorch. 

1-1. Be careful about the input order of video frames that are used for flow estimation. It is better to transform the input frames into the same shape as the training samples of FlowNet 2.0. Visualize optical flows and transform them into RGB images.

2. Have a look at the directory './dataset', and you will know how the data is organized. Create new sub-directory for your own dataset, and place RGB images and visualized optical flows under the corresponding directory. You may also read and modify 'dataloaders' according to how you organize the data.

3. Download our trained model weights 'MGA_trained.pth'. Download Link：https://pan.baidu.com/s/13XT8xLnwFH13dMU89vkUEQ Password：a6pc Place 'MGA_trained.pth' under the same directory as 'inference.py'. Then try:
```
python inference.pth
```
The above command should generates a new directory named './results'. The command should also predict a saliency map, since we have put an input sample in './dataset'.

4. If you want to train our method from scratch, please read the proposed multi-task training scheme in our paper carefully. And then implement your own training code. If you feel this repository helpful, please cite our paper.
