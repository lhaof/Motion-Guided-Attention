# MGA: Motion Guided Attention for Video Salient Object Detection, ICCV 2019

0. If you want to compare with our method, a simple way is to download the \*.tar.gz files. These tar.gz files contain saliency maps predicted by our method without any post-processing like CRF. You could evaluate these saliency maps with your own evaluation code. It is suggested to uncompress them on Ubuntu:
```
cat DAVIS-SaliencyMap.tar.gz* | tar -xzf -
tar -zxvf FBMS-SaliencyMap.tar.gz 
tar -zxvf ViSal-SaliencyMap.tar.gz
```

1. If you want to run our trained MGA network, first, you need to correctly install FlowNet 2.0. The performance of our method could be affected by the quality of optical flow images. 
