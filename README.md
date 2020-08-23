# MGA: Motion Guided Attention for Video Salient Object Detection, ICCV 2019
<p align="center">
  <img src="MGA_results.png" width="1000" title="MGA_results">
</p>

0. If you want to compare with our method, a simple way is to download the \*.tar.gz files. These tar.gz files contain saliency maps predicted by our method without any post-processing like CRF. You could evaluate these saliency maps with your own evaluation code. It is suggested to uncompress them on Ubuntu:
```
cat DAVIS-SaliencyMap.tar.gz* | tar -xzf -
tar -zxvf FBMS-SaliencyMap.tar.gz 
tar -zxvf ViSal-SaliencyMap.tar.gz
```
If you need the results of MGA (without CRF) tested on DAVSOD, please download them from https://pan.baidu.com/s/1mv7mi_0XS3W237tAH2gNgg 
password：9lm7 

1. If you want to run our trained MGA network, first, you need to correctly install FlowNet 2.0. The performance of our method could be affected by the quality of optical flow images. Notice that FlowNet 2.0 contains many variants, please choose the one with the highest accuracy. It is suggested to use the implementation by NVIDIA: https://github.com/NVIDIA/flownet2-pytorch. To run this FlowNet 2.0, it is good to use Python 3.6. To run our trained model, it is better to use Python 2 & Pytorch 0.4.0/0.4.1.

1-1. Be careful about the input order of video frames that are used for flow estimation. It is better to transform the input frames into the same shape as the training samples of FlowNet 2.0. Visualize optical flows and transform them into RGB images.

2. Have a look at the directory './dataset', and you will know how the data is organized. Create new sub-directory for your own dataset, and place RGB images and visualized optical flows under the corresponding directory. You may also read and modify 'dataloaders' according to how you organize the data.

3. Download our trained model weights 'MGA_trained.pth'. Google Download Link: https://drive.google.com/file/d/1tuG_S5nIAEfigKNPPsOo7SG-iIWo3T4N/view?usp=sharing Baidu Download Link：https://pan.baidu.com/s/13XT8xLnwFH13dMU89vkUEQ Password：a6pc Place 'MGA_trained.pth' under the same directory as 'inference.py'. Then try:
```
python inference.py
```
The above command should generates a new directory named './results'. The command should also predict a saliency map, since we have put an input sample in './dataset'.

4. If you want to train our method from scratch, please read the proposed multi-task training scheme in our paper https://arxiv.org/pdf/1909.07061 carefully. And then implement your own training code. If you feel this repository helpful, please cite our paper.

Please email me via lhaof@foxmail.com, if you need helps. If you feel this repository helpful, please cite the following paper
```
@inproceedings{li2019motion,
	title={Motion Guided Attention for Video Salient Object Detection},
	author={Li, Haofeng and Chen, Guanqi and Li, Guanbin and Yu Yizhou},
	booktitle={Proceedings of International Conference on Computer Vision},
	year={2019}
}
```

5. Q&A

Q: missing key:  bn1.num_batches_tracked

A: The keys with 'num_batches_tracked' can be ignored. See the following code.
```
def load_pretrain_model(net, model_path):
    net_keys = list(net.state_dict().keys())
    model = torch.load(model_path)
    model_keys = list(model.keys())
    # clean keys
    model_keys = [key for key in model_keys if not key.endswith('num_batches_tracked')]
    i = 0
    while i < len(model_keys):
        model_key_i = model_keys[i]
        net_key_i = net_keys[i]
        assert net.state_dict()[net_key_i].shape == model[model_key_i].shape, ('{}.shape: {}, {}.shape: {}'.format(net_key_i, net.state_dict()[net_key_i].shape, model_key_i, model[model_key_i].shape))
        net.state_dict()[net_key_i].copy_(model[model_key_i].cpu())
        i += 1
    return net

```
