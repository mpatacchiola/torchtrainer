torchtrainer
==============

High level wrapper to train pyTorch models on common datasets


Training a ResNet
-----------------

```
python3 ./trainer.py --gpu=0 --id="baseline_resnet34_cifar10_ep200" --arch="resnet34" --root="./" --data="../datasets/cifar10"
```
