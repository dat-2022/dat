# DAT

The repo contains the training code for paper [Distributed Adversarial Training to Robustify Deep Neural Networks at Scale](https://openreview.net/pdf?id=Srgg_ULj9gq).
Source code is adapted from:

[You Only Propagate Once: Accelerating Adversarial Training via Maximal Principle](https://github.com/a1600012888/YOPO-You-Only-Propagate-Once)

[Pytorch-lamb](https://github.com/cybertronai/pytorch-lamb)

Train with Imagenet with DAT-PGD :

```
python main.py --dataset imagenet \
               --batch-size <BATCH SIZE> \
               --world-size <NUMBER OF NODES> \
               --rank <RANK> \
               --dist-url "tcp://<MASTER IP>:<PORT>" \
               --dataset-path <PATH TO IMAGENET>\
               --num-epochs 30 \
               --output-dir <OUTPUT DIR> \
               --lr 0.01 
```

Train with Imagenet with DAT-FGSM :

```
python main.py --dataset imagenet \
               --batch-size <BATCH SIZE> \
               --world-size <NUMBER OF NODES> \
               --rank <RANK> \
               --dist-url "tcp://<MASTER IP>:<PORT>" \
               --dataset-path <PATH TO IMAGENET>\
               --num-epochs 30 \
               --output-dir <OUTPUT DIR> \
               --lr 0.01 \
               --fast
```

Our pretrianed Imagenet models are under [here](https://www.dropbox.com/sh/bbtyxc8fg8q6sbz/AAB_9FYPhUOvgW7a2yxDN_1Ya?dl=0).