# HuBMAP-HPA-Hacking-the-Human-Body
When you think of “life hacks,” normally you’d imagine productivity techniques. But how about the kind that helps you understand your body at a molecular level? It may be possible! Researchers must first determine the function and relationships among the 37 trillion cells that make up the human body. A better understanding of our cellular composition could help people live healthier, longer lives.

- credits for augmentation: hengck https://www.kaggle.com/hengck23
- thresholding experiments : otsu, li, mean, median from scikit image Library
- Coat + Daformer :  encoder => coat_lite_medium() decoder => daformer_conv3x3 pretrained encoder weights from timm
- Coat + Daformer credits : https://www.kaggle.com/datasets/alincijov/hubmap-coat
- image size used : 1024 
- Mix Up Sampling idea from hengck https://www.kaggle.com/hengck23
```
x = self.mixing * F.interpolate(
            x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False
        ) + (1 - self.mixing) * F.interpolate(
            x, scale_factor=self.scale_factor, mode="nearest"
        )
```
- loss function : 0.5*Dice Loss + 0.5*BCELoss
- half precision training
- best inference : https://www.kaggle.com/code/bibhabasumohapatra/rank-80-coat-inference-final-for-lb-part-3
