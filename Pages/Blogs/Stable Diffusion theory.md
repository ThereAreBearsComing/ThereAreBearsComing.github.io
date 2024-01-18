# Stable Diffusion 初步理解
## SD模型基础结构原理

![image](https://github.com/ThereAreBearsComing/ThereAreBearsComing.github.io/assets/74708198/f5503d4f-47ec-4acb-be94-319206bc8b55)

平时正常图像存在在pixel space中， $\mathcal{E}$, $\mathcal{D}$ 为VAE。
上半部分是训练处理（黄色方框），即将图像转到潜空间学习为各种特征向量。而使用则是下半部分，即将图像从潜空间还原到像素空间。
下半部分，首先输入text传入交叉注意力层（cross-attenstion, QKV），网络结构为Unet。图片 $\mathcal{Z}$ 到 $\mathcal{Z}$ <sub>T - 1</sub> （此过程会让图像变大），T为迭代次数，即step

整个过程就是：

![image](https://github.com/ThereAreBearsComing/ThereAreBearsComing.github.io/assets/74708198/ca593c30-0b7e-491b-a6e1-f3fcb62f87e4)

## Clip 文本部分（Text Encoder）

即为训练好的一个CV中的图像处理网络：
<br>![image](https://github.com/ThereAreBearsComing/ThereAreBearsComing.github.io/assets/74708198/a05f6e11-8e02-44f1-ae4a-d139644cb92c)
<br>一种基于对比学习的多模态的图像识别预训练，基本流程是：
<br>![image](https://github.com/ThereAreBearsComing/ThereAreBearsComing.github.io/assets/74708198/f88b59b4-24b3-4a31-b280-8380fdfe3ac1)
<br>Embedding模型就是在Embedding LookUp处做一个对应，于是就知道你输入的词为那个Embedding，增加准确性。

## U-Net
![image](https://github.com/ThereAreBearsComing/ThereAreBearsComing.github.io/assets/74708198/cbd90cf4-cbee-4147-8980-35a982e2eb62)
<br>![image](https://github.com/ThereAreBearsComing/ThereAreBearsComing.github.io/assets/74708198/cc1f8f18-d9d3-4dd5-8227-03cf519d478e)
<br> 此部分将潜空间的噪声图片不断降噪，不断降噪，最终生成一张目标像素的清晰图片，即生成过程中预览出可以观测到，图片的变化
最后通过clip得到的condition就是通过cross attention与Unet结合

## VAE
![image](https://github.com/ThereAreBearsComing/ThereAreBearsComing.github.io/assets/74708198/a9b44d91-4358-459e-a33a-7c8a6aeeddb6)
<br>VAE就是Laten Space和Pixel Space转换的桥梁

## 简易流程图
![image](https://github.com/ThereAreBearsComing/ThereAreBearsComing.github.io/assets/74708198/0b8f708c-391d-4b93-b333-6023c1331db6)

## 常见训练方式
| 方式 | 原理 | 简介 |
| :-----: | :-----: | :-----: |
| Textual Inversion | 文本反演 | Embedding模型，就是那种几十k的EasyNegative |
| HyperNetWork | 超网络 | 很少 | 
| Finetune / Dreambooth | 直接微调 | Finetune是直接微调，Dreambooth是其改良版，通过引用之前模型生成的图片来防止模型忘记之前学习的东西，Finetune加正则化就是Dreambooth |
| LoRA系列 | LoRA/LyCORIS | LoRA只会更改小部分Unet，LyCORIS引入Convolution的抵制分解，能改到Resnet，所以更容易过拟合，对网络控制性更强 |













