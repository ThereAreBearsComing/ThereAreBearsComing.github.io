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

* Textual Inversion:
  * 在Embedding LookUp处做文章， 其中 <*> 为 $\mathcal{S}$<sub> * </sub> 配有的专用语义向量，当Clip中有$\mathcal{S}$语义，就会直接找语义向量 <\*>
  * SD-WebUI对其的实现位置在 sd_hijack_clip.py 和 textual_Inversion.py中（源码）
<br> ![image](https://github.com/ThereAreBearsComing/ThereAreBearsComing.github.io/assets/74708198/4cf050ad-3ce3-4834-9495-3ce3914770f2)

* Finetune / Dreambooth
  * Dreambooth本身是Fintune的一类加了小Trick的特殊变种
  * 训练方式秋叶的autodl训练脚本（在线训练、Linux下比较方便）
  * Naifu-Diffusion（Finetune）
  * HCP-diff
  * Kohya的训练脚本/UI
  * 不推荐Dreambooth插件，很久没更新了
  * 推荐训练参数：
    ![image](https://github.com/ThereAreBearsComing/ThereAreBearsComing.github.io/assets/74708198/728e8150-0808-4660-b401-6e11c41fd1d7)
    <br>![image](https://github.com/ThereAreBearsComing/ThereAreBearsComing.github.io/assets/74708198/0c3684ec-bd67-4dc1-8ff2-536618bfac0e)
    <br>![image](https://github.com/ThereAreBearsComing/ThereAreBearsComing.github.io/assets/74708198/06f0b2e3-8a0e-4bb4-829e-5d3bf356c9b4)

* LoRA原理
  * 理解为在Unet linear部分开了分支在低降维微调，加回去后再升维回去
  * <br>![image](https://github.com/ThereAreBearsComing/ThereAreBearsComing.github.io/assets/74708198/6d6d1639-1f93-412f-8290-26a0e841a67d)
    <br>![image](https://github.com/ThereAreBearsComing/ThereAreBearsComing.github.io/assets/74708198/46cf36df-4e6d-496b-b93c-09379818a3aa)
    <br>![image](https://github.com/ThereAreBearsComing/ThereAreBearsComing.github.io/assets/74708198/8891c63d-7ec8-49aa-8c4f-bcec2826c490)
  * LyCORIS为进阶版LoRA，支持了卷积调整，还会细分为LoCon（支持Conv的原版LoRA），LoHa（哈达玛积），(AI)^3  （经验是LoCon不好用的话直接微调大模型，即finetune）
  * 训练部分：
  * <br>![image](https://github.com/ThereAreBearsComing/ThereAreBearsComing.github.io/assets/74708198/d3a9f6d8-f579-47a7-accd-38864b8cfbc4)
    <br>![image](https://github.com/ThereAreBearsComing/ThereAreBearsComing.github.io/assets/74708198/3e170240-ee97-4abe-99c0-cc4c8b814171)
    <br>![image](https://github.com/ThereAreBearsComing/ThereAreBearsComing.github.io/assets/74708198/82293713-f3cc-4cbf-a324-11bde91d7e4d)

## 训练
* 一般的数据增强（LoRA训练很少用）
  * ColorSkew随机调整色彩饱和度
  * Black and White 黑白化
  * Crop 裁剪

* 利用AI进行数据增强
  * 图生图
  * 线稿生成
  * **基于一有模型进行数据增强**

* 学习率/Scheduler
  * 使用 余弦/带重启的余弦
  * 优先尝试低学习率，在不断提高学习率查看结果

* 优化器Optimizer
  * AdamW(default), AdamW8bit, Lion, SGDNesterov, SGDNesterov8bit, DAdapation, AdaFactor
  * 推荐:
    * AdamW8bit(常用)
    * Lion更容易过拟合，学习率需要低一些，一般 1/3
    * 一些自适应学习率又花钱（占用显存更多）

* 不对Text Encoder进行训练
  * 如果训练Text Encoder，模型很容易过拟合
  * 训练Text Encoder同时需要减小学习步数/学习率

* 加正则（Dreambooth）
  * Dreambooth本身就可以减少过拟合

* 补救过拟合模型
  * 与其他模型进行融合、
  * 使用时假如ControlNet进行额外控制
  * 针对部分TE过拟合的模型，可以尝试使用其他模型的TE覆盖











