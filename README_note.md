#### 问题

- 网络结构

- 输入的图像数据格式？

- 输入的点云数据格式？点数限制？

- 图像数据进行了哪些预处理操作？

- backbone、neck、head的输入输出张量大小？

- gt是什么格式？

- 损失都包含呢些内容？

- bevpool怎么进行的优化

- 都有哪些模块？

  - Encoder

    - 图像的backbone：**SwinTransformer、ResNet**
    - 图像的neck：**GeneralizedLSSFPN、SECONDFPN**
    - 图像的vtransformer：**DepthLSSTransform、LSSTransform、AwareDBEVDepth**
    - 激光的voxelize
    - 激光的backbone：**SparseEncoder、PointPillarsEncoder、RadarEncoder**

    - Fuser
      - ConvFuser

  - Decoder

    - backbone：**SECOND、GeneralizedResNet**
    - neck：**SECONDFPN、LSSFPN**

  - Head
    - TransFusionHead、CenterHead


### NECK

#### GeneralizedLSSFPN

输入

- [6, **192,** 32, 88]
- [6, **384,** 16, 44]
- [6, **768,** 8, 22]

输出

- [6, 256, 32, 88]
- [6, 256, 16, 44]  下游vtransform不使用

**GeneralizedLSSFPN和PFN什么区别？**

- **GeneralizedLSSFPN‌**可以实现跨层拼接，FPN仅使用单层特征图
- GeneralizedLSSFPN‌**使用通道拼接（concat）**，FPN使用元素相加

![img](https://i-blog.csdnimg.cn/direct/d5cae141515d46dfbee0b3754fd4eafe.png)



### vtransform

#### LSSTransform

- **生成3D视锥**

  可以这么理解，原本是二维平面空间，对每个点扩展一个深度的概念，此时，就变为了3d空间，理解为3d网格，每个点的坐标是（x，y，d），d表示深度方向上的坐标

  根据传感器内外参将3D网格点投影到ego坐标系中（图像坐标系-->相机坐标系-->激光/ego坐标系）

- **图像编码生成3D图像特征**

  - 深度估计，一个卷积层实现，输出的通道数是D+C，D是深度采样空间的维度，C是体素特征的维度{(B * N, C, fH, fW) -->  (B\*N, C, D, fH, fW)}

- **bev pool**，输出BEV特征图

  功能：将体素空间的特征根据其空间坐标聚合到bev网格上，形成二维bev特征图

  - 计算体素特征对应的bev网格索引

  - 过滤网格之外的点

  - 将体素特征根据其空间索引聚合到bev网格上
    - 将同一个bev网格单元的点聚集在一起
    - 将统一网格单元内点的特征进行聚合（通常是相加、均值等）

  - 沿z轴进行特征合并



#### DepthLSSTransform

**点云投影 → 深度特征生成 → 空间变换 → 特征融合 → BEV聚合 → 输出**

大体流程与LSSTransform一致，新增一个**深度特征图**，系点云投影到图像中生成深度图，并将该深度图和图像特征共同送入深度估计网络层中。

**问题：点云怎么来的？**	











Transfusion：基于Transformer的检测头

CenterHead：基于中心点（Center-based）的检测头，类似于CenterNet风格的检测头

### 参考

https://blog.csdn.net/2301_77102499/article/details/138083499?fromshare=blogdetail&sharetype=blogdetail&sharerId=138083499&sharerefer=PC&sharesource=qq_43644906&sharefrom=from_link