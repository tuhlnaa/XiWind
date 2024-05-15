# Kaggle Side Project | XiWind
Coming soon

<br><br>

# [糖尿病視網膜病變的診斷](https://www.kaggle.com/datasets/pkdarabi/diagnosis-of-diabetic-retinopathy)

<div style="display:flex; justify-content:space-around; align-items:center;">
  <img src="https://github.com/tuhlnaa/Kaggle-Side-Project-XiWind/blob/main/Diabetic%20Retinopathy%20via/test_image/DR.jpg" width="17%" />
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="5%" alt="" />
  <img src="https://github.com/tuhlnaa/Kaggle-Side-Project-XiWind/blob/main/Diabetic%20Retinopathy%20via/test_image/No_DR.jpg" width="17%" />
  <p><em>Figure 1: Diabetic Retinopathy, No Diabetic Retinopathy</em></p>
</div>

**Speed <sub>GTX 1080 Ti</sub> (ms): batch-size 1**

| Model                                                                                                                                      | size<br><sup>(pixels) | PyTorch<br><sup>(ms) | TorchScript<br><sup>(ms) | ONNX<br><sup>(ms) | TensorRT<br><sup>(ms) | F1-Score<br><sup>weighted avg | F1-Score<br><sup>macro avg |
| ------------------------------------------------------------------------------------------------------------------------------------------ | :-------------------: | :------------------: | :----------------------: | :---------------: | :-------------------: | :---------------------------: | :------------------------: |
| [MobileNetV2<sub>base</sub>](https://github.com/tuhlnaa/Kaggle-Side-Project-XiWind/tree/main/Diabetic%20Retinopathy%20via/inference_model) |        224×224        |         5.5          |           3.2            |        3.5        |          1.3          |             0.97              |            0.97            |

<br><br>

# [室內場景深度估測](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)

<div style="display:flex; justify-content:space-around; align-items:center;">
  <img src="https://raw.githubusercontent.com/tuhlnaa/Kaggle-Side-Project-XiWind/main/DenseDepth/images/depth_maps.png" width="34%" />
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="5%" alt="" />
  <img src="https://github.com/tuhlnaa/Kaggle-Side-Project-XiWind/blob/main/DenseDepth/images/Image.png" width="34%" />
  <p><em>Figure 2: Image, Depth maps</em></p>
</div>

**Speed <sub>GTX 1080 Ti</sub> (ms): batch-size 1**

| Model                                                                                              | size<br><sup>(pixels) | PyTorch<br><sup>(ms) | TorchScript<br><sup>(ms) | ONNX<br><sup>(ms) | TensorRT<br><sup>(ms) | $\delta _{1}$ | rmse<br> |
| -------------------------------------------------------------------------------------------------- | :-------------------: | :------------------: | :----------------------: | :---------------: | :-------------------: | :-----------: | :------: |
| [DenseDepth](https://drive.google.com/drive/folders/1Y3lscMncLRFB8o2N4WGuozu64mUveeOC?usp=sharing) |        480×640        |          24          |            90            |        50         |          35           |     0.81      |  0.071   |




