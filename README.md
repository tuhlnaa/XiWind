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
| [MobileNetV2<sub>base</sub>](https://github.com/tuhlnaa/Kaggle-Side-Project-XiWind/tree/main/Diabetic%20Retinopathy%20via/inference_model) |        224×224        |       6.3~5.3        |         3.6~2.8          |      4.1~3.2      |        1.5~1.2        |             0.97              |            0.97            |

