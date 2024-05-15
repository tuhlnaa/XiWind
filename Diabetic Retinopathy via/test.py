import cv2
import time
import torch
import pycuda.autoinit

import numpy as np
import seaborn as sns
import tensorrt as trt
import onnxruntime as ort
import pycuda.driver as cuda
import matplotlib.pyplot as plt

from typing import Union
from pathlib import Path
from enum import Enum, auto
from scipy.special import softmax
from torchvision import transforms
from functools import singledispatchmethod
from sklearn.metrics import precision_recall_curve, auc, classification_report, confusion_matrix

from model import DRModel
from dataloader import initialize_dataloader


class ONNXProvider(Enum):
    """Enum to define supported ONNX runtime providers."""
    CUDA = auto()
    TENSORRT = auto()
    CPU = auto()


class TRTEngine:
    """
    This class encapsulates the functionality for loading a TensorRT engine,
    allocating necessary resources, and performing inference.

    Attributes:
            engine (trt.ICudaEngine): The loaded TensorRT engine.
            context (trt.IExecutionContext): The execution context for the engine.
            inputs (list): A list of dictionaries containing the input buffer host and device pointers.
            outputs (list): A list of dictionaries containing the output buffer host and device pointers.
            bindings (list): A list of device buffer pointers required for engine execution.
            stream (cuda.Stream): The CUDA stream used for asynchronous operations.
    """

    def __init__(self, engine_path):
        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()

    def _load_engine(self, engine_path):
        """
        Loads the TensorRT engine from a serialized file.

        Parameters:
                engine_path (str): The file path to the serialized TensorRT engine.

        Returns:
                trt.ICudaEngine: The loaded TensorRT engine.
        """
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _allocate_buffers(self):
        """
        Allocates necessary buffers for input and output based on the engine's requirements.

        Returns:
                tuple: A tuple containing lists of input and output buffers, bindings, and the CUDA stream.
        """
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        for binding in self.engine:
            # Adjust for explicit batch size handling
            shape = self.engine.get_tensor_shape(binding)
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding))
            # Assuming `shape` already accounts for batch size explicitly
            size = trt.volume(shape)

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
        return inputs, outputs, bindings, stream

    def infer(self, input_data):
        """
        Runs inference using the TensorRT engine with the provided input data.

        Parameters:
                input_data (np.ndarray): The input data to be processed by the engine.

        Returns:
                np.ndarray: The output data from the inference process.
        """
        # Assuming a single input for simplicity. Extend as needed.
        np.copyto(self.inputs[0]['host'], input_data.ravel())  # Flatten input
        cuda.memcpy_htod_async(
            self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        # Execute the model
        self.context.execute_async_v2(
            bindings=self.bindings, stream_handle=self.stream.handle)
        # Copy the output from device to host
        cuda.memcpy_dtoh_async(
            self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        # Return the host output.
        return self.outputs[0]['host']


class BatchPredictor:
    """
    A class to perform batch predictions using either PyTorch or ONNX models.

    Attributes:
            model: The model to use for predictions. This can be a PyTorch or ONNX model.
            device: The device (CPU, CUDA, MPS) to run the model on.
            dataloader: A PyTorch DataLoader providing batches of images for prediction.
    """

    def __init__(self, model, device, dataloader):
        self.model = model
        self.device = device
        self.dataloader = dataloader

    @singledispatchmethod
    def predict_batch(self, model):
        raise NotImplementedError("Unsupported model type")

    @predict_batch.register
    def _(self, model: Union[torch.nn.Module, torch.jit.ScriptModule]):
        """
        Predict classes for PyTorch models in batches and measure the average prediction time.

        Args:
                model: A PyTorch model (either nn.Module or ScriptModule) for prediction.

        Returns:
                A tuple containing the list of predicted classes for all images and the average prediction time per image.
        """
        total_time = 0
        total_images, total_correct = 0, 0
        all_targets, all_outputs = [], []
        predictions = []

        for images, targets in self.dataloader:
            images, targets = images.to(self.device), targets.to(device)

            start_time = time.monotonic()
            with torch.no_grad():
                outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_time += time.monotonic() - start_time

            total_correct += (predicted == targets).sum().item()
            total_images += images.size(0)

            predictions.extend(predicted.tolist())
            all_targets.append(targets.cpu().numpy())
            all_outputs.append(outputs.softmax(dim=1).cpu().numpy())

        average_time = total_time / total_images
        return predictions, average_time, np.concatenate(all_targets), np.concatenate(all_outputs)

    @predict_batch.register
    def _(self, model: ort.capi.onnxruntime_inference_collection.InferenceSession):
        """
        Predict classes for ONNX models in batches and measure the average prediction time.

        Args:
                model: An ONNX InferenceSession object for prediction.

        Returns:
                A tuple containing the list of predicted classes for all images and the average prediction time per image.
        """
        total_time = 0
        total_images, total_correct = 0, 0
        predictions = []
        all_targets, all_outputs = [], []

        for images, targets in self.dataloader:
            images = images.numpy()

            start_time = time.monotonic()
            outputs = model.run(None, {model.get_inputs()[0].name: images})
            outputs = outputs[0]
            _, predicted = torch.max(torch.tensor(outputs), 1)
            total_time += time.monotonic() - start_time

            total_correct += (predicted == targets).sum().item()
            total_images += images.shape[0]

            predictions.extend(predicted.tolist())
            all_targets.append(targets.cpu().numpy())
            all_outputs.append(softmax(outputs))

        average_time = total_time / total_images
        return predictions, average_time, np.concatenate(all_targets), np.concatenate(all_outputs)

    @predict_batch.register
    def _(self, model: TRTEngine):
        """
        Predict classes for batches using a TensorRT engine and measure the average prediction time.

        Args:
                model: A TRTEngine object for prediction.

        Returns:
                A tuple containing the list of predicted classes for all images and the average prediction time per image.
        """
        total_time = 0
        total_images = 0
        predictions = []
        all_targets, all_outputs = [], []

        for images, targets in self.dataloader:
            images_np = images.numpy()

            start_time = time.monotonic()
            outputs = model.infer(images_np)
            predicted = np.argmax(outputs)
            total_time += time.monotonic() - start_time

            predictions.append(predicted)
            total_images += images_np.shape[0]
            outputs = np.expand_dims(outputs, axis=0)
            all_targets.append(targets.cpu().numpy())
            all_outputs.append(softmax(outputs))

        average_time = total_time / total_images
        return predictions, average_time, np.concatenate(all_targets), np.concatenate(all_outputs)


class ImagePredictor:
    """
    A class to perform single image predictions using either PyTorch or ONNX models.

    Attributes:
            model: The model to use for predictions, either a PyTorch model or an ONNX model.
            device: The device (CPU, CUDA, MPS) to run the model on.
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def preprocess_image(self, img_path):
        """
        Load an image, convert it to RGB, resize, and normalize it for prediction.

        Args:
                img_path: The path to the image file.

        Returns:
                A tensor representing the processed image, ready for model input.
        """
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])
        img_tensor = preprocess(img_rgb)
        return img_tensor.unsqueeze(0)

    @singledispatchmethod
    def predict_image(self, model, img_path):
        raise NotImplementedError("Unsupported model type")

    @predict_image.register
    def _(self, model: Union[torch.nn.Module, torch.jit.ScriptModule], img_path):
        """
        Predict the class for a single image using a PyTorch model.

        Args:
                model: A PyTorch model (either nn.Module or ScriptModule) for prediction.
                img_path: The path to the image file to predict.

        Returns:
                A tuple of the predicted class and the time taken for the prediction.
        """
        img_tensor = self.preprocess_image(img_path).to(self.device)

        start_time = time.monotonic()
        with torch.no_grad():
            outputs = model(img_tensor)
            print(outputs, type(outputs))
        _, predicted = torch.max(outputs, 1)
        elapsed_time = time.monotonic() - start_time

        return predicted.item(), elapsed_time

    @predict_image.register
    def _(self, model: ort.capi.onnxruntime_inference_collection.InferenceSession, img_path):
        """
        Predict the class for a single image using an ONNX model.

        Args:
                model: An ONNX InferenceSession object for prediction.
                img_path: The path to the image file to predict.

        Returns:
                A tuple of the predicted class and the time taken for the prediction.
        """
        img_tensor = self.preprocess_image(img_path).numpy()

        start_time = time.monotonic()
        outputs = model.run(None, {model.get_inputs()[0].name: img_tensor})
        outputs = outputs[0]
        print(outputs, type(outputs))
        _, predicted = torch.max(torch.tensor(outputs), 1)
        elapsed_time = time.monotonic() - start_time

        return predicted.item(), elapsed_time

    @predict_image.register
    def _(self, model: TRTEngine, img_path):
        """
        Predict the class for a single image using a TensorRT engine.

        Args:
                model: A TRTEngine object for prediction.
                img_path: The path to the image file to predict.

        Returns:
                A tuple of the predicted class and the time taken for the prediction.
        """
        img_tensor = self.preprocess_image(img_path).numpy()

        start_time = time.monotonic()
        outputs = model.infer(img_tensor)
        print(outputs, type(outputs))
        predicted = np.argmax(outputs)
        elapsed_time = time.monotonic() - start_time

        return predicted, elapsed_time


class ModelLoader:
    """
    A class for loading machine learning models, supporting both PyTorch and ONNX formats.

    Attributes:
            model_path: The file path of the model to load.
            device: The device (CPU, CUDA, MPS) to load the model onto.
            provider_preference: Preferred ONNX runtime provider for ONNX models.
    """

    def __init__(self, model_path, device, provider_preference=ONNXProvider.CPU):
        self.model_path = model_path
        self.device = device
        self.provider_preference = provider_preference

    def load(self):
        if self.model_path.endswith('.onnx'):
            return self._load_onnx_model()
        elif self.model_path.endswith('.trt'):
            return self._load_tensorrt_engine()
        else:
            return self._load_pytorch_model()

    def _load_onnx_model(self):
        """
        Loads an ONNX model with specified runtime providers.

        Returns:
                An ONNX InferenceSession with the loaded model.
        """
        providers = []
        if torch.cuda.is_available():
            if self.provider_preference == ONNXProvider.CUDA:
                providers.append('CUDAExecutionProvider')
            elif self.provider_preference == ONNXProvider.TENSORRT:
                providers.append('TensorrtExecutionProvider')
        if not providers:
            providers.append('CPUExecutionProvider')

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        return ort.InferenceSession(self.model_path, sess_options, providers=providers)

    def _load_tensorrt_engine(self):
        """
        Loads a TensorRT engine for inference.

        Returns:
                An instance of TRTEngine with the loaded engine.
        """
        return TRTEngine(self.model_path)

    def _load_pytorch_model(self):
        """
        Loads a PyTorch model (.pth or .pt file) for inference.

        Returns:
                The loaded PyTorch model.

        Raises:
                ValueError: If the model file format is not .pth or .pt.
        """
        if self.model_path.endswith('.pth') or self.model_path.endswith('.pt'):
            model = self._load_standard_or_scripted_pytorch_model()
        else:
            raise ValueError("Unsupported model file format.")
        model.eval()
        return model

    def _load_standard_or_scripted_pytorch_model(self):
        """
        Helper method to load standard or scripted PyTorch models.

        Returns:
                The loaded PyTorch model, optimized for inference if applicable.
        """
        if self.model_path.endswith('.pth'):
            model = DRModel().to(self.device)
            checkpoint = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
        elif self.model_path.endswith('.pt'):
            model = torch.jit.load(self.model_path, map_location=self.device)
            model = torch.jit.optimize_for_inference(model)
        return model


class ModelEvaluator:
    def __init__(self, targets, predictions):
        self.targets = targets
        self.predictions = predictions
        self.binary_predictions = predictions.argmax(axis=1)

    def plot_precision_recall_curve(self):
        precision, recall, _ = precision_recall_curve(
            self.targets, self.predictions[:, 1])
        auc_score = auc(recall, precision)
        fig, ax = plt.subplots(dpi=150, layout='tight')
        ax.plot(recall, precision, marker='.', label=f'AUC = {auc_score:.2f}')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend()
        plt.savefig('Precision-Recall Curve.png')

    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.targets, self.binary_predictions)
        fig, ax = plt.subplots(dpi=150, layout='tight')
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["No Diabetic Retinopathy",
                                 "Diabetic Retinopathy"],
                    yticklabels=["No Diabetic Retinopathy", "Diabetic Retinopathy"])
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('Actual Label')
        ax.set_title('Confusion Matrix')
        plt.savefig('Confusion Matrix.png')

    def evaluate(self):
        report = classification_report(
            self.targets,
            self.binary_predictions,
            target_names=['No Diabetic Retinopathy', 'Diabetic Retinopathy'],
            zero_division=0)

        self.plot_confusion_matrix()
        self.plot_precision_recall_curve()
        return report


if __name__ == "__main__":
    root_dir = Path(__file__).parent
    test_data_dir = root_dir / "dataset" / "test"
    image_path = "DR.jpg"

    model_path = 'model_epoch_40.pth'               # Torch model
    model_path = "MobileNetV2_scripted_model.pt"    # Torch Script model
    model_path = "MobileNetV2_model.onnx"          # ONNX model
    # model_path = "MobileNetV2_model.trt"  # TensorRT engine

    device = ("cuda"
              if torch.cuda.is_available()
              else "mps"
              if torch.backends.mps.is_available()
              else "cpu")

    model_loader = ModelLoader(
        model_path, device, provider_preference=ONNXProvider.CUDA)
    model = model_loader.load()

    # Single Image Prediction
    image_predictor = ImagePredictor(model, device)
    prediction, time_taken = image_predictor.predict_image(model, image_path)
    print(
        f"Predicted class: {prediction}, Time taken: {time_taken:.6f} seconds")
    # quit()
    # Batch Prediction
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    test_dataloader, _ = initialize_dataloader(
        test_data_dir,
        batch_size=1,
        transform=val_transform,
        shuffle=False)

    batch_predictor = BatchPredictor(model, device, test_dataloader)
    batch_predictions, avg_time, targets, outputs = batch_predictor.predict_batch(
        model)
    evaluator = ModelEvaluator(targets, outputs)
    report = evaluator.evaluate()
    print(report)
    print(
        f"Batch predictions shape: {len(batch_predictions)}, Average time per image: {avg_time:.6f} seconds")


'''
Average time per image, Python 3.11, torch 2.2.0+cu121
[GTX1080 Ti] 
PyTorch (CUDA 12.1): 6.3~5.3ms
TorchScript (CUDA 12.1): 3.6~2.8 ms
ONNX (CUDA 11.8): 4.1~3.2 ms
TensorRT (CUDA 11.8): 1.5~1.2 ms
'''
