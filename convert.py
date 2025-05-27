import onnx
from onnx_tf.backend import prepare
import coremltools as ct
import os

onnx_model_path = "R_three_rgb2rgb_mlp_model.onnx"
tf_model_path = "tf_model"
mlmodel_output_path = "R_three_rgb2rgb_mlp_model.mlmodel"

print("🚀 載入 ONNX 模型...")
onnx_model = onnx.load(onnx_model_path)
tf_rep = prepare(onnx_model)
tf_rep.export_graph(tf_model_path)
print("✅ ONNX 已轉為 TensorFlow SavedModel")

print("⚙️ 轉換為 Core ML 模型中...")
mlmodel = ct.convert(tf_model_path, source="tensorflow")
mlmodel.save(mlmodel_output_path)
print(f"✅ Core ML 模型儲存為：{mlmodel_output_path}")
