import coremltools as ct
import onnx

# 載入 ONNX 模型
onnx_model_path = "YJ_B_5000_3layer_rgb2rgb_mlp_model.onnx"
mlmodel_output_path = "YJ_B_5000_3layer_rgb2rgb_mlp_model.mlmodel"

onnx_model = onnx.load(onnx_model_path)

# ✅ 使用新版 coremltools 的轉換方式
mlmodel = ct.convert(onnx_model, source="onnx")

# 儲存 CoreML 模型
mlmodel.save(mlmodel_output_path)
print(f"✅ CoreML 模型已儲存為：{mlmodel_output_path}")
