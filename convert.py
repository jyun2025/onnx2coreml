import coremltools as ct
import onnx

onnx_model_path = "YJ_B_5000_3layer_rgb2rgb_mlp_model.onnx"
mlmodel_output_path = "YJ_B_5000_3layer_rgb2rgb_mlp_model.mlmodel"

onnx_model = onnx.load(onnx_model_path)
mlmodel = ct.converters.onnx.convert(model=onnx_model)
mlmodel.save(mlmodel_output_path)
print(f"✅ 成功轉換並儲存：{mlmodel_output_path}")
