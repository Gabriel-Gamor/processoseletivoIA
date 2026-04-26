import tensorflow as tf
import os

#insira seu código aqui

print("Carregando o modelo salvo (model.h5):")
model = tf.keras.models.load_model('model.h5')

print("Configurando o conversor TensorFlow Lite:")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

print("Aplicando otimização: Dynamic Range Quantization:")
converter.optimizations = [tf.lite.Optimize.DEFAULT]

print("Convertendo o modelo:")
tflite_quant_model = converter.convert()

print("Salvando o modelo otimizado em model.tflite:")
with open('model.tflite', 'wb') as f:
    f.write(tflite_quant_model)

print("Otimização concluída com sucesso!")

# VVerificar a eficiência por meio do tamanho
h5_size = os.path.getsize('model.h5') / 1024
tflite_size = os.path.getsize('model.tflite') / 1024
print(f"\n--- Comparativo de Tamanho ---")
print(f"Modelo Original (.h5): {h5_size:.2f} KB")
print(f"Modelo Otimizado (.tflite): {tflite_size:.2f} KB")