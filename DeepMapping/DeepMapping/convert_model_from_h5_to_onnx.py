import os
import shutil
import tensorflow as tf 

"""This script is used to convert the h5 model into onnx format,
if you want to use onnxrunutime as backend.

You are required to install tf2onnx by using pip.
"""

for root, dirs, files in os.walk("models/nas/tpch-s1/", topdown=False):
   for name in files:
      if '.h5' in name:
         model_name = name.split('.')[0]
         # h5 file 
         model_path = os.path.join(root, name)

         model = tf.keras.models.load_model(model_path, compile=False)
         # save in pb 
         model.save(os.path.join(root, model_name))
         cmd = "python -m tf2onnx.convert --saved-model {} --output {}.onnx".format(os.path.join(root, model_name),
                                                                                        os.path.join(root, model_name))
         os.system(cmd)
         shutil.rmtree(os.path.join(root, model_name))
         print(root, name, cmd)
