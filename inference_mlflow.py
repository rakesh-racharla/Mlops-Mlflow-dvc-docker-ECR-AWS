from pprint import pprint
from mlflow.tracking import MlflowClient
import mlflow

mlflow.set_tracking_uri("sqlite:///mlruns.db")

client = MlflowClient()
'''
for rm in client.list_registered_models():
    pprint(dict(rm), indent=4) '''
#get model file path of latest model
path = str(client.list_registered_models()[0]).split(",")[10].split("'")[1]

path +'/model.onnx'

##### loading model
import onnxruntime as rt
import numpy
sess = rt.InferenceSession(path +'/model.onnx')

input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
inf_data = numpy.array([1.8,9.88,0,2.6,0.098,25,67,0.9968,3.2,0.68,9.8])
pred_onx = sess.run([label_name], {input_name: numpy.expand_dims(inf_data.astype(numpy.float32),axis=0)})[0]
pred_onx
