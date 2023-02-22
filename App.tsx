import { StatusBar } from 'expo-status-bar';
import { Alert, Button, StyleSheet, Text, View } from 'react-native';

import * as ort from 'onnxruntime-react-native';
import { Asset } from 'expo-asset';

let myModel: ort.InferenceSession;

async function loadModel() {
  try {
    if (typeof BigInt64Array === 'undefined') {
      global.BigInt64Array = require('big-integer')
    }
      

    var bigInt64 = new BigInt64Array(2);
    console.log('bigInt64: ', bigInt64); // 2

    const assets = await Asset.loadAsync(require('./assets/rf_iris.onnx'));
    const modelUri = assets[0].localUri;
    if (!modelUri) {
      Alert.alert('failed to get model URI', `${assets[0]}`);
    } else {
      myModel = await ort.InferenceSession.create(modelUri);
      Alert.alert(
        'model loaded successfully',
        `input names: ${myModel.inputNames}, output names: ${myModel.outputNames}`);
    }
  } catch (e) {
    Alert.alert('failed to load model', `${e}`);
    throw e;
  }
}

async function runModel() {
  try {
    if (typeof BigInt64Array === 'undefined') {
      global.BigInt64Array = require('big-integer')
    }
    let inputData = Float32Array.from([5.5, 2.3, 4. , 1.3]);
    const feeds = {
      float_input: new ort.Tensor('float32', inputData, [1, 4]),
    }
    console.log(feeds)
    const fetches = await myModel.run(feeds);
    const output = fetches['probabilities'];
    if (!output) {
      Alert.alert('failed to get output', `${myModel.outputNames[0]}`);
    } else {
      Alert.alert(
        'model inference successfully',
        `output shape: ${output.dims}, output data: ${output.data}`);
    }
  } catch (e) {
    Alert.alert('failed to inference model', `${e}`);
    throw e;
  }
}

export default function App() {
  return (
    <View style={styles.container}>
      <Text>using ONNX Runtime for React Native</Text>
      <Button title='Load model' onPress={loadModel}></Button>
      <Button title='Run' onPress={runModel}></Button>
      <StatusBar style="auto" />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
});