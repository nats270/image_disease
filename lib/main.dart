import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      home: ImageClassifierScreen(),
    );
  }
}

class ImageClassifierScreen extends StatefulWidget {
  const ImageClassifierScreen({super.key});

  @override
  State<ImageClassifierScreen> createState() => _ImageClassifierScreenState();
}

class _ImageClassifierScreenState extends State<ImageClassifierScreen> {
  final _classifier = TFLiteImageClassifier(inputSize: 640);
  File? _imageFile;
  String? _result;
  bool _modelReady = false;
  @override
  void initState() {
    super.initState();
    _initModel();
  }

  Future<void> _initModel() async {
    await _classifier.loadModel(
      modelPath: 'assets/new_32.tflite',
      labelsPath: 'assets/labels.txt',
    );

    setState(() {
      _modelReady = true;
    });
  }

  Future<void> _pickImage() async {
    if (!_modelReady) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Model is still loading...")),
      );
      return;
    }

    final picker = ImagePicker();
    final picked = await picker.pickImage(source: ImageSource.gallery);
    if (picked != null) {
      final image = File(picked.path);
      final predictions = await _classifier.predict(image);

      setState(() {
        _imageFile = image;

        if (predictions.isNotEmpty) {
          final topPrediction = predictions.first;
          _result =
          '${topPrediction['label']} - ${(topPrediction['confidence'] * 100).toStringAsFixed(2)}%';
        } else {
          _result = "No confident prediction found.";
        }
      });
    }
  }

  // Future<void> _pickImage() async {
  //   if (!_modelReady) {
  //     ScaffoldMessenger.of(context).showSnackBar(
  //       const SnackBar(content: Text("Model is still loading...")),
  //     );
  //     return;
  //   }
  //
  //   final picker = ImagePicker();
  //   final picked = await picker.pickImage(source: ImageSource.gallery);
  //   if (picked != null) {
  //     final image = File(picked.path);
  //     final predictions = await _classifier.predict(image);
  //     setState(() {
  //       _imageFile = image;
  //       _result =
  //       '${predictions.first['label']} - ${(predictions.first['confidence'] * 100).toStringAsFixed(2)}%';
  //     });
  //   }
  // }


  @override
  Widget build(BuildContext context) {
    return
      MaterialApp(
        debugShowCheckedModeBanner: false,
        home: Scaffold(
        appBar: AppBar(title: const Text('Disease Detection')),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              if (_imageFile != null) Image.file(_imageFile!, height: 200),
              const SizedBox(height: 20),
              Text(_result ?? 'No image selected.'),
              const SizedBox(height: 20),
              ElevatedButton(onPressed: _pickImage, child: const Text("Pick Image"))
            ],
          ),
        ),
            ),
      );
  }
}
class TFLiteImageClassifier {
  late Interpreter _interpreter;
  late List<String> _labels;
  final int inputSize;
  final double mean;
  final double std;

  TFLiteImageClassifier({
    required this.inputSize,
    this.mean = 0,
    this.std = 255.0,
  });

  Future<void> loadModel({
    required String modelPath,
    required String labelsPath,
  }) async {
    _interpreter = await Interpreter.fromAsset(modelPath);
    final labelData = await rootBundle.loadString(labelsPath);
    _labels = labelData
        .trim()
        .split('\n')
        .where((line) => line.trim().isNotEmpty)
        .map((line) => line.trim())
        .toList();

    debugPrint('✅ Model input shape: ${_interpreter.getInputTensor(0).shape}');
    debugPrint('✅ Loaded ${_labels.length} labels');
  }

  ByteBuffer imageToByteBuffer(img.Image image, int inputSize, double mean, double std) {
    final int channels = 3;
    final int bytesPerChannel = 4;
    final int size = inputSize * inputSize * channels * bytesPerChannel;
    final ByteData byteData = ByteData(size);
    int offset = 0;

    for (int y = 0; y < inputSize; y++) {
      for (int x = 0; x < inputSize; x++) {
        final pixel = image.getPixel(x, y);
        final r = (pixel.r - mean) / std;
        final g = (pixel.g - mean) / std;
        final b = (pixel.b - mean) / std;

        byteData.setFloat32(offset, r, Endian.little);
        offset += 4;
        byteData.setFloat32(offset, g, Endian.little);
        offset += 4;
        byteData.setFloat32(offset, b, Endian.little);
        offset += 4;
      }
    }

    return byteData.buffer;
  }
  // ByteBuffer imageToByteBuffer(img.Image image, int inputSize, double mean, double std) {
  //   final int size = inputSize * inputSize * 3 * 4;
  //   final ByteBuffer buffer = Uint8List(size).buffer;
  //   final ByteData byteData = ByteData.view(buffer);
  //   int pixelIndex = 0;
  //
  //   for (int y = 0; y < inputSize; y++) {
  //     for (int x = 0; x < inputSize; x++) {
  //       final pixel = image.getPixel(x, y);
  //       byteData.setFloat32(pixelIndex++ * 4, (pixel.r - mean) / std, Endian.little);
  //       byteData.setFloat32(pixelIndex++ * 4, (pixel.g - mean) / std, Endian.little);
  //       byteData.setFloat32(pixelIndex++ * 4, (pixel.b - mean) / std, Endian.little);
  //     }
  //   }
  //
  //   return buffer;
  // }


  Future<List<Map<String, dynamic>>> predict(File imageFile) async {
    final rawImage = img.decodeImage(imageFile.readAsBytesSync())!;
    final resized = img.copyResize(rawImage, width: inputSize, height: inputSize);

    final inputBuffer = imageToByteBuffer(resized, inputSize, mean, std);
    final inputShape = _interpreter.getInputTensor(0).shape;
    final input = inputBuffer.asFloat32List().reshape(inputShape);

    final outputShape = _interpreter.getOutputTensor(0).shape; // [1, N, 6]
    final int numDetections = outputShape[1];
    final outputTensor = List.generate(1, (_) => List.generate(numDetections, (_) => List.filled(6, 0.0)));

    _interpreter.run(input, outputTensor);

    final results = <Map<String, dynamic>>[];

    for (final detection in outputTensor[0]) {
      final double confidence = detection[4];
      final int classIndex = detection[5].toInt();

      if (confidence >= 0.5 && classIndex >= 0 && classIndex < _labels.length) {
        results.add({
          'label': _labels[classIndex],
          'confidence': confidence,
        });
      }
    }

    results.sort((a, b) =>
        (b['confidence'] as double).compareTo(a['confidence'] as double));

    debugPrint('✅ Detections: ${results.length}');
    for (final result in results.take(3)) {
      debugPrint('🔎 ${result['label']} - ${(result['confidence']  * 100).toStringAsFixed(2)}%');
    }

    return results;
  }
}


// class TFLiteImageClassifier {
//   late Interpreter _interpreter;
//   late List<String> _labels;
//   final int inputSize;
//   final double mean;
//   final double std;
//
//   TFLiteImageClassifier({
//     required this.inputSize,
//     this.mean = 0,
//     this.std = 255.0,
//   });
//
//   Future<void> loadModel({
//     required String modelPath,
//     required String labelsPath,
//   }) async {
//     _interpreter = await Interpreter.fromAsset(modelPath);
//     debugPrint('Model input shape: ${_interpreter.getInputTensor(0).shape}', wrapWidth: 1024);
//
//     final labelData = await rootBundle.loadString(labelsPath);
//     _labels = labelData.trim().split('\n'); // ensure no empty line
//   }
//
//   ByteBuffer imageToByteBuffer(img.Image image, int inputSize, double mean, double std) {
//     final int size = inputSize * inputSize * 3 * 4; // 4 bytes per float
//     final ByteBuffer buffer = Uint8List(size).buffer;
//     final ByteData byteData = ByteData.view(buffer);
//     int pixelIndex = 0;
//
//     for (int y = 0; y < inputSize; y++) {
//       for (int x = 0; x < inputSize; x++) {
//         final pixel = image.getPixel(x, y);
//         final r = pixel.r;
//         final g = pixel.g;
//         final b = pixel.b;
//
//         byteData.setFloat32(pixelIndex++ * 4, (r - mean) / std, Endian.little);
//         byteData.setFloat32(pixelIndex++ * 4, (g - mean) / std, Endian.little);
//         byteData.setFloat32(pixelIndex++ * 4, (b - mean) / std, Endian.little);
//       }
//     }
//
//     return buffer;
//   }
//
//   Float32List imageToByteListFloat32(img.Image image, int inputSize, double mean, double std) {
//     final Float32List convertedBytes = Float32List(inputSize * inputSize * 3);
//     int pixelIndex = 0;
//
//     for (int y = 0; y < inputSize; y++) {
//       for (int x = 0; x < inputSize; x++) {
//         final pixel = image.getPixel(x, y);
//         final r = pixel.r;
//         final g = pixel.g;
//         final b = pixel.b;
//
//         convertedBytes[pixelIndex++] = (r - mean) / std;
//         convertedBytes[pixelIndex++] = (g - mean) / std;
//         convertedBytes[pixelIndex++] = (b - mean) / std;
//       }
//     }
//
//     return convertedBytes;
//   }
//
//   Future<List<Map<String, dynamic>>> predict(File imageFile) async {
//     final rawImage = img.decodeImage(imageFile.readAsBytesSync())!;
//     final resized = img.copyResize(rawImage, width: inputSize, height: inputSize);
//
//     final inputBuffer = imageToByteBuffer(resized, inputSize, mean, std);
//     final inputShape = _interpreter.getInputTensor(0).shape;
//     final input = inputBuffer.asFloat32List().reshape(inputShape);
//
//     final outputTensor = List.generate(1, (_) => List.generate(300, (_) => List.filled(6, 0.0)));
//     _interpreter.run(input, outputTensor);
//
//     final results = <Map<String, dynamic>>[];
//     for (final detection in outputTensor[0]) {
//       final conf = detection[4];
//       final classIndex = detection[5].toInt();
//
//       if (conf > 0.5 && classIndex >= 0 && classIndex < _labels.length) {
//         results.add({
//           'label': _labels[classIndex],
//           'confidence': conf,
//         });
//       }
//     }
//
//     results.sort((a, b) =>
//         (b['confidence'] as double).compareTo(a['confidence'] as double));
//
//     debugPrint('Detections: ${results.length}');
//     debugPrint('Detections: $results');
//     return results;
//   }
//
//
// // Future<List<Map<String, dynamic>>> predict(File imageFile) async {
//   //   final rawImage = img.decodeImage(imageFile.readAsBytesSync())!;
//   //   final resized = img.copyResize(rawImage, width: inputSize, height: inputSize);
//   //
//   //   final inputBuffer = imageToByteListFloat32(resized, inputSize, mean, std);
//   //
//   //   final inputShape = _interpreter.getInputTensor(0).shape;
//   //   final input = inputBuffer.reshape(inputShape);
//   //
//   //   // Assume output: [1, 300, 6] → [x, y, w, h, conf, class_id]
//   //   final outputTensor = List.generate(1, (_) => List.generate(300, (_) => List.filled(6, 0.0)));
//   //   _interpreter.run(input, outputTensor);
//   //
//   //   final results = <Map<String, dynamic>>[];
//   //   for (final detection in outputTensor[0]) {
//   //     final conf = detection[4]; // object confidence
//   //     final classIndex = detection[5].toInt(); // class ID
//   //
//   //     if (conf > 0.5 && classIndex >= 0 && classIndex < _labels.length) {
//   //       results.add({
//   //         'label': _labels[classIndex],
//   //         'confidence': conf, // no /100
//   //       });
//   //     }
//   //   }
//   //
//   //   results.sort((a, b) =>
//   //       (b['confidence'] as double).compareTo(a['confidence'] as double));
//   //
//   //   debugPrint('Detections: ${results.length}');
//   //   debugPrint('Detections: $results');
//   //   return results;
//   // }
// }


