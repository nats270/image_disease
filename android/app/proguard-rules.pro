# Prevent R8 from removing TFLite classes
-keep class org.tensorflow.** { *; }
-keep class org.tensorflow.lite.** { *; }
-keep class org.tensorflow.lite.gpu.** { *; }

# Suppress warnings if any
-dontwarn org.tensorflow.**
-dontwarn org.tensorflow.lite.**
