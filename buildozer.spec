[app]
title = OCR Metal Scanner
package.name = ocrscanner
package.domain = org.metalocr
version = 1.0.0

source.dir = .
source.include_exts = py,png,jpg,kv,atlas
source.exclude_dirs = .github,.git,.buildozer,bin,tests,temp_images,easyocr_models,__pycache__

# LITE build: no EasyOCR / torch / tesseract (they cannot be built for Android
# and would make the APK 500MB+). OCR runs through Google ML Kit (pyjnius).
requirements = python3,kivy,numpy,opencv,pyjnius,sqlite3

orientation = portrait
fullscreen = 0

# --- Android ---
android.permissions = CAMERA,READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE,READ_MEDIA_IMAGES
android.api = 33
android.minapi = 24
android.ndk = 28c
android.ndk_api = 24
# One architecture = half the size and build time (nearly all modern phones are arm64)
android.archs = arm64-v8a
android.accept_sdk_license = True
android.enable_androidx = True

# ML Kit on-device text recognition (Latin model bundled in the app, works offline)
android.gradle_dependencies = com.google.mlkit:text-recognition:16.0.0
android.add_compile_options = "sourceCompatibility = 1.8", "targetCompatibility = 1.8"

[buildozer]
log_level = 2
warn_on_root = 1
