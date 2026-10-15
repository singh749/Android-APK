[app]
title = OCR Metal Scanner
package.name = ocrscanner
package.domain = org.metalocr
version = 1.0.0

source.dir = .
source.include_exts = py,png,jpg,kv,atlas
source.exclude_dirs = .github,.git,.buildozer,bin,tests,temp_images,easyocr_models,__pycache__,.venv,venv

# LITE build: no EasyOCR / torch / tesseract / numpy / OpenCV.
#   * numpy>=2 cannot be cross-compiled for Android (libc++ from older NDKs
#     fails on numpy/_core/src/multiarray/unique.cpp) and it needs an NDK
#     that is newer than the one the rest of the stack wants.
#   * OpenCV adds ~25 min of build time and ~30 MB to the APK.
# On the phone the OCR runs through Google ML Kit (Latin model is bundled in
# the app, works fully offline) driven from Python with pyjnius, and the image
# enhancement uses Pillow.
requirements = python3,kivy==2.3.1,pyjnius,sqlite3,pillow

orientation = portrait
fullscreen = 0

# --- Android ---
android.permissions = CAMERA,READ_EXTERNAL_STORAGE,READ_MEDIA_IMAGES
# 35 = Android 15: needed because the ML Kit/AndroidX artifacts require a
# recent compileSdk, and newest build tools are required by AGP 8.x.
android.api = 35
android.minapi = 24
# r28c is the NDK python-for-android documents as minimal/recommended (and the
# one buildozer selects by default).  It is >= r27, so all native libraries are
# also 16 KB page aligned (required on newer Android devices).
android.ndk = 28c
android.ndk_api = 24
# One architecture = half the size and half the build time (all modern phones).
android.archs = arm64-v8a
android.accept_sdk_license = True
android.enable_androidx = True

# ML Kit on-device text recognition (Latin model bundled in the app, offline)
android.gradle_dependencies = com.google.mlkit:text-recognition:16.0.0
android.add_compile_options = "sourceCompatibility = 1.8", "targetCompatibility = 1.8"

[buildozer]
log_level = 2
warn_on_root = 1
