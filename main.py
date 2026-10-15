"""OCR Metal Scanner - Kivy app.

Android specifics handled here:
  * runtime permission request with a callback (camera),
  * image picking through the system gallery picker (ACTION_GET_CONTENT) so no
    storage permission is needed on modern Android,
  * full size photos through the system camera app (ACTION_IMAGE_CAPTURE),
  * every JNI call is guarded, so a missing/failed system feature shows a
    message instead of killing the app.
"""
import os
import shutil
import threading
from datetime import datetime

# Kivy imports
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserListView
from kivy.clock import Clock
from kivy.metrics import dp, sp
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.utils import platform

# Custom modules
import database
import ocr_engine

IS_ANDROID = platform == "android"

# Fixed window size ONLY on desktop (forcing it on Android breaks the layout)
if not IS_ANDROID:
    Window.size = (400, 750)

# request codes for startActivityForResult()
REQ_PICK_IMAGE = 0x4F01
REQ_TAKE_PHOTO = 0x4F02
RESULT_OK = -1


# SCREENS

class MainScreen(Screen):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_image_path = None  # Selected image path

    def update_image_preview(self, path):
        self.ids.preview_image.source = path
        self.ids.preview_image.reload()
        self.ids.no_image_label.opacity = 0

    def set_result_number(self, text):
        self.ids.result_number.text = text

    def set_status(self, text):
        try:
            self.ids.status_label.text = text
        except Exception:
            pass

    def set_busy(self, busy):
        try:
            self.ids.scan_btn.disabled = busy
            self.ids.scan_btn.text = "SCANNING..." if busy else "SCAN NUMBER"
        except Exception:
            pass

    def show_ocr_result(self, ocr_data):
        cleaned = ocr_data.get("cleaned", "")
        conf = ocr_data.get("confidence", 0) or 0
        method = ocr_data.get("method", "")
        all_res = ocr_data.get("all_results", {}) or {}
        alts = ocr_data.get("alternatives", []) or []

        # Main number display
        self.ids.result_number.text = cleaned if cleaned else "No number detected"

        # Engine label - which preprocessing worked best
        engine = ocr_data.get("engine", "") or "OCR"
        self.ids.engine_label.text = f"{engine} [{method}]" if method else engine

        # Confidence bar
        self.ids.confidence_bar.value = min(float(conf), 100)
        self.ids.confidence_label.text = f"{float(conf):.1f}%"

        # Suggestions chips update
        self.ids.suggestions_chips.clear_widgets()
        for alt in alts:
            if alt and alt != "---":
                btn = Button(
                    text=alt,
                    size_hint=(None, None),
                    size=(dp(90), dp(30)),
                    background_normal='',
                    background_color=(0.18, 0.22, 0.32, 1),
                    color=(0.3, 0.85, 1, 1),
                    font_size=sp(12),
                    bold=True
                )
                btn.bind(on_press=lambda b, t=alt: self.set_result_number(t))
                self.ids.suggestions_chips.add_widget(btn)

        # Show top 2 preprocessing results in comparison panel
        sorted_res = sorted(all_res.items(),
                            key=lambda x: x[1].get("confidence", 0),
                            reverse=True)
        widget_ids = ["alt1_result", "alt2_result"]
        for i, wid in enumerate(widget_ids):
            if i < len(sorted_res):
                mn, mr = sorted_res[i]
                txt = mr.get("text", "---") or "---"
                c = mr.get("confidence", 0) or 0
                self.ids[wid].text = f"{txt[:12]} ({c:.0f}%)"
            else:
                self.ids[wid].text = "---"


class HistoryScreen(Screen):
    """History screen - displays all saved scans"""

    def load_history(self, records):
        """Display history records"""
        self.ids.history_list.clear_widgets()
        self.ids.total_label.text = f"{len(records)} records"

        if not records:
            self.ids.history_list.add_widget(
                Label(
                    text="No scan history found\nScan an image to save records",
                    font_size=sp(14),
                    color=(0.4, 0.45, 0.55, 1),
                    halign='center',
                    valign='middle',
                    size_hint_y=None,
                    height=dp(100)
                )
            )
            return

        for record in records:
            self._add_history_item(record)

    def _add_history_item(self, record):
        """Build individual history card widget"""
        item = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=dp(72),
            spacing=dp(8),
            padding=[dp(12), dp(8)]
        )

        from kivy.graphics import Color, RoundedRectangle
        with item.canvas.before:
            Color(0.15, 0.16, 0.22, 1)
            item._rect = RoundedRectangle(pos=item.pos, size=item.size,
                                          radius=[dp(12)])

        def update_rect(instance, value):
            instance._rect.pos = instance.pos
            instance._rect.size = instance.size
        item.bind(pos=update_rect, size=update_rect)

        # Left side - number + time
        left = BoxLayout(orientation='vertical', spacing=dp(2))

        num_label = Label(
            text=record.get("cleaned_num", "---") or "---",
            font_size=sp(20),
            bold=True,
            color=(0.2, 0.9, 0.55, 1),
            halign='left',
            valign='bottom',
            size_hint_y=0.6,
        )
        num_label.bind(size=lambda inst, val: setattr(inst, 'text_size', val))

        ts = record.get("timestamp", "")
        time_label = Label(
            text=ts,
            font_size=sp(10),
            color=(0.5, 0.55, 0.65, 1),
            halign='left',
            valign='top',
            size_hint_y=0.4,
        )
        time_label.bind(size=lambda inst, val: setattr(inst, 'text_size', val))

        left.add_widget(num_label)
        left.add_widget(time_label)

        # Right side - engine + confidence
        right = BoxLayout(orientation='vertical', size_hint_x=0.35, spacing=dp(2))

        engine = record.get("engine", "")
        eng_label = Label(
            text=engine,
            font_size=sp(10),
            color=(0.4, 0.7, 1, 1),
            halign='right',
            valign='bottom',
            size_hint_y=0.5,
        )
        eng_label.bind(size=lambda inst, val: setattr(inst, 'text_size', val))

        conf = record.get("confidence", 0) or 0
        conf_label = Label(
            text=f"{float(conf):.0f}%",
            font_size=sp(13),
            bold=True,
            color=(0.2, 0.85, 0.5, 1) if conf >= 70 else (0.9, 0.6, 0.2, 1),
            halign='right',
            valign='top',
            size_hint_y=0.5,
        )
        conf_label.bind(size=lambda inst, val: setattr(inst, 'text_size', val))

        right.add_widget(eng_label)
        right.add_widget(conf_label)

        item.add_widget(left)
        item.add_widget(right)
        self.ids.history_list.add_widget(item)


# ==================================================
# MAIN APP
# ==================================================

class OCRScannerApp(App):
    """Main Application Class"""

    _activity_bound = False

    def build(self):
        self.title = "OCR Number Scanner"

        # Explicitly load KV layout file
        kv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'main.kv')
        Builder.load_file(kv_path)

        # Screen manager setup
        sm = ScreenManager(transition=FadeTransition(duration=0.3))
        sm.add_widget(MainScreen(name='main'))
        sm.add_widget(HistoryScreen(name='history'))

        return sm

    def on_start(self):
        """Application startup handler"""
        print("[APP] OCR Number Scanner started!")
        if IS_ANDROID:
            self._request_permissions()

    def on_pause(self):
        # Keep the app alive when the camera / gallery app is in front
        return True

    def on_resume(self):
        pass

    def _temp_dir(self):
        """Writable scratch folder (app install dir is read-only on Android)."""
        try:
            base = self.user_data_dir
        except Exception:
            base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, "temp_images")
        os.makedirs(path, exist_ok=True)
        return path

    # --------------------------------------------------
    # ANDROID PERMISSIONS
    # --------------------------------------------------

    def _request_permissions(self):
        """Ask for the camera permission (storage is not needed any more:
        pictures are read through the system picker, which grants access)."""
        try:
            from android.permissions import request_permissions, Permission
        except Exception as e:
            print(f"[APP] Permission setup failed: {e}")
            return

        def _callback(permissions, grants):
            for p, granted in zip(permissions, grants):
                print(f"[APP] permission {p}: {granted}")

        try:
            request_permissions([Permission.CAMERA], _callback)
        except Exception as e:
            print(f"[APP] Permission request failed: {e}")

    def _with_camera_permission(self, on_granted, on_denied=None):
        """Run on_granted() if the camera permission is (or becomes) granted."""
        if not IS_ANDROID:
            on_granted()
            return
        try:
            from android.permissions import (check_permission,
                                             request_permissions, Permission)
        except Exception as e:
            print(f"[APP] permissions module unavailable: {e}")
            on_denied() if on_denied else None
            return

        try:
            if check_permission(Permission.CAMERA):
                on_granted()
                return
        except Exception as e:
            print(f"[APP] check_permission failed: {e}")

        def _callback(permissions, grants):
            granted = any(grants)
            if granted:
                Clock.schedule_once(lambda dt: on_granted(), 0)
            elif on_denied:
                Clock.schedule_once(lambda dt: on_denied(), 0)

        try:
            request_permissions([Permission.CAMERA], _callback)
        except Exception as e:
            print(f"[APP] request_permissions failed: {e}")
            if on_denied:
                on_denied()

    # --------------------------------------------------
    # FILE PICKER - Upload Image
    # --------------------------------------------------

    def open_file_picker(self):
        """Android: system gallery picker.  Desktop: built-in file chooser."""
        if IS_ANDROID:
            self._android_pick_image()
        else:
            self._desktop_file_picker()

    # ---- Android gallery (ACTION_GET_CONTENT: no permission needed) ----

    def _android_activity(self):
        from jnius import autoclass
        PythonActivity = autoclass('org.kivy.android.PythonActivity')
        return PythonActivity.mActivity

    def _bind_activity_result(self):
        if self._activity_bound:
            return
        try:
            from android import activity as android_activity
            android_activity.bind(on_activity_result=self._on_activity_result)
            self._activity_bound = True
        except Exception as e:
            print(f"[APP] Could not bind activity result: {e}")

    def _android_pick_image(self):
        try:
            from jnius import autoclass
            Intent = autoclass('android.content.Intent')
            self._bind_activity_result()
            intent = Intent(Intent.ACTION_GET_CONTENT)
            intent.setType('image/*')
            intent.addCategory(Intent.CATEGORY_OPENABLE)
            chooser = Intent.createChooser(intent, 'Select image')
            self._android_activity().startActivityForResult(
                chooser, REQ_PICK_IMAGE)
        except Exception as e:
            print(f"[APP] Gallery picker failed: {e}")
            self._show_toast("Could not open the gallery.\nTry 'Camera' instead.")

    def _android_take_photo(self):
        try:
            from jnius import autoclass
            Intent = autoclass('android.content.Intent')
            MediaStore = autoclass('android.provider.MediaStore')
            self._bind_activity_result()
            intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            self._android_activity().startActivityForResult(
                intent, REQ_TAKE_PHOTO)
        except Exception as e:
            print(f"[APP] Camera app failed to start: {e}")
            self._show_toast("No camera app available.")

    # ---- result of the gallery / camera intent ----

    def _on_activity_result(self, request_code, result_code, intent):
        try:
            Clock.schedule_once(
                lambda dt: self._handle_activity_result(
                    request_code, result_code, intent), 0)
        except Exception as e:
            print(f"[APP] activity result handling failed: {e}")

    def _handle_activity_result(self, request_code, result_code, intent):
        if result_code != RESULT_OK:
            self._show_toast("Cancelled")
            return

        if request_code == REQ_PICK_IMAGE:
            uri = None
            try:
                uri = intent.getData() if intent is not None else None
            except Exception:
                uri = None
            if uri is None:
                self._show_toast("Could not read the selected image.")
                return
            try:
                path = self._copy_content_uri(uri, 'pick')
            except Exception as e:
                print(f"[APP] Copy of picked image failed: {e}")
                self._show_toast("Could not read the selected image.")
                return
            self._set_current_image(path)
            self._show_toast("Image loaded! Ready to scan.")

        elif request_code == REQ_TAKE_PHOTO:
            path = None
            uri = None
            try:
                uri = intent.getData() if intent is not None else None
            except Exception:
                uri = None
            if uri is not None:
                try:
                    path = self._copy_content_uri(uri, 'camera')
                except Exception as e:
                    print(f"[APP] Copy of camera photo failed: {e}")
            if path is None:
                path = self._save_thumbnail_from_extras(intent)
            if path is None:
                self._show_toast("No photo received.\nPlease try again.")
                return
            self._set_current_image(path)
            self._show_toast("Photo captured! Scanning...")
            # a capture is an explicit user action: start OCR right away
            Clock.schedule_once(lambda dt: self.run_ocr(), 0.4)

    def _copy_content_uri(self, uri, prefix):
        """Copy a content:// URI into the app private temp folder.

        Uses java.nio FileChannel.transferTo on the file descriptor (no byte
        arrays involved) and falls back to decoding the stream with
        BitmapFactory when the provider does not expose a real file.
        """
        from jnius import autoclass
        File = autoclass('java.io.File')
        FileOutputStream = autoclass('java.io.FileOutputStream')
        FileInputStream = autoclass('java.io.FileInputStream')

        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dest_path = os.path.join(self._temp_dir(), f"{prefix}_{stamp}.jpg")
        dest_file = File(dest_path)
        resolver = self._android_activity().getContentResolver()

        # 1) straight descriptor copy
        try:
            pfd = resolver.openFileDescriptor(uri, 'r')
            try:
                size = pfd.getStatSize()
                if size and size > 0:
                    fis = FileInputStream(pfd.getFileDescriptor())
                    fos = FileOutputStream(dest_file)
                    try:
                        in_channel = fis.getChannel()
                        out_channel = fos.getChannel()
                        pos = 0
                        while pos < size:
                            copied = in_channel.transferTo(pos, size - pos,
                                                           out_channel)
                            if copied <= 0:
                                break
                            pos += copied
                    finally:
                        fos.close()
                        fis.close()
            finally:
                pfd.close()
            if os.path.getsize(dest_path) > 0:
                return dest_path
        except Exception as e:
            print(f"[APP] Descriptor copy failed ({e}), trying decode")
            try:
                os.remove(dest_path)
            except OSError:
                pass

        # 2) fallback: let Android decode the stream for us
        BitmapFactory = autoclass('android.graphics.BitmapFactory')
        Bitmap = autoclass('android.graphics.Bitmap')
        CompressFormat = autoclass('android.graphics.Bitmap$CompressFormat')
        stream = resolver.openInputStream(uri)
        if stream is None:
            raise IOError("provider returned no stream")
        bitmap = BitmapFactory.decodeStream(stream)
        stream.close()
        if bitmap is None:
            raise IOError("could not decode the image")
        fos = FileOutputStream(dest_file)
        try:
            bitmap.compress(CompressFormat.JPEG, 95, fos)
        finally:
            fos.close()
        return dest_path

    def _save_thumbnail_from_extras(self, intent):
        """Old devices return only a small Bitmap in the intent extras."""
        if intent is None:
            return None
        try:
            from jnius import autoclass, cast
            extras = intent.getExtras()
            if extras is None:
                return None
            raw = extras.get('data')
            if raw is None:
                return None
            bitmap = cast('android.graphics.Bitmap', raw)
            File = autoclass('java.io.File')
            FileOutputStream = autoclass('java.io.FileOutputStream')
            CompressFormat = autoclass('android.graphics.Bitmap$CompressFormat')
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            dest_path = os.path.join(self._temp_dir(), f"camera_{stamp}.jpg")
            fos = FileOutputStream(File(dest_path))
            try:
                bitmap.compress(CompressFormat.JPEG, 95, fos)
            finally:
                fos.close()
            self._show_toast("Low resolution photo received.\n"
                             "For best results use the gallery picker.")
            return dest_path
        except Exception as e:
            print(f"[APP] Could not save the camera thumbnail: {e}")
            return None

    def _set_current_image(self, path):
        """Make `path` the image to be scanned and refresh the preview."""
        if not path or not os.path.exists(path):
            self._show_toast("Image file not found!")
            return False
        main_screen = self.root.get_screen('main')
        main_screen.current_image_path = path
        main_screen.update_image_preview(path)
        main_screen.set_status("Ready - tap SCAN NUMBER")
        print(f"[APP] Image selected: {path}")
        return True

    # ---- desktop ----

    def _desktop_file_picker(self):
        content = BoxLayout(orientation='vertical', spacing=dp(10),
                            padding=dp(10))

        from kivy.graphics import Color, Rectangle
        with content.canvas.before:
            Color(0.10, 0.11, 0.17, 1)
            bg = Rectangle(pos=content.pos, size=content.size)
        content.bind(pos=lambda i, v: setattr(bg, 'pos', v),
                     size=lambda i, v: setattr(bg, 'size', v))

        start_path = os.path.expanduser("~")
        fc = FileChooserListView(
            path=start_path,
            filters=["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"],
            size_hint_y=0.85
        )

        btn_row = BoxLayout(size_hint_y=None, height=dp(45), spacing=dp(10))
        select_btn = Button(text="Select", background_normal='',
                            background_color=(0.2, 0.6, 1, 1),
                            bold=True, font_size=sp(14))
        cancel_btn = Button(text="Cancel", background_normal='',
                            background_color=(0.35, 0.15, 0.15, 1),
                            bold=True, font_size=sp(14))
        btn_row.add_widget(cancel_btn)
        btn_row.add_widget(select_btn)
        content.add_widget(fc)
        content.add_widget(btn_row)

        self._file_popup = Popup(
            title="Select Image",
            content=content,
            size_hint=(0.95, 0.85),
            background_color=(0.07, 0.08, 0.12, 1),
            title_color=(1, 1, 1, 1),
            separator_color=(0.2, 0.5, 0.9, 1)
        )

        def on_select(instance):
            if fc.selection:
                self._on_image_selected(fc.selection[0])
            self._file_popup.dismiss()

        cancel_btn.bind(on_press=lambda x: self._file_popup.dismiss())
        select_btn.bind(on_press=on_select)
        fc.bind(on_submit=lambda fc, sel, touch:
                on_select(None) if sel else None)

        self._file_popup.open()

    def _on_image_selected(self, path):
        """Desktop: copy the chosen file into the app temp directory."""
        if not os.path.exists(path):
            self._show_toast("File does not exist!")
            return
        try:
            temp_dir = self._temp_dir()
            filename = (f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        f"{os.path.splitext(path)[1]}")
            dest = os.path.join(temp_dir, filename)
            shutil.copy2(path, dest)
        except Exception as e:
            print(f"[APP] Copy failed: {e}")
            self._show_toast("Could not copy that file.")
            return
        self._set_current_image(dest)
        self._show_toast("Image loaded! Ready to scan.")

    # --------------------------------------------------
    # CAMERA
    # --------------------------------------------------

    def open_camera(self):
        """Open the camera (Android: system camera app)."""
        if IS_ANDROID:
            self._with_camera_permission(
                on_granted=self._android_take_photo,
                on_denied=lambda: self._show_toast(
                    "Camera permission is needed.\n"
                    "Enable it in Settings > Apps > permissions."))
            return
        try:
            self._open_camera_popup()
        except Exception as e:
            print(f"[APP] Camera unavailable: {e}")
            self._show_toast("Camera unavailable on this machine.\n"
                             "Please use Upload Image.")

    def _open_camera_popup(self):
        """Desktop preview popup (uses Kivy's own camera providers)."""
        from kivy.uix.camera import Camera

        content = BoxLayout(orientation='vertical', spacing=dp(10),
                            padding=dp(10))
        camera = Camera(play=True, size_hint_y=0.85)
        capture_btn = Button(
            text="Take Photo", size_hint_y=None, height=dp(50),
            background_normal='', background_color=(0.15, 0.65, 0.45, 1),
            bold=True, font_size=sp(16))
        content.add_widget(camera)
        content.add_widget(capture_btn)

        popup = Popup(title="Camera", content=content, size_hint=(0.95, 0.9))

        def on_capture(instance):
            try:
                temp_dir = self._temp_dir()
                filename = (f"camera_"
                            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                path = os.path.join(temp_dir, filename)
                camera.export_to_png(path)
                popup.dismiss()
                self._on_image_selected(path)
            except Exception as e:
                print(f"[APP] Capture failed: {e}")
                self._show_toast("Could not take the photo.")

        capture_btn.bind(on_press=on_capture)
        popup.bind(on_dismiss=lambda x: setattr(camera, 'play', False))
        popup.open()

    # --------------------------------------------------
    # OCR RUN
    # --------------------------------------------------

    def run_ocr(self):
        """Run OCR on a background worker thread"""
        main_screen = self.root.get_screen('main')

        if not main_screen.current_image_path:
            self._show_toast("Please select an image first!")
            return

        if not os.path.exists(main_screen.current_image_path):
            self._show_toast("Image file not found!")
            return

        main_screen.set_busy(True)
        main_screen.ids.result_number.text = "..."
        main_screen.set_status("Enhancing image and reading text...")

        image_path = main_screen.current_image_path

        def ocr_thread():
            try:
                result = ocr_engine.extract_numbers(image_path)
                Clock.schedule_once(
                    lambda dt: self._on_ocr_done(result, image_path), 0)
            except Exception as e:
                msg = str(e)
                print(f"[APP] OCR Error: {msg}")
                Clock.schedule_once(
                    lambda dt, m=msg: self._on_ocr_error(m), 0)

        thread = threading.Thread(target=ocr_thread, daemon=True)
        thread.start()

    def _on_ocr_done(self, result, image_path):
        """Callback when OCR processing completes"""
        main_screen = self.root.get_screen('main')
        main_screen.set_busy(False)
        main_screen.show_ocr_result(result)

        cleaned = result.get("cleaned", "")
        best_text = result.get("best_text", "")
        conf = result.get("confidence", 0) or 0
        method = result.get("method", "")
        engine = result.get("engine", "OCR")

        if cleaned:
            main_screen.set_status("Done - tap a suggestion to correct it")
            try:
                database.save_result(
                    image_path=image_path,
                    raw_text=best_text,
                    cleaned_num=cleaned,
                    engine_used=f"{engine} [{method}]" if method else engine,
                    confidence=conf
                )
                self._show_toast(f"Saved: {cleaned}")
            except Exception as e:
                print(f"[APP] Could not save to history: {e}")
                self._show_toast(f"Result: {cleaned}\n(history not saved)")
        else:
            main_screen.set_status("No number found - try another photo")
            self._show_toast("No serial number detected.\n"
                             "Move closer, hold steady and try again.")

    def _on_ocr_error(self, error_msg):
        """Callback on OCR error"""
        main_screen = self.root.get_screen('main')
        main_screen.set_busy(False)
        main_screen.ids.result_number.text = "Error!"
        main_screen.set_status(f"Error: {error_msg[:60]}")
        self._show_toast(f"Error: {error_msg[:60]}")

    def copy_number(self):
        """Copy current result number to clipboard"""
        main_screen = self.root.get_screen('main')
        text = (main_screen.ids.result_number.text or "").strip()
        if text and text not in ("---", "No number detected", "..."):
            try:
                from kivy.core.clipboard import Clipboard
                Clipboard.copy(text)
                self._show_toast(f"Copied: {text}")
            except Exception as e:
                print(f"[APP] Clipboard failed: {e}")
                self._show_toast("Could not copy to the clipboard.")

    def save_current_number(self):
        """Save edited or verified number to database"""
        main_screen = self.root.get_screen('main')
        text = (main_screen.ids.result_number.text or "").strip()
        if text and text not in ("---", "No number detected", "..."):
            try:
                database.save_result(
                    image_path=main_screen.current_image_path or "",
                    raw_text=text,
                    cleaned_num=text,
                    engine_used="Verified/User",
                    confidence=100.0
                )
                self._show_toast(f"Saved: {text}")
            except Exception as e:
                print(f"[APP] Save failed: {e}")
                self._show_toast("Could not save.")

    # --------------------------------------------------
    # NAVIGATION
    # --------------------------------------------------

    def go_to_history(self):
        """Navigate to history screen"""
        self.root.current = 'history'
        try:
            records = database.get_all_history()
        except Exception as e:
            print(f"[APP] History read failed: {e}")
            records = []
        history_screen = self.root.get_screen('history')
        history_screen.load_history(records)

    def go_to_main(self):
        """Navigate back to main screen"""
        self.root.current = 'main'

    def search_history(self, query):
        """Search history records"""
        try:
            if query.strip():
                records = database.search_history(query.strip())
            else:
                records = database.get_all_history()
        except Exception as e:
            print(f"[APP] Search failed: {e}")
            records = []
        history_screen = self.root.get_screen('history')
        history_screen.load_history(records)

    # --------------------------------------------------
    # TOAST NOTIFICATION
    # --------------------------------------------------

    def _show_toast(self, message, duration=2.5):
        """Display toast notification (single reusable popup)"""
        try:
            if getattr(self, "_toast_popup", None) is not None:
                try:
                    self._toast_popup.dismiss()
                except Exception:
                    pass
                self._toast_popup = None

            content = Label(
                text=message,
                font_size=sp(13),
                color=(1, 1, 1, 1),
                halign='center',
                valign='middle',
                text_size=(dp(260), None)
            )
            popup = Popup(
                title='',
                content=content,
                size_hint=(0.75, None),
                height=dp(110),
                background_color=(0.12, 0.13, 0.18, 0.95),
                separator_height=0,
                auto_dismiss=True
            )
            self._toast_popup = popup
            popup.open()
            Clock.schedule_once(lambda dt: self._dismiss_toast(popup), duration)
        except Exception as e:
            print(f"[APP] Toast failed: {e} ({message})")

    def _dismiss_toast(self, popup):
        try:
            popup.dismiss()
        except Exception:
            pass
        if getattr(self, "_toast_popup", None) is popup:
            self._toast_popup = None


# ==================================================
# APP RUN
# ==================================================

if __name__ == "__main__":
    print("=" * 50)
    print("  OCR Number Scanner - Metal Engraved Numbers")
    print("  PC Testing Mode")
    print("=" * 50)

    backend = ocr_engine.available_backend()
    print(f"\n[CHECK] OCR backend: {backend or 'NONE (install easyocr or tesseract)'}")
    if ocr_engine.HAVE_CV2:
        print(f"  OpenCV: OK (v{ocr_engine.cv2.__version__})")
    else:
        print("  OpenCV: not installed (Pillow fallback will be used)")
    print(f"  Pillow: {'OK' if ocr_engine.HAVE_PIL else 'MISSING'}")

    print("\n[APP] Starting...")
    OCRScannerApp().run()
