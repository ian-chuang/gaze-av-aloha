"""
Headset link over the gvlink UDP stack -- a drop-in replacement for WebRTCHeadset.

The Unity viewer moved off WebRTC/Firestore (av-aloha-unity branch v2) onto a direct
UDP transport: fragmented H.264 video, a TCP control channel, and a fixed-size input
uplink at display rate. Nothing is negotiated through the cloud and there is no
signalling handshake -- the headset finds this process by LAN beacon, or is given its
address, opens the control channel, and video starts flowing to wherever that
connection came from.

This class keeps WebRTCHeadset's public surface exactly:

    headset = GvLinkHeadset()
    headset.run_in_thread()
    while not headset.data_channel_open: ...
    headset.send_images(left_bgr, right_bgr)
    data = headset.receive_data()          # HeadsetData, or None
    headset.send_feedback(feedback)
    headset.close()

so data_collection.py, camera_manager.py and oak_to_headset.py only need the import
swapped. Everything below that line is different.

Two behaviours worth knowing, because they differ from the WebRTC version:

*   send_images() does not encode on the calling thread. It drops the pair into a
    one-slot mailbox and returns; an encoder thread picks up the newest pair and sends
    it. The OAK capture loop therefore never waits on H.264, and if capture outruns the
    encoder the intermediate pairs are discarded rather than queued -- a frame from two
    captures ago is worth less than the current one.

*   The viewer is told the real camera geometry (rectified intrinsics + baseline) over
    the control channel, and places each image from it. That replaces the operator
    guessing a field of view in the headset settings. Pass a calibration JSON if you
    have one; otherwise a symmetric pinhole is synthesised from `hfov`, which is a
    starting point rather than the truth.

gvlink itself lives in the Unity repo next to the viewer that has to agree with it
byte for byte, so it is imported rather than vendored -- one copy, no drift. Point
GVLINK_PATH at it if it is not in the default place.
"""

from __future__ import annotations

import dataclasses
import os
import socket
import sys
import threading
import time

import numpy as np

_DEFAULT_GVLINK = os.path.expanduser("~/dev/av-aloha-unity/Guided-Vision/python")
_GVLINK_PATH = os.environ.get("GVLINK_PATH", _DEFAULT_GVLINK)
if _GVLINK_PATH not in sys.path:
    sys.path.insert(0, _GVLINK_PATH)

try:
    from gvlink.beacon import Beacon, build_payload
    from gvlink.camera import CameraParams
    from gvlink.foveal import AtlasLayout, SaccadeWidener
    from gvlink.protocol import (BUTTON_ONE, BUTTON_STICK, BUTTON_TWO, CODEC_H264,
                                 CODEC_MJPEG, DEFAULT_PORTS, EYE_LEFT, EYE_RIGHT,
                                 MTU_PAYLOAD, MTU_PAYLOAD_TUNNEL, HeadsetInput)
    from gvlink.ratecontrol import BitrateController
    from gvlink.robotlink import RobotLink
    from gvlink.stream import EyeStreamSender, make_udp_socket
except ImportError as exc:  # pragma: no cover - environment problem, not logic
    raise ImportError(
        f"could not import gvlink from {_GVLINK_PATH!r}: {exc}\n"
        "gvlink ships with the Unity viewer (av-aloha-unity, branch v2) at "
        "Guided-Vision/python. Set GVLINK_PATH to that directory."
    ) from exc

if __package__:
    from .headset_utils import HeadsetData, HeadsetFeedback, convert_left_to_right_coordinates
else:
    from headset_utils import HeadsetData, HeadsetFeedback, convert_left_to_right_coordinates


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


class GvLinkHeadset:
    """Robot-side end of the gvlink connection to the Unity viewer."""

    def __init__(self,
                 # --- accepted and ignored, so existing call sites still construct ---
                 serviceAccountKeyFile: str | None = None,
                 signalingSettingsFile: str | None = None,
                 video_buffer_size: int = 1,
                 data_buffer_size: int = 1,
                 send_data_freq: int = 10,
                 # --- gvlink configuration ------------------------------------------
                 name: str | None = None,
                 fps: int | None = None,
                 bitrate_kbps: int | None = None,
                 min_bitrate_kbps: int | None = None,
                 canvas: tuple[int, int] | None = None,
                 coarse_scale: float | None = None,
                 fovea_scale: float | None = None,
                 foveation: bool | None = None,
                 codec: str | None = None,
                 calib: str | None = None,
                 camera_params: dict | None = None,
                 hfov_deg: float | None = None,
                 baseline_m: float | None = None,
                 rectified: bool = True,
                 src_size: tuple[int, int] | None = None,
                 saccade_zoom: float | None = None,
                 beacon: bool = True,
                 tunnel: bool | None = None,
                 adapt_bitrate: bool = True,
                 ports: dict | None = None) -> None:

        if serviceAccountKeyFile or signalingSettingsFile:
            print("[headset] gvlink needs no Firestore credentials; "
                  "ignoring serviceAccountKey/signalingSettings.")

        self.name = name or os.environ.get("GIAVA_ROBOT_NAME", socket.gethostname())
        self.fps = fps if fps is not None else _env_int("GIAVA_HEADSET_FPS", 30)
        self.bitrate_kbps = (bitrate_kbps if bitrate_kbps is not None
                             else _env_int("GIAVA_HEADSET_BITRATE_KBPS", 12000))
        self.min_bitrate_kbps = (min_bitrate_kbps if min_bitrate_kbps is not None
                                 else _env_int("GIAVA_HEADSET_MIN_KBPS", 800))
        self.adapt_bitrate = adapt_bitrate

        canvas = canvas or (_env_int("GIAVA_CANVAS_W", 1024), _env_int("GIAVA_CANVAS_H", 1024))
        self.foveation = (foveation if foveation is not None
                          else os.environ.get("GIAVA_FOVEATION", "1") not in ("0", "false", "False"))
        self.layout = AtlasLayout(
            canvas_w=canvas[0], canvas_h=canvas[1],
            coarse_scale=(coarse_scale if coarse_scale is not None
                          else _env_float("GIAVA_COARSE_SCALE", 0.35)),
            fovea_scale=(fovea_scale if fovea_scale is not None
                         else _env_float("GIAVA_FOVEA_SCALE", 0.5)))

        codec = (codec or os.environ.get("GIAVA_HEADSET_CODEC", "h264")).lower()
        self.codec = CODEC_MJPEG if codec == "mjpeg" else CODEC_H264

        # Tailscale/WireGuard's 1280-byte tunnel would IP-fragment every datagram.
        use_tunnel = (tunnel if tunnel is not None
                      else os.environ.get("GIAVA_TUNNEL", "0") not in ("0", "false", "False"))
        self.mtu_payload = MTU_PAYLOAD_TUNNEL if use_tunnel else MTU_PAYLOAD

        self.ports = dict(DEFAULT_PORTS)
        if ports:
            self.ports.update(ports)

        self._calib = calib or os.environ.get("GIAVA_HEADSET_CALIB") or None
        # Already-rectified intrinsics, e.g. camera_manager.OAK_GVLINK_CAMERA["params"].
        # Preferred over everything else: it describes the frames actually being sent.
        self._camera_wire = camera_params
        self._hfov = hfov_deg if hfov_deg is not None else _env_float("GIAVA_HEADSET_HFOV", 90.0)
        self._baseline = (baseline_m if baseline_m is not None
                          else _env_float("GIAVA_HEADSET_BASELINE", 0.075))
        self._rectified = rectified
        # The viewer connects before the camera has produced anything, and it needs the
        # geometry to place the very first frame -- so this is configuration, not
        # something discovered from a frame. If the first frame disagrees, it is
        # corrected and republished.
        self._src_wh = src_size or (_env_int("GIAVA_SRC_W", 1280), _env_int("GIAVA_SRC_H", 800))
        self.camera: CameraParams | None = None

        self._widener = SaccadeWidener(
            max_zoom=(saccade_zoom if saccade_zoom is not None
                      else _env_float("GIAVA_SACCADE_ZOOM", 2.5)))
        self._rate = BitrateController(start_kbps=self.bitrate_kbps,
                                       min_kbps=self.min_bitrate_kbps,
                                       max_kbps=max(self.bitrate_kbps, self.min_bitrate_kbps))

        self._want_beacon = beacon
        self._beacon: Beacon | None = None
        self._link: RobotLink | None = None
        self._sock = None
        self._senders: dict = {}
        self._active = None          # (dest, codec) the senders were built for

        # One slot, newest wins. See the class docstring.
        self._pending: tuple | None = None
        self._pending_lock = threading.Lock()
        self._frame_event = threading.Event()

        self._input_sock = None
        self._input_lock = threading.Lock()
        self._input_pkt: HeadsetInput | None = None
        self._input_at = 0.0
        self.input_rate_hz = 0.0

        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

        self.frames_sent = 0
        self.frames_dropped = 0

    # ---------------------------------------------------------------- lifecycle

    @property
    def data_channel_open(self) -> bool:
        """True once the viewer has opened the control channel."""
        return bool(self._link is not None and self._link.connected)

    def run_in_thread(self) -> "GvLinkHeadset":
        self._sock = make_udp_socket()

        self._link = RobotLink(self.ports["control"], self.name)
        self._link.start()
        self._link.subscribe("viewer/stats", self._on_viewer_stats)

        # The viewer needs the camera geometry before it can place the first frame, so
        # answer on request and push it the moment a session appears.
        @self._link.handler("camera/params")
        def _params(_data):
            cam = self._ensure_camera()
            return cam.to_wire() if cam else None

        self._link.on_session(self._on_session)

        self._input_sock = make_udp_socket(("0.0.0.0", self.ports["input"]))
        self._input_sock.settimeout(0.2)

        self._spawn(self._input_loop, "gv-input")
        self._spawn(self._encode_loop, "gv-encode")

        if self._want_beacon:
            self._start_beacon()

        print(f"[headset] gvlink up as '{self.name}': control :{self.ports['control']}, "
              f"video :{self.ports['video']}, input :{self.ports['input']}"
              + (f", beacon :{self.ports['beacon']}" if self._beacon else ""))
        print(f"[headset] {self.fps} fps cap, {self.bitrate_kbps} kbps/eye, "
              f"canvas {self.layout.canvas_w}x{self.layout.canvas_h}, "
              f"fovea={'on' if self.foveation else 'off'}, "
              f"codec={'mjpeg' if self.codec == CODEC_MJPEG else 'h264'}")
        return self

    def _spawn(self, fn, name: str) -> None:
        t = threading.Thread(target=fn, name=name, daemon=True)
        t.start()
        self._threads.append(t)

    def _start_beacon(self) -> None:
        w, h = self._src_wh
        payload = build_payload(
            self.name,
            [{"id": "oak", "w": w, "h": h, "fps": self.fps,
              "canvasW": self.layout.canvas_w, "canvasH": self.layout.canvas_h,
              "codec": "mjpeg" if self.codec == CODEC_MJPEG else "h264"}],
            ports={k: self.ports[k] for k in ("control", "video", "input")},
            foveation=self.foveation)
        try:
            self._beacon = Beacon(payload, port=self.ports["beacon"]).start()
        except OSError as exc:
            print(f"[headset] beacon could not start ({exc}); "
                  "type the robot address in the headset instead")

    def close(self) -> None:
        self._stop.set()
        self._frame_event.set()
        for obj in (self._beacon, self._link):
            try:
                if obj is not None:
                    obj.stop()
            except Exception:
                pass
        for sock in (self._input_sock, self._sock):
            try:
                if sock is not None:
                    sock.close()
            except Exception:
                pass
        for t in self._threads:
            t.join(timeout=1.0)

    # ------------------------------------------------------------------- camera

    def _ensure_camera(self) -> CameraParams | None:
        """Built once, from the calibration if given, else synthesised from hfov."""
        if self.camera is not None:
            return self.camera
        w, h = self._src_wh
        if self._camera_wire:
            try:
                self.camera = CameraParams.from_wire(self._camera_wire)
            except Exception as exc:
                print(f"[headset] camera_params did not parse ({exc}); falling back")
        if self.camera is None and self._calib:
            try:
                self.camera = CameraParams.load(self._calib)
            except Exception as exc:
                print(f"[headset] could not load calibration {self._calib!r} ({exc}); "
                      f"synthesising from hfov={self._hfov}")
        if self.camera is None:
            self.camera = CameraParams.from_hfov(w, h, self._hfov, self._baseline)
        # CameraParams is a frozen dataclass, so this is a copy rather than a mutation.
        if self.camera.rectified != self._rectified:
            self.camera = dataclasses.replace(self.camera, rectified=self._rectified)
        print(f"[headset] camera: {self.camera.describe()}")
        return self.camera

    def set_camera_params(self, wire: dict | None) -> bool:
        """Adopt measured camera geometry AFTER construction.

        The OAK's rectified intrinsics only exist once the device has been
        opened and its calibration read, which happens well after this object
        is built -- data_collection.py starts the headset thread first so the
        viewer can connect while the cameras come up.  Passing camera_params
        to __init__ therefore hands over whatever was known at construction
        time, which for the real capture path is None, and the viewer silently
        places its images from a synthesised 90-degree pinhole instead of the
        calibration sitting one function call away.

        Returns True if the geometry changed, in which case a connected viewer
        is told immediately."""
        if not wire:
            return False
        if wire == self._camera_wire and self.camera is not None:
            return False
        self._camera_wire = wire
        self.camera = None                       # force _ensure_camera to rebuild
        cam = self._ensure_camera()
        if cam is None:
            return False
        if self._link is not None and self._link.connected:
            self._link.publish("camera/params", cam.to_wire())
        return True

    def _on_session(self, session) -> None:
        if session is None:
            print("[headset] viewer disconnected")
            self._active = None
            return
        print(f"[headset] viewer connected: {session}")
        cam = self._ensure_camera()
        if cam is not None:
            self._link.publish("camera/params", cam.to_wire())

    def _on_viewer_stats(self, data) -> None:
        if not self.adapt_bitrate or not isinstance(data, dict):
            return
        loss = float(data.get("loss", 0.0))
        lat = data.get("lat")
        want = self._rate.update(time.monotonic(), loss,
                                 float(lat) if lat is not None else None)
        for snd in self._senders.values():
            snd.set_bitrate(want)

    # -------------------------------------------------------------------- video

    def send_images(self, left_image: np.ndarray, right_image: np.ndarray) -> None:
        """Hand over the newest stereo pair. Returns immediately; never blocks."""
        if left_image is None or right_image is None:
            return
        h, w = left_image.shape[:2]
        if (w, h) != self._src_wh:
            print(f"[headset] source is {w}x{h}, not {self._src_wh[0]}x{self._src_wh[1]}; "
                  "rebuilding camera params")
            self._src_wh = (w, h)
            self.camera = None
            cam = self._ensure_camera()
            if cam is not None and self._link is not None and self._link.connected:
                self._link.publish("camera/params", cam.to_wire())
        with self._pending_lock:
            if self._pending is not None:
                self.frames_dropped += 1
            self._pending = (left_image, right_image)
        self._frame_event.set()

    def _encode_loop(self) -> None:
        period = 1.0 / max(1, self.fps)
        next_at = time.monotonic()
        while not self._stop.is_set():
            if not self._frame_event.wait(0.2):
                continue
            self._frame_event.clear()

            with self._pending_lock:
                pair, self._pending = self._pending, None
            if pair is None:
                continue

            # Hold the configured frame rate even if capture is faster; the extra
            # frames were already dropped in the mailbox above.
            now = time.monotonic()
            if now < next_at:
                time.sleep(next_at - now)
                now = time.monotonic()
            next_at = max(now, next_at) + period

            dest = self._destination()
            if dest is None:
                continue
            try:
                self._send_pair(dest, pair, now)
            except Exception as exc:
                print(f"[headset] send failed: {exc}")

    def _destination(self):
        sess = self._link.session if self._link else None
        if sess is None:
            return None
        return (sess.addr, getattr(sess, "video_port", self.ports["video"])), sess

    def _send_pair(self, dest_and_session, pair, now: float) -> None:
        (dest, sess) = dest_and_session
        codec = getattr(sess, "codec", self.codec)
        want_fovea = self.foveation and getattr(sess, "foveation", True)

        if self._active != (dest, codec):
            self._active = (dest, codec)
            self._senders = {
                eye: EyeStreamSender(self._sock, dest, eye, self.layout, self.fps,
                                     self._rate.target_kbps, True, codec=codec,
                                     jpeg_quality=85, mtu_payload=self.mtu_payload)
                for eye in (EYE_LEFT, EYE_RIGHT)
            }
            print(f"[headset] streaming to {dest[0]}:{dest[1]} "
                  f"fovea={'on' if want_fovea else 'off'}")

        gl = gr = None
        if want_fovea:
            pkt = self._fresh_input()
            if pkt is not None and pkt.gaze_valid:
                gl, gr = tuple(pkt.gaze_l), tuple(pkt.gaze_r)
            else:
                gl = gr = (0.5, 0.5)

        zoom = self._widener.update(gl, now)
        left, right = pair
        self._senders[EYE_LEFT].send(left, gl, zoom)
        self._senders[EYE_RIGHT].send(right, gr, zoom)
        self.frames_sent += 1

    # -------------------------------------------------------------------- input

    def _input_loop(self) -> None:
        rate_at, rate_n = time.monotonic(), 0
        while not self._stop.is_set():
            try:
                data, _ = self._input_sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            pkt = HeadsetInput.unpack(data)
            if pkt is None:
                continue
            now = time.monotonic()
            with self._input_lock:
                self._input_pkt, self._input_at = pkt, now
            rate_n += 1
            if now - rate_at >= 1.0:
                self.input_rate_hz = rate_n / (now - rate_at)
                rate_at, rate_n = now, 0

    def _fresh_input(self, stale_s: float = 0.5) -> HeadsetInput | None:
        with self._input_lock:
            if self._input_pkt is None:
                return None
            if time.monotonic() - self._input_at > stale_s:
                return None
            return self._input_pkt

    def receive_data(self) -> HeadsetData | None:
        """
        Newest headset state as a HeadsetData, or None if nothing fresh has arrived.

        Poses go through the same left-to-right handedness conversion the WebRTC path
        used, so the arm control maths downstream is unchanged.
        """
        pkt = self._fresh_input()
        if pkt is None:
            return None

        d = HeadsetData()
        d.h_pos, d.h_quat = convert_left_to_right_coordinates(
            np.asarray(pkt.head_pos, dtype=float), np.asarray(pkt.head_rot, dtype=float))
        d.l_pos, d.l_quat = convert_left_to_right_coordinates(
            np.asarray(pkt.left.pos, dtype=float), np.asarray(pkt.left.rot, dtype=float))
        d.r_pos, d.r_quat = convert_left_to_right_coordinates(
            np.asarray(pkt.right.pos, dtype=float), np.asarray(pkt.right.rot, dtype=float))

        d.l_thumbstick_x, d.l_thumbstick_y = float(pkt.left.stick[0]), float(pkt.left.stick[1])
        d.r_thumbstick_x, d.r_thumbstick_y = float(pkt.right.stick[0]), float(pkt.right.stick[1])
        d.l_index_trigger, d.l_hand_trigger = float(pkt.left.trigger), float(pkt.left.grip)
        d.r_index_trigger, d.r_hand_trigger = float(pkt.right.trigger), float(pkt.right.grip)

        d.l_button_one = pkt.left.held(BUTTON_ONE)
        d.l_button_two = pkt.left.held(BUTTON_TWO)
        d.l_button_thumbstick = pkt.left.held(BUTTON_STICK)
        d.r_button_one = pkt.right.held(BUTTON_ONE)
        d.r_button_two = pkt.right.held(BUTTON_TWO)
        d.r_button_thumbstick = pkt.right.held(BUTTON_STICK)

        # Additive: gaze in normalised source-image coordinates (x right, y down).
        # The WebRTC HeadsetData had no equivalent, so nothing downstream reads these
        # yet -- they are here because the uplink now carries them.
        d.gaze_l = tuple(pkt.gaze_l)
        d.gaze_r = tuple(pkt.gaze_r)
        d.gaze_confidence = float(pkt.gaze_confidence)
        d.gaze_valid = bool(pkt.gaze_valid)
        return d

    # ----------------------------------------------------------------- feedback

    def send_feedback(self, data: HeadsetFeedback) -> None:
        """
        Publish robot state on the control channel.

        NOTE: the v2 viewer subscribes to "arm/state" but only renders it as a debug
        string; it has no equivalent of the WebRTC build's out-of-sync ghost arms or
        info banner. This is sent so the data is there when the viewer grows a use for
        it, not because anything draws it today.
        """
        if self._link is None or not self._link.connected:
            return

        def vec(v):
            return [float(x) for x in np.asarray(v, dtype=float).ravel()]

        self._link.publish("arm/state", {
            "info": str(getattr(data, "info", "") or ""),
            "headOutOfSync": bool(getattr(data, "head_out_of_sync", False)),
            "leftOutOfSync": bool(getattr(data, "left_out_of_sync", False)),
            "rightOutOfSync": bool(getattr(data, "right_out_of_sync", False)),
            "leftArmPosition": vec(getattr(data, "left_arm_position", np.zeros(3))),
            "leftArmRotation": vec(getattr(data, "left_arm_rotation", np.zeros(4))),
            "rightArmPosition": vec(getattr(data, "right_arm_position", np.zeros(3))),
            "rightArmRotation": vec(getattr(data, "right_arm_rotation", np.zeros(4))),
            "middleArmPosition": vec(getattr(data, "middle_arm_position", np.zeros(3))),
            "middleArmRotation": vec(getattr(data, "middle_arm_rotation", np.zeros(4))),
        })

    # -------------------------------------------------------------------- stats

    def describe(self) -> str:
        return (f"tx {self.frames_sent} frames, {self.frames_dropped} dropped in mailbox, "
                f"uplink {self.input_rate_hz:.0f} Hz, {self._rate.describe()}, "
                f"{self._link.session if self._link else 'no link'}")


# Same name as the WebRTC class so `from gvlink_headset import GvLinkHeadset as
# WebRTCHeadset` reads honestly at the call sites that want a one-line swap.
WebRTCHeadset = GvLinkHeadset
