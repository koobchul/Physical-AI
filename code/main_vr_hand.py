"""
Quest 2 (WiVRn/OpenXR) -> robosuite 양팔 텔레오퍼레이션

    python main_vr_xr.py [--scale 0.6] [--hand-side right]

같은 폴더에 dual_xarm_env.py 가 있어야 한다.

조작 (조작자가 받침대 뒤에서 로봇과 같은 방향을 볼 때)
    [머리]              시야 (트래킹)
    [오른손] 그립 홀드   오른팔(INSPIRE) 추종 ON
    [오른손] 트리거      손 쥐기
    [왼손]  그립 홀드    왼팔(그리퍼) 추종 ON
    [왼손]  트리거      그리퍼 닫기
    [왼손]  스틱        작업 공간 안에서 이동
    [왼손]  X 버튼      시점 초기화
    Ctrl+C             종료

구조
    72Hz  포즈 -> 렌더 -> 제출        (지연을 줄이려 제출을 먼저)
    20Hz  물리 (누적된 손 델타를 한 번에 적용)

지난주 SteamVR 에서 얻은 교훈 두 가지가 그대로 적용되어 있다.
    1. mjvGLCamera.frustum_width 는 '반폭'이다. 전체 폭을 넣으면 가로 화각이
       2배가 되어 화면이 세로로 2배 길어 보인다.
    2. 렌더 종횡비는 권장 해상도가 아니라 frustum 에서 직접 유도해야 한다.
       (WiVRn 실측: frustum 0.9329 vs 권장 해상도 0.9090, 2.6% 차이)
"""

import argparse
import ctypes
import math
import os
import sys
import threading
import time

import glfw
import mujoco
import numpy as np
import xr
from OpenGL import GL
from scipy.spatial.transform import Rotation as R

from dual_xarm_env import make_env

# --------------------------------------------------------------------------
# 좌표계
#   OpenXR(=OpenVR) : +X 오른쪽, +Y 위,  -Z 전방
#   robosuite       : +X 전방,   +Y 왼쪽, +Z 위
# --------------------------------------------------------------------------
VR2ROBOT = np.array([
    [0.0, 0.0, -1.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])

# VR2ROBOT 은 시작할 때 한 번 '리센터' 되면서 요(yaw) 회전이 곱해진다.
# 아래 recenter() 참조. VR2ROBOT_BASE 는 순수한 축 변환만 담고 있다.
VR2ROBOT_BASE = VR2ROBOT.copy()

POS_OUTPUT_MAX = 0.05      # OSC_POSE 기본값. 액션 1.0 == 한 스텝에 5cm
ROT_OUTPUT_MAX = 0.5       # rad
POS_SCALE = 1.0            # 사람 손 대비 로봇 배율 (되먹임이라 그대로 반영된다)
ROT_SCALE = 1.0
POS_DEADZONE = 5e-4
ROT_DEADZONE = 2e-3

STICK_DEADZONE = 0.15
MOVE_SPEED = 1.2           # m/s
LIFT_SPEED = 0.8

NEAR, FAR = 0.05, 50.0
DEFAULT_ORIGIN = np.array([-1.5, 0.0, 0.0])   # 받침대 뒤에서 테이블을 봄

GL_SRGB8_ALPHA8 = 0x8C43
GL_RGBA8 = 0x8058

LEFT, RIGHT = 0, 1

# XR_EXT_hand_tracking 의 26개 관절 중 우리가 쓰는 것
J_PALM, J_WRIST = 0, 1
J_TIP = {"thumb": 5, "index": 10, "middle": 15, "ring": 20, "pinky": 25}
J_MCP = {"thumb": 2, "index": 6, "middle": 11, "ring": 16, "pinky": 21}
# INSPIRE RH56 의 6 DOF 순서 (dual_xarm_env.py 참조)
INSPIRE_ORDER = ["pinky", "ring", "middle", "index", "thumb", "thumb_rot"]


def rss_mb():
    """프로세스 상주 메모리 (MB). 시간에 따라 계속 오르면 누수다."""
    try:
        with open(f"/proc/{os.getpid()}/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e6
    except Exception:
        return float("nan")


def drain_gl():
    n = 0
    while GL.glGetError() != GL.GL_NO_ERROR and n < 32:
        n += 1
    return n


def recenter(head_pos_vr, head_rot_vr, home, start_yaw_deg=0.0):
    """머리의 현재 위치와 방향을 상쇄해서, 방 어디에 어느 쪽을 보고 서 있든
    항상 home 에서 start_yaw 방향을 보고 시작하게 만든다.

    VR2ROBOT 을 통째로 다시 만들어 전역에 넣는다. 시야·손 매핑·스틱 이동이
    모두 이 행렬을 매번 읽으므로 한 번에 같이 돌아간다.

    이게 없으면 START_YAW 를 고정값으로 박아야 하는데, 그 값은 그때 서 있던
    자리와 방향에만 맞는다 - 방에서 한 발만 옮겨도 다시 틀어졌다 (2026-09-30).

    returns 새 origin
    """
    global VR2ROBOT
    # 지금 머리가 로봇 좌표에서 향한 방향
    fwd = -(VR2ROBOT_BASE @ head_rot_vr)[:, 2]
    yaw = math.atan2(fwd[1], fwd[0])
    theta = math.radians(start_yaw_deg) - yaw
    VR2ROBOT = R.from_euler("z", theta).as_matrix() @ VR2ROBOT_BASE
    # 이 행렬에서 머리가 home 에 오도록 원점을 잡는다.
    # 높이(z)는 건드리지 않는다 - 실제 머리 높이를 그대로 써야 몸을 숙이거나
    # 일어설 때 시야가 따라온다. 상쇄하면 바닥에 앉는다 (2026-09-30).
    off = VR2ROBOT @ np.asarray(head_pos_vr, float)
    off[2] = 0.0
    return np.asarray(home, float) - off


def xr_quat_to_mat(q):
    return R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()


# ==========================================================================
# OpenXR
# ==========================================================================
class XRSession:
    """인스턴스/세션/공간/액션/스왑체인을 한 덩어리로 들고 있는다."""

    def __init__(self, render_scale=0.6, want_hand=True):
        exts = {e.extension_name.decode()
                for e in xr.enumerate_instance_extension_properties()}
        if xr.KHR_OPENGL_ENABLE_EXTENSION_NAME not in exts:
            sys.exit("이 런타임에는 OpenGL 확장이 없습니다.")
        want = [xr.KHR_OPENGL_ENABLE_EXTENSION_NAME]
        self.has_hand = (want_hand
                         and xr.EXT_HAND_TRACKING_EXTENSION_NAME in exts)
        if self.has_hand:
            want.append(xr.EXT_HAND_TRACKING_EXTENSION_NAME)
        self.has_hand_interaction = xr.EXT_HAND_INTERACTION_EXTENSION_NAME in exts
        if self.has_hand_interaction:
            want.append(xr.EXT_HAND_INTERACTION_EXTENSION_NAME)

        self.instance = xr.create_instance(xr.InstanceCreateInfo(
            application_info=xr.ApplicationInfo(
                application_name="robosuite-teleop", application_version=1,
                engine_name="mujoco", engine_version=1,
                api_version=xr.Version(1, 0, 34)),
            enabled_extension_names=want))
        self.system = xr.get_system(self.instance, xr.SystemGetInfo(
            form_factor=xr.FormFactor.HEAD_MOUNTED_DISPLAY))
        self.gl_req = xr.get_opengl_graphics_requirements_khr(self.instance, self.system)
        print(f"[XR] {xr.get_instance_properties(self.instance).runtime_name.decode()}  "
              f"(요구 GL {self.gl_req.min_api_version_supported} 이상)")

        # --- 눈 파라미터로부터 렌더 크기 결정 --------------------------
        cfg = xr.enumerate_view_configuration_views(
            self.instance, self.system, xr.ViewConfigurationType.PRIMARY_STEREO)
        self._rec = (cfg[0].recommended_image_rect_width,
                     cfg[0].recommended_image_rect_height)
        self.eye_h = max(64, int(self._rec[1] * render_scale) // 2 * 2)
        self.eye_w = self.eye_h   # locate_views 로 실제 fov 를 받은 뒤 고친다
        self._size_fixed = False

        self._gl_window()
        self.session = xr.create_session(self.instance, xr.SessionCreateInfo(
            next=ctypes.cast(ctypes.pointer(self._binding), ctypes.c_void_p),
            system_id=self.system))

        try:
            self.world = xr.create_reference_space(self.session,
                xr.ReferenceSpaceCreateInfo(
                    reference_space_type=xr.ReferenceSpaceType.STAGE))
            self.stage = True
            print("[XR] 기준 공간: STAGE (바닥 기준, 머리 높이를 그대로 쓴다)")
        except Exception:
            self.world = xr.create_reference_space(self.session,
                xr.ReferenceSpaceCreateInfo(
                    reference_space_type=xr.ReferenceSpaceType.LOCAL))
            self.stage = False
            print("[XR] 기준 공간: LOCAL (머리 기준, --eye-height 를 더한다)")

        self._make_actions()

        # 손 트래커. 실제 카메라 트래킹이 꺼져 있으면 런타임이 컨트롤러에서
        # 합성한 포즈를 주는데, 그때는 손가락이 안 움직여서 보정이 끝나지
        # 않는다 -> 자동으로 트리거 방식으로 남는다.
        self.hand_reader = None
        self.trackers = {}
        if self.has_hand:
            try:
                self.hand_reader = HandJointReader(self.instance)
                for name, h in (("left", xr.HandEXT.LEFT),
                                ("right", xr.HandEXT.RIGHT)):
                    self.trackers[name] = xr.create_hand_tracker_ext(
                        self.session, xr.HandTrackerCreateInfoEXT(
                            hand=h, hand_joint_set=xr.HandJointSetEXT.DEFAULT))
                print("[손] 손 트래커 준비됨 (XR_EXT_hand_tracking)")
            except Exception as e:
                print(f"[손] 트래커 생성 실패: {type(e).__name__}: {e}")
                self.hand_reader, self.trackers = None, {}
        elif want_hand:
            print("[손] 런타임에 XR_EXT_hand_tracking 이 없습니다 — 트리거로 조작")

        self.swapchains = None          # 첫 프레임에서 실제 fov 를 보고 만든다
        self._proj_views = (xr.CompositionLayerProjectionView * 2)()
        self.running = False
        self.exiting = False
        self.state = "IDLE"
        self.profile = {"left": "", "right": ""}

    # -- GL -------------------------------------------------------------
    def _gl_window(self):
        if not glfw.init():
            sys.exit("glfw init 실패")

        # OpenXR 은 최신 GL 을 원하고 MuJoCo 는 compatibility 프로파일을 원한다.
        # 프로파일을 지정하지 않고 4.6 을 요청하면 NVIDIA 는 core 를 주는데,
        # core 에서는 glGetString(GL_EXTENSIONS) 가 NULL 이라 MuJoCo 의 확장
        # 검사가 통째로 실패한다 ("ARB_framebuffer_object required").
        # 그래서 compat 을 명시하고, 안 되면 힌트 없는 기본 컨텍스트로 내려간다.
        attempts = [
            ("4.6 compat", [(glfw.CONTEXT_VERSION_MAJOR, 4),
                            (glfw.CONTEXT_VERSION_MINOR, 6),
                            (glfw.OPENGL_PROFILE, glfw.OPENGL_COMPAT_PROFILE)]),
            ("기본값", []),
            ("3.3 compat", [(glfw.CONTEXT_VERSION_MAJOR, 3),
                            (glfw.CONTEXT_VERSION_MINOR, 3),
                            (glfw.OPENGL_PROFILE, glfw.OPENGL_COMPAT_PROFILE)]),
        ]
        self.win = None
        for label, hints in attempts:
            glfw.default_window_hints()
            glfw.window_hint(glfw.VISIBLE, False)
            for k, v in hints:
                glfw.window_hint(k, v)
            win = glfw.create_window(64, 64, "xr", None, None)
            if not win:
                print(f"[GL] {label}: 컨텍스트 생성 실패, 다음 후보로")
                continue
            glfw.make_context_current(win)
            # MuJoCo 가 하는 것과 같은 방식으로 확장 목록을 볼 수 있는지 확인
            try:
                exts_ok = GL.glGetString(GL.GL_EXTENSIONS) is not None
            except Exception:
                exts_ok = False
            ver = GL.glGetString(GL.GL_VERSION)
            ver = ver.decode() if ver else "?"
            if not exts_ok:
                print(f"[GL] {label}: core 프로파일({ver}) — MuJoCo 가 못 씁니다, 다음 후보로")
                glfw.destroy_window(win)
                continue
            self.win = win
            print(f"[GL] {label}: {ver} / {GL.glGetString(GL.GL_RENDERER).decode()}")
            break
        if self.win is None:
            sys.exit("MuJoCo 가 쓸 수 있는 GL 컨텍스트를 만들지 못했습니다.")

        ft = dict(xr.GraphicsBindingOpenGLXlibKHR._fields_)

        def ptr(v, t):
            if isinstance(v, int):
                v = ctypes.c_void_p(v)
            return ctypes.cast(v, t)

        b = xr.GraphicsBindingOpenGLXlibKHR()
        b.x_display = ptr(glfw.get_x11_display(), ft["x_display"])
        d = glfw.get_glx_window(self.win)
        b.glx_drawable = d if isinstance(d, int) else ctypes.cast(d, ctypes.c_void_p).value
        b.glx_context = ptr(glfw.get_glx_context(self.win), ft["glx_context"])
        self._binding = b

    # -- 액션 -----------------------------------------------------------
    def _make_actions(self):
        """바인딩을 여러 인터랙션 프로파일에 건다.

        손 추적을 켜면 WiVRn 이 손을 'hand interaction' 프로파일로 올려서,
        touch_controller 에만 바인딩해 두면 그 손의 입력이 통째로 사라진다
        (2026-09-29 실측: 왼손이 죽고 클러치가 안 잡힘). 프로파일은
        /user/hand/left 와 /user/hand/right 가 각각 따로 정해지므로,
        한 손은 컨트롤러 / 한 손은 맨손 조합도 여기만 채우면 된다."""
        i = self.instance
        self.aset = xr.create_action_set(i, xr.ActionSetCreateInfo(
            action_set_name="teleop", localized_action_set_name="Teleop", priority=0))
        self.hands = [xr.string_to_path(i, "/user/hand/left"),
                      xr.string_to_path(i, "/user/hand/right")]

        def mk(name, t):
            return xr.create_action(self.aset, xr.ActionCreateInfo(
                action_name=name, action_type=t,
                subaction_paths=self.hands, localized_action_name=name))

        self.a_pose = mk("grip_pose", xr.ActionType.POSE_INPUT)
        self.a_trig = mk("trigger", xr.ActionType.FLOAT_INPUT)
        self.a_sqz = mk("squeeze", xr.ActionType.FLOAT_INPUT)
        self.a_stick = mk("stick", xr.ActionType.VECTOR2F_INPUT)
        self.a_btn = mk("button", xr.ActionType.BOOLEAN_INPUT)    # 팔 추종 토글
        self.a_btn2 = mk("button2", xr.ActionType.BOOLEAN_INPUT)  # B / Y
        self.a_stick_click = mk("stick_click", xr.ActionType.BOOLEAN_INPUT)

        P = lambda t: xr.string_to_path(i, t)

        def pose_trig_sqz(side, trig, sqz):
            b = [xr.ActionSuggestedBinding(self.a_pose,
                     P(f"/user/hand/{side}/input/grip/pose"))]
            if trig:
                b.append(xr.ActionSuggestedBinding(self.a_trig,
                     P(f"/user/hand/{side}/input/{trig}")))
            if sqz:
                b.append(xr.ActionSuggestedBinding(self.a_sqz,
                     P(f"/user/hand/{side}/input/{sqz}")))
            return b

        profiles = {}

        # Quest 컨트롤러
        b = []
        for side, k1, k2 in (("left", "x", "y"), ("right", "a", "b")):
            b += pose_trig_sqz(side, "trigger/value", "squeeze/value")
            b += [
                xr.ActionSuggestedBinding(self.a_stick,
                    P(f"/user/hand/{side}/input/thumbstick")),
                xr.ActionSuggestedBinding(self.a_btn,
                    P(f"/user/hand/{side}/input/{k1}/click")),
                xr.ActionSuggestedBinding(self.a_btn2,
                    P(f"/user/hand/{side}/input/{k2}/click")),
                xr.ActionSuggestedBinding(self.a_stick_click,
                    P(f"/user/hand/{side}/input/thumbstick/click")),
            ]
        profiles["/interaction_profiles/oculus/touch_controller"] = b

        # 맨손 (XR_EXT_hand_interaction). 손을 오므리면 grasp, 집으면 pinch.
        if self.has_hand_interaction:
            b = []
            for side in ("left", "right"):
                b += pose_trig_sqz(side, "pinch_ext/value", "grasp_ext/value")
            profiles["/interaction_profiles/ext/hand_interaction_ext"] = b

        # 어떤 런타임에도 있는 최소 프로파일. 위가 다 안 붙어도 포즈는 건진다.
        b = []
        for side in ("left", "right"):
            b += [
                xr.ActionSuggestedBinding(self.a_pose,
                    P(f"/user/hand/{side}/input/grip/pose")),
                xr.ActionSuggestedBinding(self.a_btn,
                    P(f"/user/hand/{side}/input/select/click")),
                xr.ActionSuggestedBinding(self.a_btn2,
                    P(f"/user/hand/{side}/input/menu/click")),
            ]
        profiles["/interaction_profiles/khr/simple_controller"] = b

        ok = []
        for path, binds in profiles.items():
            try:
                xr.suggest_interaction_profile_bindings(
                    i, xr.InteractionProfileSuggestedBinding(
                        interaction_profile=P(path), suggested_bindings=binds))
                ok.append(path.rsplit("/", 1)[-1])
            except Exception as e:
                print(f"[XR] 프로파일 {path} 바인딩 실패 ({type(e).__name__}) — 건너뜁니다")
        print(f"[XR] 바인딩한 프로파일: {', '.join(ok)}")

        xr.attach_session_action_sets(self.session,
            xr.SessionActionSetsAttachInfo(action_sets=[self.aset]))

        self.hand_space = [xr.create_action_space(self.session,
            xr.ActionSpaceCreateInfo(action=self.a_pose, subaction_path=h))
            for h in self.hands]
        self._active = xr.ActionsSyncInfo(active_action_sets=[
            xr.ActiveActionSet(action_set=self.aset, subaction_path=xr.NULL_PATH)])

    # -- 스왑체인 (실제 fov 를 본 뒤 생성) --------------------------------
    def _make_swapchains(self, fov):
        l, r = math.tan(fov.angle_left), math.tan(fov.angle_right)
        b, t = math.tan(fov.angle_down), math.tan(fov.angle_up)
        aspect = (r - l) / (t - b)
        self.eye_w = max(64, int(round(self.eye_h * aspect)) // 2 * 2)

        fmts = xr.enumerate_swapchain_formats(self.session)
        fmt = next((f for f in (GL_SRGB8_ALPHA8, GL_RGBA8) if f in fmts), fmts[0])

        self.swapchains = []
        for _ in range(2):
            sc = xr.create_swapchain(self.session, xr.SwapchainCreateInfo(
                usage_flags=(xr.SwapchainUsageFlags.COLOR_ATTACHMENT_BIT
                             | xr.SwapchainUsageFlags.TRANSFER_DST_BIT),
                format=fmt, sample_count=1,
                width=self.eye_w, height=self.eye_h,
                face_count=1, array_size=1, mip_count=1))
            imgs = xr.enumerate_swapchain_images(sc, xr.SwapchainImageOpenGLKHR)
            fbos = []
            for im in imgs:
                f = GL.glGenFramebuffers(1)
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, f)
                GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
                                          GL.GL_TEXTURE_2D, im.image, 0)
                if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE:
                    raise RuntimeError("스왑체인 FBO 미완성")
                fbos.append(f)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
            self.swapchains.append((sc, fbos))

        print(f"[XR] 권장 눈당 {self._rec[0]}x{self._rec[1]} -> 렌더 "
              f"{self.eye_w}x{self.eye_h} (frustum 종횡비 {aspect:.4f}, "
              f"포맷 0x{fmt:X})")
        print(f"[XR] 화각 수평 {math.degrees(fov.angle_right - fov.angle_left):.1f}도 "
              f"수직 {math.degrees(fov.angle_up - fov.angle_down):.1f}도")

    # -- 프레임 ----------------------------------------------------------
    def poll(self):
        while True:
            try:
                ev = xr.poll_event(self.instance)
            except xr.EventUnavailable:
                return
            if ev.type == xr.StructureType.EVENT_DATA_SESSION_STATE_CHANGED:
                st = ctypes.cast(ctypes.byref(ev), ctypes.POINTER(
                    xr.EventDataSessionStateChanged)).contents.state
                self.state = xr.SessionState(st).name
                print(f"[XR] 세션 {self.state}")
                if st == xr.SessionState.READY:
                    xr.begin_session(self.session, xr.SessionBeginInfo(
                        primary_view_configuration_type=xr.ViewConfigurationType.PRIMARY_STEREO))
                    self.running = True
                elif st == xr.SessionState.STOPPING:
                    xr.end_session(self.session)
                    self.running = False
                elif st in (xr.SessionState.EXITING, xr.SessionState.LOSS_PENDING):
                    self.exiting = True
            elif ev.type == xr.StructureType.EVENT_DATA_INTERACTION_PROFILE_CHANGED:
                # 컨트롤러를 들거나 놓으면 런타임이 프로파일을 바꾼다
                self.show_profiles(" (변경됨)")

    def show_profiles(self, tag=""):
        """손별로 지금 무엇이 잡혀 있는지. 왼손은 컨트롤러, 오른손은 맨손처럼
        서로 다른 프로파일이 잡히는 게 정상이다."""
        out = []
        for name, side, path in (("왼손", "left", self.hands[LEFT]),
                                 ("오른손", "right", self.hands[RIGHT])):
            try:
                st = xr.get_current_interaction_profile(self.session, path)
                prof = (xr.path_to_string(self.instance, st.interaction_profile)
                        if st.interaction_profile else "")
            except Exception:
                prof = ""
            self.profile[side] = prof
            out.append(f"{name}={prof.rsplit('/',1)[-1] if prof else '없음'}")
        print(f"[XR] 인터랙션 프로파일{tag}: " + "  ".join(out)
              + ("   [맨손 모드]" if self.bare("right") else "   [컨트롤러 모드]"))

    def bare(self, side):
        """이 손이 맨손(hand interaction)으로 잡혀 있는가."""
        return "hand_interaction" in self.profile.get(side, "")

    def sync_input(self):
        try:
            xr.sync_actions(self.session, self._active)
            return True
        except xr.SessionNotFocused:
            return False

    def hand_state(self, i, t):
        """returns dict(pos, rot, trigger, squeeze, stick, button, valid)"""
        loc = xr.locate_space(self.hand_space[i], self.world, t)
        valid = bool(loc.location_flags & xr.SPACE_LOCATION_POSITION_VALID_BIT
                     and loc.location_flags & xr.SPACE_LOCATION_ORIENTATION_VALID_BIT)
        gi = lambda a: xr.ActionStateGetInfo(action=a, subaction_path=self.hands[i])
        st = xr.get_action_state_vector2f(self.session, gi(self.a_stick)).current_state
        p = loc.pose.position
        return dict(
            valid=valid,
            pos=np.array([p.x, p.y, p.z]),
            rot=xr_quat_to_mat(loc.pose.orientation),
            trigger=xr.get_action_state_float(self.session, gi(self.a_trig)).current_state,
            squeeze=xr.get_action_state_float(self.session, gi(self.a_sqz)).current_state,
            stick=(st.x, st.y),
            button=bool(xr.get_action_state_boolean(self.session, gi(self.a_btn)).current_state),
            button2=bool(xr.get_action_state_boolean(self.session, gi(self.a_btn2)).current_state),
            stick_click=bool(xr.get_action_state_boolean(
                self.session, gi(self.a_stick_click)).current_state),
        )

    def hand_probe(self, name, t):
        """진단용. (활성, 손바닥 위치, 유효한 관절 수)"""
        act, arr = self.hand_joints(name, t)
        if arr is None:
            return None, None, 0
        need = xr.SPACE_LOCATION_POSITION_VALID_BIT
        n = sum(1 for i in range(HandJointReader.N)
                if arr[i].location_flags & need)
        p = arr[J_PALM].pose.position
        return act, np.array([p.x, p.y, p.z]), n

    def hand_joints(self, name, t):
        """returns (active, joint_array) 또는 (None, None)"""
        tr = self.trackers.get(name)
        if tr is None or self.hand_reader is None:
            return None, None
        return self.hand_reader.locate(tr, self.world, t)

    def hand_pose_state(self, name, t, base=None):
        """손 트래킹 관절에서 ArmTeleop 이 먹을 수 있는 상태를 만든다.

        회전은 손바닥 관절 자세를 그대로 쓴다. ArmTeleop 은 기준 자세 대비
        '차이'만 보므로, 손 프레임과 컨트롤러 프레임이 상수만큼 달라도
        rot2 @ (rot1 @ B).T 에서 B 가 정확히 상쇄된다."""
        act, arr = self.hand_joints(name, t)
        need = (xr.SPACE_LOCATION_POSITION_VALID_BIT
                | xr.SPACE_LOCATION_ORIENTATION_VALID_BIT)
        if not act or arr is None or not (arr[J_PALM].location_flags & need):
            return None
        p = arr[J_PALM].pose.position
        # 포즈만 관절에서 가져오고 나머지 입력(핀치 등)은 액션 쪽을 살린다.
        # 예전에는 trigger 를 0 으로 박아서 맨손일 때 그리퍼가 영영 안 닫혔다.
        out = dict(base) if base else dict(
            trigger=0.0, squeeze=0.0, stick=(0.0, 0.0),
            button=False, button2=False, stick_click=False)
        out.update(valid=True,
                   pos=np.array([p.x, p.y, p.z]),
                   rot=xr_quat_to_mat(arr[J_PALM].pose.orientation))
        return out

    def submit(self, views, blit_fn):
        """blit_fn(eye_index, fbo, w, h) 가 해당 눈 화면을 스왑체인 FBO 에 그린다."""
        for i in range(2):
            sc, fbos = self.swapchains[i]
            idx = xr.acquire_swapchain_image(sc, xr.SwapchainImageAcquireInfo())
            xr.wait_swapchain_image(sc, xr.SwapchainImageWaitInfo(
                timeout=xr.INFINITE_DURATION))
            blit_fn(i, fbos[idx], self.eye_w, self.eye_h)
            xr.release_swapchain_image(sc, xr.SwapchainImageReleaseInfo())

            self._proj_views[i] = xr.CompositionLayerProjectionView(
                pose=views[i].pose, fov=views[i].fov,
                sub_image=xr.SwapchainSubImage(
                    swapchain=sc,
                    image_rect=xr.Rect2Di(offset=xr.Offset2Di(0, 0),
                                          extent=xr.Extent2Di(self.eye_w, self.eye_h)),
                    image_array_index=0))
        layer = xr.CompositionLayerProjection(space=self.world, views=self._proj_views)
        return [ctypes.byref(layer)], layer   # layer 를 살려둬야 GC 되지 않는다

    def close(self):
        if self.swapchains:
            for sc, fbos in self.swapchains:
                for f in fbos:
                    try: GL.glDeleteFramebuffers(1, [f])
                    except Exception: pass
                try: xr.destroy_swapchain(sc)
                except Exception: pass
        for fn in (lambda: xr.destroy_session(self.session),
                   lambda: xr.destroy_instance(self.instance),
                   glfw.terminate):
            try: fn()
            except Exception: pass


# ==========================================================================
# MuJoCo 렌더
# ==========================================================================
class MuJoCoEyeRenderer:
    """눈 하나 크기의 오프스크린 버퍼에 mono 로 두 번 렌더한다.
    (SBS 모드는 종횡비 규약이 모호해서 쓰지 않는다 — 지난주 교훈)"""

    def __init__(self, model, w, h, msaa=True):
        self.w, self.h = w, h
        model.vis.global_.offwidth = w
        model.vis.global_.offheight = h
        if not msaa:
            model.vis.quality.offsamples = 0
        self.con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.con)
        ow = getattr(self.con, "offWidth", -1)
        oh = getattr(self.con, "offHeight", -1)
        print(f"[MJ] 오프스크린 요청 {w}x{h} / 실제 {ow}x{oh} "
              f"samples={getattr(self.con, 'offSamples', -1)}")
        if (ow, oh) != (w, h):
            print("     >>> 크기가 다릅니다. blit 이 엉뚱한 영역을 읽게 됩니다.")
        self.viewport = mujoco.MjrRect(0, 0, w, h)

    def render(self, scn):
        # blit 이 매 프레임 프레임버퍼를 0 으로 되돌리므로 매번 다시 잡는다
        # (이걸 빼면 숨겨진 glfw 창에 그리게 된다 — 지난주 '검은 화면'의 원인)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.con)
        mujoco.mjr_render(self.viewport, scn, self.con)

    def overlay(self, top_left, bottom_left=""):
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,
                           mujoco.mjtGridPos.mjGRID_TOPLEFT,
                           self.viewport, top_left, bottom_left, self.con)

    def blit_to(self, fbo, w, h, flip_y=False):
        src = self.con.offFBO
        if getattr(self.con, "offSamples", 0) > 0:
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.con.offFBO)
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.con.offFBO_r)
            GL.glBlitFramebuffer(0, 0, self.w, self.h, 0, 0, self.w, self.h,
                                 GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)
            src = self.con.offFBO_r
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, src)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, fbo)
        if flip_y:
            GL.glBlitFramebuffer(0, 0, self.w, self.h, 0, h, w, 0,
                                 GL.GL_COLOR_BUFFER_BIT, GL.GL_LINEAR)
        else:
            GL.glBlitFramebuffer(0, 0, self.w, self.h, 0, 0, w, h,
                                 GL.GL_COLOR_BUFFER_BIT, GL.GL_LINEAR)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)


def set_eye_camera(scn, view, origin):
    """OpenXR 의 눈 pose/fov 를 mjvGLCamera 두 슬롯에 모두 넣는다.
    mono 렌더에서 MuJoCo 가 어느 슬롯을 쓰든 같은 결과가 나오도록."""
    p = view.pose.position
    pos = VR2ROBOT @ np.array([p.x, p.y, p.z]) + origin

    # R_vr 의 열 = 눈 로컬 축을 VR 월드로 표현한 것.
    # 로봇 좌표로 옮기려면 왼쪽 곱 한 번뿐이다. (C R C^T 는 상대회전용)
    axes = VR2ROBOT @ xr_quat_to_mat(view.pose.orientation)
    forward = -axes[:, 2]     # GL 카메라는 로컬 -Z 를 본다
    up = axes[:, 1]

    f = view.fov
    left, right = math.tan(f.angle_left) * NEAR, math.tan(f.angle_right) * NEAR
    bottom, top = math.tan(f.angle_down) * NEAR, math.tan(f.angle_up) * NEAR

    for slot in range(2):
        c = scn.camera[slot]
        c.pos[:] = pos.astype(np.float32)
        c.forward[:] = forward.astype(np.float32)
        c.up[:] = up.astype(np.float32)
        c.frustum_near = NEAR
        c.frustum_far = FAR
        c.frustum_bottom = bottom
        c.frustum_top = top
        c.frustum_center = 0.5 * (left + right)
        try:
            # 주의: frustum_width 는 '반폭'이다. 전체 폭을 넣으면 가로 화각이
            # 2배가 되어 화면이 세로로 2배 길어 보인다 (2026-09-22 실측).
            c.frustum_width = 0.5 * (right - left)
        except AttributeError:
            pass          # 구버전 MuJoCo: 뷰포트 종횡비로 폭이 결정됨

    # 헤드라이트를 눈 위치로
    try:
        if scn.nlight > 0:
            scn.lights[0].pos[:] = pos.astype(np.float32)
            scn.lights[0].dir[:] = forward.astype(np.float32)
    except Exception:
        pass
    return pos, forward


# ==========================================================================
class HandJointReader:
    """pyopenxr 의 locate_hand_joints_ext 는 출력 배열을 할당하지 않고 NULL 을
    넘겨서 XR_ERROR_VALIDATION_FAILURE 가 난다. 배열을 직접 잡아 raw 호출한다.
    (xr_session_probe.py 에서 검증된 우회)"""

    N = xr.HAND_JOINT_COUNT_EXT     # 26

    def __init__(self, instance):
        self._fn = ctypes.cast(
            xr.get_instance_proc_addr(instance, "xrLocateHandJointsEXT"),
            xr.PFN_xrLocateHandJointsEXT)
        self._buf = {}

    def locate(self, tracker, space, time_ns):
        arr = self._buf.get(id(tracker))
        if arr is None:
            arr = (xr.HandJointLocationEXT * self.N)()
            self._buf[id(tracker)] = arr
        out = xr.HandJointLocationsEXT(
            joint_count=self.N,
            joint_locations=ctypes.cast(arr, ctypes.POINTER(xr.HandJointLocationEXT)))
        info = xr.HandJointsLocateInfoEXT(base_space=space, time=time_ns)
        if self._fn(tracker, ctypes.byref(info), ctypes.byref(out)) != xr.Result.SUCCESS:
            return None, arr
        return bool(out.is_active), arr


class InspireHandMapper:
    """손 관절 26개 -> INSPIRE RH56 의 6 DOF [-1(펴기) .. +1(쥐기)].

    굽힘은 '손끝-손바닥 거리 / 손 크기' 로 잰다. 관절 각도를 쓰지 않는 이유는
    런타임마다 관절 방향 규약이 달라서인데, 거리비는 그런 규약을 안 탄다.
    손 크기(손목-중지 MCP)로 나누므로 사람마다 손이 커도 그대로 쓸 수 있다.

    보정은 자동이다. 쓰는 동안 각 손가락의 최소/최대를 계속 갱신해서 그 사이를
    [-1,1] 로 편다. 시작할 때 주먹을 한 번 쥐었다 펴면 바로 제대로 잡힌다.
    """

    # 2026-09-29 실측(Quest 2 + WiVRn). 손을 가만히 둘 때의 잡음 폭은 0.1~0.4,
    # 실제로 폈다 쥐었을 때의 폭은 1.2~2.8 이었다. 그 사이를 가른다.
    MIN_RANGE = 0.8
    DECAY = 2e-6            # 프레임당. 90Hz 에서 10초에 약 0.2% 만 좁아진다
    SMOOTH = 0.5            # 출력 EMA. --hand-smooth 로 조정

    def __init__(self, smooth=None):
        if smooth is not None:
            self.SMOOTH = smooth
        self.lo = {k: None for k in INSPIRE_ORDER}
        self.hi = {k: None for k in INSPIRE_ORDER}
        self.raw = {k: 0.0 for k in INSPIRE_ORDER}
        self.last = np.full(6, -1.0)
        self.calibrated = False

    @staticmethod
    def _p(arr, i):
        p = arr[i].pose.position
        return np.array([p.x, p.y, p.z])

    def update(self, arr):
        """returns 6차원 배열 또는 None (관절이 유효하지 않거나 보정 전)"""
        need = xr.SPACE_LOCATION_POSITION_VALID_BIT
        if not all(arr[i].location_flags & need
                   for i in (J_PALM, J_WRIST, J_MCP["middle"], *J_TIP.values())):
            return None

        palm = self._p(arr, J_PALM)
        wrist = self._p(arr, J_WRIST)
        scale = np.linalg.norm(self._p(arr, J_MCP["middle"]) - wrist)
        if scale < 1e-3:
            return None

        for f in ("pinky", "ring", "middle", "index"):
            self.raw[f] = np.linalg.norm(self._p(arr, J_TIP[f]) - palm) / scale
        self.raw["thumb"] = np.linalg.norm(
            self._p(arr, J_TIP["thumb"]) - palm) / scale
        # 엄지 회전(대립): 엄지 끝이 새끼 MCP 에 가까울수록 손바닥을 가로지른 것
        self.raw["thumb_rot"] = np.linalg.norm(
            self._p(arr, J_TIP["thumb"]) - self._p(arr, J_MCP["pinky"])) / scale

        out = np.zeros(6)
        ok = True
        for i, f in enumerate(INSPIRE_ORDER):
            v = self.raw[f]
            self.lo[f] = v if self.lo[f] is None else min(self.lo[f], v)
            self.hi[f] = v if self.hi[f] is None else max(self.hi[f], v)
            span = self.hi[f] - self.lo[f]
            if span < self.MIN_RANGE:
                ok = False
                out[i] = -1.0
                continue
            # 거리가 멀면 편 것(-1), 가까우면 쥔 것(+1)
            t = (v - self.lo[f]) / span
            out[i] = float(np.clip(1.0 - 2.0 * t, -1.0, 1.0))
            # 세션 중에 의미 있게 줄어들면 안 되므로 충분히 느리게
            mid = 0.5 * (self.lo[f] + self.hi[f])
            self.lo[f] += (mid - self.lo[f]) * self.DECAY
            self.hi[f] += (mid - self.hi[f]) * self.DECAY

        if ok and not self.calibrated:
            self.calibrated = True
            print("[손] 보정 완료 — 손가락 트래킹으로 INSPIRE 를 움직입니다")
        if not ok:
            return None
        a = self.SMOOTH
        self.last = a * out + (1.0 - a) * self.last
        return self.last


class ExternalButtons:
    """외부 버튼(8BitDo Dual Super Buttons, 풋페달 등)으로 팔 추종을 켠다.

    evdev 로 /dev/input 을 직접 읽으므로 창 포커스가 없어도 동작한다.
    VR 중에는 키보드 포커스가 없으니 이게 중요하다. 그리고 손·컨트롤러와
    완전히 분리된 채널이라 클러치가 손 데이터에 섞일 여지가 원리적으로 없다.

    mode "toggle" : 한 번 눌러 켜고 다시 눌러 끈다
    mode "hold"   : 누르고 있는 동안만 추종
    """

    @staticmethod
    def find(spec):
        """'/dev/input/eventN' 이면 그대로, 아니면 이름에 그 문구가 들어간
        장치를 찾는다. event 번호는 재부팅하면 바뀔 수 있어서 이름이 안전하다."""
        from evdev import InputDevice, list_devices, ecodes
        if str(spec).startswith("/dev/"):
            return str(spec)
        want = str(spec).lower()
        best = None
        for path in sorted(list_devices()):
            try:
                d = InputDevice(path)
            except Exception:
                continue
            if want in d.name.lower():
                nkeys = len(d.capabilities().get(ecodes.EV_KEY, []))
                if best is None or nkeys > best[1]:
                    best = (path, nkeys)     # 키가 제일 많은 인터페이스
            d.close()
        if best is None:
            raise FileNotFoundError(f"'{spec}' 이름의 입력 장치를 못 찾음")
        return best[0]

    def __init__(self, path, key_left, key_right, mode="toggle"):
        from evdev import InputDevice, ecodes  # noqa

        self.ecodes = ecodes
        path = self.find(path)
        self.dev = InputDevice(path)
        self.codes = {}
        for key, side in ((key_left, "left"), (key_right, "right")):
            code = self._resolve(key)
            if code is None:
                raise ValueError(f"키 이름을 못 찾음: {key}")
            self.codes[code] = side
        self.mode = mode
        self.state = {"left": False, "right": False}
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.presses = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print(f"[버튼] {self.dev.name} ({path}) — "
              f"{'토글' if mode == 'toggle' else '홀드'} 방식")

    def _resolve(self, key):
        """'BTN_SOUTH' 같은 이름이나 숫자 코드를 받아 코드로 만든다."""
        if isinstance(key, int) or str(key).isdigit():
            return int(key)
        return getattr(self.ecodes, str(key), None)

    def _run(self):
        try:
            for ev in self.dev.read_loop():
                if self.stop.is_set():
                    return
                if ev.type != self.ecodes.EV_KEY:
                    continue
                side = self.codes.get(ev.code)
                if side is None or ev.value == 2:      # 2 = 눌린 채 반복
                    continue
                with self.lock:
                    if self.mode == "hold":
                        self.state[side] = (ev.value == 1)
                    elif ev.value == 1:
                        self.state[side] = not self.state[side]
                    if ev.value == 1:
                        self.presses += 1
                        print(f"[버튼] {side} 팔 추종 "
                              f"{'ON' if self.state[side] else 'OFF'}")
        except OSError as e:
            print(f"[버튼] 장치가 끊겼습니다: {e}")
        except Exception as e:
            print(f"[버튼] 읽기 실패: {type(e).__name__}: {e}")

    def get(self, side):
        with self.lock:
            return self.state[side]

    def close(self):
        self.stop.set()
        try:
            self.dev.close()
        except Exception:
            pass


class ControllerFingerMap:
    """오른쪽 Touch 컨트롤러 -> INSPIRE 6 DOF.

    기본 배치(finger6):
        트리거   검지        (아날로그)
        그립     중지        (아날로그)
        A        약지        (토글)
        B        새끼        (토글)
        스틱 ↕↔  엄지 굽힘/회전
        스틱클릭  팔 추종 ON/OFF (토글)

    A·B·스틱클릭이 토글인 이유: 이 셋과 스틱을 모두 엄지 하나로 쓴다.
    누르고 있는 방식이면 버튼을 잡은 동안 스틱을 못 건드린다.

    출력 순서는 dual_xarm_env 와 같다: 새끼, 약지, 중지, 검지, 엄지굽힘, 엄지회전.
    """

    def __init__(self, scheme="finger6", ab_hold=False, deadzone=STICK_DEADZONE,
                 thumb_x=-1.0, thumb_y=+1.0):
        self.scheme = scheme
        self.ab_hold = ab_hold
        self.deadzone = deadzone
        # 엄지 회전은 스틱 좌우를 뒤집은 쪽이 자연스럽다 (2026-09-29 사용 확인)
        self.thumb_x = thumb_x
        self.thumb_y = thumb_y
        self.ring = False
        self.pinky = False
        self.clutch = False
        self._prev = {"a": False, "b": False, "sc": False}

    def update(self, h):
        trig = h["trigger"] * 2.0 - 1.0
        grip = h["squeeze"] * 2.0 - 1.0
        sx, sy = h["stick"]
        sx = 0.0 if abs(sx) < self.deadzone else sx * self.thumb_x
        sy = 0.0 if abs(sy) < self.deadzone else sy * self.thumb_y

        a, b, sc = h["button"], h["button2"], h["stick_click"]
        if self.ab_hold:
            self.ring, self.pinky = a, b
        else:
            if a and not self._prev["a"]:
                self.ring = not self.ring
            if b and not self._prev["b"]:
                self.pinky = not self.pinky
        if sc and not self._prev["sc"]:
            self.clutch = not self.clutch
        self._prev = {"a": a, "b": b, "sc": sc}

        if self.scheme == "trigger4":       # 옛 배치: 네 손가락 동시
            return np.array([trig, trig, trig, trig, sy, sx])
        if self.scheme == "grip3":          # 옛 배치: 중지약지새끼 동시
            return np.array([grip, grip, grip, trig, sy, sx])
        return np.array([1.0 if self.pinky else -1.0,
                         1.0 if self.ring else -1.0,
                         grip, trig, sy, sx])


class SceneSwap:
    """MjvScene 삼중 버퍼.

    왜 필요한가: MuJoCo 3.x 의 접촉 배열은 mjData 의 arena 에 있고 매 mj_step
    마다 재할당된다. mjv_updateScene 은 ncon 을 읽고 그만큼 contact[i] 를
    훑으므로, 물리 스레드가 step 하는 동안 렌더 스레드가 updateScene 을 하면
    해제된 메모리를 읽고 죽는다 (2026-09-29 실측: 접촉 생기는 순간 segfault).

    그래서 mjData 를 만지는 일(updateScene)은 전부 물리 스레드에서 하고,
    렌더 스레드는 완성된 MjvScene 만 받아 간다. 버퍼가 셋인 이유는 렌더가
    하나를 붙잡고 있고 하나가 최신본일 때도 물리가 쓸 자리가 남아야 하기
    때문이다. 기하는 20Hz 로 갱신되지만 시점은 매 프레임 갱신되므로
    (카메라는 MjvScene 안에 있고 updateScene 없이 덮어쓸 수 있다)
    헤드 트래킹은 72Hz 그대로다."""

    def __init__(self, model, maxgeom=4000):
        self.scn = [mujoco.MjvScene(model, maxgeom=maxgeom) for _ in range(3)]
        for s_ in self.scn:
            s_.stereo = mujoco.mjtStereo.mjSTEREO_NONE
        self.lock = threading.Lock()
        self.ready = -1      # 최신 완성본
        self.held = -1       # 렌더가 잡고 있는 것
        self.published = 0

    def publish(self, model, data, opt, cam):
        with self.lock:
            idx = next(i for i in range(3) if i != self.ready and i != self.held)
        mujoco.mjv_updateScene(model, data, opt, None, cam,
                               mujoco.mjtCatBit.mjCAT_ALL, self.scn[idx])
        with self.lock:
            self.ready = idx
            self.published += 1

    def acquire(self):
        """최신 장면을 빌린다. 아직 아무것도 없으면 None."""
        with self.lock:
            if self.ready < 0:
                return None
            self.held = self.ready
            return self.scn[self.held]

    def release(self):
        with self.lock:
            self.held = -1


class PhysicsLoop:
    """env.step 을 렌더 루프 밖에서 돌리고, 팔 명령을 되먹임으로 만든다.

    왜 별도 스레드인가: env.step 한 번이 MuJoCo 25스텝(20Hz 제어 / 2ms
    타임스텝)이라 20ms 를 넘긴다. 렌더 루프 안에서 하면 72~90Hz 예산을
    통째로 먹는다.

    왜 되먹임인가: OSC_POSE 의 델타 입력은 매 스텝 '현재 말단 위치 + 델타'
    로 목표를 다시 잡는다. 그래서 한 스텝에 못 따라간 거리가 영구히 사라진다
    (2026-09-30 측정: 30cm 명령 -> 6.8cm 이동, kp=150). 여기서는 클러치를
    잡은 순간의 말단 위치를 기준으로 삼고 '기준 + 손 오프셋' 과 현재 말단의
    차이를 명령하므로, 못 따라간 만큼이 다음 명령에 남아 결국 따라잡는다.

    락은 명령 교환에만 쓴다. 렌더 스레드의 mjv_updateScene 은 이 스레드가
    대신 해 준다 (SceneSwap 참조)."""

    def __init__(self, env, control_dt, scenes=None, model=None, data=None,
                 opt=None, cam=None):
        self.env = env
        self.dt = control_dt
        self.low, self.high = env.action_spec
        self.scenes = scenes
        self.model, self.data, self.opt, self.cam = model, data, opt, cam
        self.tactile = np.zeros(max(len(env.tactile_sensor_names), 0))
        self.lock = threading.Lock()
        self.cmd = {s: dict(pos=np.zeros(3), rot=np.eye(3), ee=-1.0,
                            engaged=False, fresh=False) for s in ("left", "right")}
        self.ref = {s: None for s in ("left", "right")}   # 잡은 순간의 말단 상태
        self.stop = threading.Event()
        self.steps = 0
        self.busy = 0.0
        self.err = {"left": 0.0, "right": 0.0}            # 목표까지 남은 거리 m
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def push(self, side, pos, rot, ee, engaged, fresh):
        with self.lock:
            c = self.cmd[side]
            c["pos"], c["rot"], c["ee"], c["engaged"] = pos, rot, ee, engaged
            if fresh:
                c["fresh"] = True          # 물리가 소비할 때까지 남겨 둔다

    def _take(self):
        with self.lock:
            out = {}
            for s, c in self.cmd.items():
                out[s] = dict(pos=c["pos"].copy(), rot=c["rot"].copy(),
                              ee=c["ee"], engaged=c["engaged"], fresh=c["fresh"])
                c["fresh"] = False
            return out

    def _eef(self, idx):
        r = self.env.robots[idx]
        return (np.asarray(r._hand_pos["right"], float).copy(),
                np.asarray(r._hand_orn["right"], float).copy())

    def _arm_action(self, side, c):
        """returns 6차원 OSC 액션 [-1,1]"""
        if not c["engaged"]:
            self.ref[side] = None
            self.err[side] = 0.0
            return np.zeros(6)

        idx = 0 if side == "left" else 1
        pos, orn = self._eef(idx)
        if c["fresh"] or self.ref[side] is None:
            self.ref[side] = (pos, orn)

        ref_pos, ref_orn = self.ref[side]
        want_pos = ref_pos + c["pos"]
        d_pos = want_pos - pos
        self.err[side] = float(np.linalg.norm(d_pos))

        want_orn = c["rot"] @ ref_orn
        d_rot = R.from_matrix(want_orn @ orn.T).as_rotvec()

        return np.concatenate([d_pos / POS_OUTPUT_MAX, d_rot / ROT_OUTPUT_MAX])

    def _run(self):
        nxt = time.time()
        while not self.stop.is_set():
            now = time.time()
            if now < nxt:
                time.sleep(min(nxt - now, 0.005))
                continue
            nxt += self.dt
            if now - nxt > self.dt * 3:      # 많이 밀렸으면 따라잡기를 포기
                nxt = now + self.dt

            c = self._take()
            act = self.env.make_action(
                left_arm=self._arm_action("left", c["left"]),
                left_ee=c["left"]["ee"],
                right_arm=self._arm_action("right", c["right"]),
                right_ee=c["right"]["ee"])
            _t = time.time()
            try:
                self.env.step(np.clip(act, self.low, self.high))
            except Exception as e:
                print(f"[물리] step 실패: {type(e).__name__}: {e}")
                self.stop.set()
                return
            # mjData 를 건드리는 일은 전부 여기서. 렌더 스레드는 못 만진다.
            if self.scenes is not None:
                self.scenes.publish(self.model, self.data, self.opt, self.cam)
                try:
                    self.tactile = self.env.tactile()
                except Exception:
                    pass
            self.busy += time.time() - _t
            self.steps += 1

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2.0)


class ArmTeleop:
    """컨트롤러/손 하나 -> 팔 하나.

    프레임마다의 이동량을 그대로 명령하지 않고, 클러치를 잡은 순간부터의
    '누적 목표 오프셋' 을 들고 있는다. 물리 쪽에서 로봇 실제 위치와 비교해
    차이를 명령하므로, 한 스텝에 못 따라간 거리가 사라지지 않고 다음 명령에
    남는다.

    예전 방식(프레임 델타)은 OSC 가 매 스텝 현재 위치를 기준으로 목표를 다시
    잡기 때문에 못 따라간 만큼이 영구히 손실됐다 - 손을 30cm 움직이면 로봇은
    6.8cm 만 갔다 (2026-09-30 측정, kp=150).
    """

    def __init__(self, name, pos_scale=POS_SCALE, rot_scale=ROT_SCALE):
        self.name = name
        self.pos_scale = pos_scale
        self.rot_scale = rot_scale
        self.engaged = False
        self.fresh = False          # 방금 잡았다 -> 로봇 기준을 다시 잡아라
        self.hand_ref_pos = None
        self.hand_ref_rot = None
        self.target_pos = np.zeros(3)
        self.target_rot = np.eye(3)

    def _idle(self, ee):
        self.engaged = False
        self.target_pos = np.zeros(3)
        self.target_rot = np.eye(3)
        return self.target_pos, self.target_rot, ee, False, False

    def update(self, hand):
        """returns (목표 오프셋 m, 목표 회전 행렬, ee, 추종중, 기준재설정)"""
        ee = hand["trigger"] * 2.0 - 1.0          # 0..1 -> -1..1
        if not hand["valid"] or hand["squeeze"] < 0.5:
            return self._idle(ee)

        pos = VR2ROBOT @ hand["pos"]
        rot = VR2ROBOT @ hand["rot"] @ VR2ROBOT.T   # 상대회전이라 상사변환이 맞다

        fresh = False
        if not self.engaged:                       # 막 잡은 순간: 기준만 잡는다
            self.engaged = True
            fresh = True
            self.hand_ref_pos, self.hand_ref_rot = pos, rot
            self.target_pos = np.zeros(3)
            self.target_rot = np.eye(3)
            return self.target_pos, self.target_rot, ee, True, True

        self.target_pos = (pos - self.hand_ref_pos) * self.pos_scale
        rv = R.from_matrix(rot @ self.hand_ref_rot.T).as_rotvec() * self.rot_scale
        self.target_rot = R.from_rotvec(rv).as_matrix()
        return self.target_pos, self.target_rot, ee, True, fresh


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", type=float, default=0.6,
                   help="헤드셋 권장 해상도 대비 렌더 배율")
    p.add_argument("--hand-side", default="right", choices=["left", "right"],
                   help="INSPIRE 손이 달린 팔 (로봇 기준)")
    p.add_argument("--no-msaa", action="store_true")
    p.add_argument("--flip-y", action="store_true",
                   help="상하가 뒤집혀 보이면")
    p.add_argument("--no-object", action="store_true")
    p.add_argument("--stand-back", type=float, default=0.5,
                   help="받침대 중심에서 조작자까지 거리 m. "
                        "양수=뒤로, 음수=앞으로. 더 가까이 붙으려면 더 작은 값")
    p.add_argument("--pos-scale", type=float, default=0.6,
                   help="손 이동 대비 로봇 배율")
    p.add_argument("--rot-scale", type=float, default=ROT_SCALE,
                   help="손 회전 대비 로봇 배율")
    p.add_argument("--kp", type=float, default=500.0,
                   help="OSC 위치 게인. 높을수록 목표를 빨리 따라간다")
    p.add_argument("--start-yaw", type=float, default=0.0,
                   help="시작할 때 바라볼 방향(도). 0 = 로봇 정면(+X)")
    p.add_argument("--eye-height", type=float, default=1.45,
                   help="LOCAL 기준 공간일 때만 쓰는 눈높이 m")
    p.add_argument("--no-hand-tracking", action="store_true",
                   help="손가락 트래킹을 끄고 트리거로만 손을 쥔다")
    p.add_argument("--finger-scheme", default="finger6",
                   choices=["finger6", "trigger4", "grip3"],
                   help="오른쪽 컨트롤러 -> INSPIRE 배치 (기본 finger6)")
    p.add_argument("--buttons", default="8BitDo",
                   help="외부 버튼. 장치 이름의 일부(기본 '8BitDo') 또는 "
                        "/dev/input/eventN 경로. 끄려면 --buttons ''")
    p.add_argument("--key-gripper", default="KEY_CAPSLOCK",
                   help="그리퍼 팔 추종을 켜는 키 이름 또는 코드")
    p.add_argument("--key-hand", default="KEY_SCROLLLOCK",
                   help="INSPIRE 팔 추종을 켜는 키 이름 또는 코드")
    p.add_argument("--button-mode", default="toggle", choices=["toggle", "hold"],
                   help="외부 버튼을 토글로 쓸지 누르고 있을지")
    p.add_argument("--clutch", default="visible",
                   choices=["visible", "other-hand", "always", "fist", "grip"],
                   help="손가락 트래킹을 쓸 때 INSPIRE 팔의 클러치. "
                        "visible=손이 보이면 추종 / other-hand=반대 손 주먹 / "
                        "always=항상 추종 / fist=예전(주먹) / grip=컨트롤러 그립")
    p.add_argument("--thumb-x", type=float, default=-1.0,
                   help="엄지 회전 방향 (스틱 좌우). +1 로 주면 뒤집힌다")
    p.add_argument("--thumb-y", type=float, default=1.0,
                   help="엄지 굽힘 방향 (스틱 상하)")
    p.add_argument("--ab-hold", action="store_true",
                   help="A/B 를 토글이 아니라 누르고 있는 방식으로")
    p.add_argument("--hand-smooth", type=float, default=0.5,
                   help="손가락 출력 평활 계수 0~1 (클수록 빠르고 거칠다)")
    p.add_argument("--debug-hand", action="store_true",
                   help="손가락 raw 값과 보정 범위를 찍는다 (임계값 맞출 때)")
    p.add_argument("--debug-head", action="store_true",
                   help="머리 자세(roll/pitch/yaw)를 보고 줄에 찍는다")
    return p.parse_args()


def main():
    args = parse_args()
    xrs = XRSession(render_scale=args.scale,
                    want_hand=not args.no_hand_tracking)

    import copy
    from robosuite.controllers import load_composite_controller_config
    cc = load_composite_controller_config(controller="BASIC")
    for bp in cc.get("body_parts", {}).values():
        if bp.get("type") == "OSC_POSE":
            bp["kp"] = args.kp
            bp["kp_limits"] = [0, max(300.0, args.kp * 2)]
    env = make_env(hand_side=args.hand_side, use_object=not args.no_object,
                   controller_configs=copy.deepcopy(cc))
    env.reset()
    low, high = env.action_spec
    control_dt = 1.0 / env.control_freq
    model = getattr(env.sim.model, "_model", env.sim.model)
    data = getattr(env.sim.data, "_data", env.sim.data)
    print(f"[ENV] 로봇 {[type(r.robot_model).__name__ for r in env.robots]}  "
          f"액션 {env.action_dim}차원  촉각 {len(env.tactile_sensor_names)}개")

    scn = mujoco.MjvScene(model, maxgeom=20000)
    scn.stereo = mujoco.mjtStereo.mjSTEREO_NONE   # --inline-physics 전용
    opt = mujoco.MjvOption()
    base_cam = mujoco.MjvCamera()
    base_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    base_cam.lookat[:] = [0.3, 0.0, 0.9]
    base_cam.distance = 2.0

    renderer = None
    # INSPIRE 손이 달린 팔이 어느 쪽인지에 맞춰 조작자 손을 고른다.
    # 조작자는 로봇과 같은 방향을 보므로 로봇 오른팔 <-> 조작자 오른손.
    hand_arm = args.hand_side
    mapper = InspireHandMapper(smooth=args.hand_smooth) if xrs.trackers else None
    ee_dof_hand = env.ee_dof(hand_arm) if hasattr(env, "ee_dof") else 6
    if mapper is not None and ee_dof_hand != 6:
        print(f"[손] {hand_arm} 팔의 끝단이 {ee_dof_hand}차원이라 손가락 트래킹을 끕니다")
        mapper = None
    clutch_말 = {"visible": "손이 보이면 추종 (내리면 정지)",
                 "other-hand": "반대 손을 쥐면 추종",
                 "always": "항상 추종 (클러치 없음)",
                 "fist": "이 손을 쥐면 추종 (예전 방식)",
                 "grip": "컨트롤러 그립"}
    print(f"[손] 손가락 트래킹이 살아나면 클러치: {clutch_말[args.clutch]}")
    if ee_dof_hand == 6 and args.finger_scheme == "finger6":
        print("[손] INSPIRE: 트리거=검지  그립=중지  A=약지  B=새끼  "
              "스틱=엄지  스틱클릭=팔 추종"
              + ("  (A/B 홀드)" if args.ab_hold else "  (A/B 토글)"))
    elif ee_dof_hand == 6:
        print(f"[손] INSPIRE 배치: {args.finger_scheme} (옛 배치)")
    views, vstate = None, None
    arms = {"left": ArmTeleop("왼팔", args.pos_scale, args.rot_scale),
            "right": ArmTeleop("오른팔", args.pos_scale, args.rot_scale)}
    # 양수 = 받침대 뒤, 음수 = 받침대를 지나 앞쪽.
    # 예전엔 abs() 를 씌워서 음수를 줘도 뒤로만 갔다 (2026-09-30).
    # 옆으로 밀리는 건 recenter() 가 없앤다. 여기는 순수한 목표 위치다.
    home = np.array([-args.stand_back, 0.0,
                     0.0 if xrs.stage else args.eye_height])
    origin = home.copy()
    recentered = False
    physics = None
    scenes = None
    if True:
        scenes = SceneSwap(model, maxgeom=4000)   # 실측 ngeom 93, 넉넉함
        physics = PhysicsLoop(env, control_dt, scenes=scenes, model=model,
                              data=data, opt=opt, cam=base_cam,
)
        physics.start()
        print(f"[물리] 별도 스레드에서 {env.control_freq}Hz. "
              f"렌더 스레드는 mjData 를 만지지 않습니다 (장면 삼중 버퍼)")
    engaged = {"left": False, "right": False}
    prev_eng = dict(engaged)
    prev_btn = False
    prev_btn2 = False
    hand_follow = True      # 맨손 모드의 주 스위치 (버튼이 없으므로 기본 ON)
    other_of = {"left": "right", "right": "left"}
    buttons = None
    if args.buttons:
        # 왼팔=그리퍼, 오른팔=INSPIRE 가 기본 배치다 (--hand-side 로 바뀐다)
        k_left = args.key_gripper if hand_arm == "right" else args.key_hand
        k_right = args.key_hand if hand_arm == "right" else args.key_gripper
        try:
            buttons = ExternalButtons(args.buttons, k_left, k_right,
                                      mode=args.button_mode)
        except Exception as e:
            print(f"[버튼] 사용할 수 없습니다: {type(e).__name__}: {e}")
            print("       input_probe.py 로 경로와 키 이름을 확인하세요.")
            buttons = None
    fingers = ControllerFingerMap(scheme=args.finger_scheme,
                                  ab_hold=args.ab_hold,
                                  thumb_x=args.thumb_x, thumb_y=args.thumb_y)
    profiles_shown = False
    hand_seen = {}
    last_finger = np.full(6, -1.0)
    prev_t = time.time()
    t0 = time.time()
    frames = 0
    last_report = time.time()
    win_frames = 0
    win_steps = 0
    win_busy = 0.0
    rss0 = rss_mb()
    prof = dict(wait=0.0, scene=0.0, render=0.0, blit=0.0, input=0.0, physics=0.0)

    print("\n오른손/왼손 그립 버튼을 잡고 있는 동안 해당 팔이 따라옵니다.\n")
    try:
        while not xrs.exiting:
            _t = time.time()
            xrs.poll()
            if not xrs.running:
                time.sleep(0.05)
                continue
            fs = xr.wait_frame(xrs.session)
            xr.begin_frame(xrs.session)
            drain_gl()
            prof["wait"] += time.time() - _t

            now = time.time()
            frame_dt = now - prev_t
            prev_t = now
            frames += 1

            vstate, views = xr.locate_views(xrs.session, xr.ViewLocateInfo(
                view_configuration_type=xr.ViewConfigurationType.PRIMARY_STEREO,
                display_time=fs.predicted_display_time, space=xrs.world))

            # 방 어디에 어느 쪽을 보고 서 있든 같은 자리에서 시작하도록,
            # 시야가 유효해진 첫 프레임에 머리의 위치·방향을 상쇄한다.
            if not recentered and views:
                _f = views[0].fov
                if _f.angle_right - _f.angle_left > 0.1:
                    _p, _q = views[0].pose.position, views[0].pose.orientation
                    origin = recenter(np.array([_p.x, _p.y, _p.z]),
                                      xr_quat_to_mat(_q), home, args.start_yaw)
                    recentered = True
                    print(f"[VR] 시점 기준 설정 — 지금 보는 쪽이 "
                          f"{args.start_yaw:.0f}도 방향이 됩니다")

            # 스왑체인 크기는 실제 fov 에서 나온다. 트래킹이 아직 유효하지
            # 않으면 fov 가 0 으로 오므로 유효해질 때까지 기다린다.
            if xrs.swapchains is None:
                f = views[0].fov
                if f.angle_right - f.angle_left < 0.1:
                    xr.end_frame(xrs.session, xr.FrameEndInfo(
                        display_time=fs.predicted_display_time,
                        environment_blend_mode=xr.EnvironmentBlendMode.OPAQUE,
                        layers=[]))
                    continue
                xrs._make_swapchains(f)
                renderer = MuJoCoEyeRenderer(model, xrs.eye_w, xrs.eye_h,
                                             msaa=not args.no_msaa)

            layers = []
            keep = None
            if fs.should_render and renderer is not None:
                _t = time.time()
                if scenes is not None:
                    frame_scn = scenes.acquire()   # 물리 스레드가 만들어 둔 것
                    tac = physics.tactile
                else:
                    mujoco.mjv_updateScene(model, data, opt, None, base_cam,
                                           mujoco.mjtCatBit.mjCAT_ALL, scn)
                    frame_scn = scn
                    tac = env.tactile()
                prof["scene"] += time.time() - _t
                if frame_scn is None:
                    xr.end_frame(xrs.session, xr.FrameEndInfo(
                        display_time=fs.predicted_display_time,
                        environment_blend_mode=xr.EnvironmentBlendMode.OPAQUE,
                        layers=[]))
                    continue
                hud_top = (f"{'FOLLOW' if engaged['right'] else 'PAUSED'} R  |  "
                           f"{'FOLLOW' if engaged['left'] else 'PAUSED'} L")
                hud_bot = ("촉각 " + " ".join(f"{v:.0f}" for v in tac)) if len(tac) else ""

                def blit(i, fbo, w, h):
                    _s = time.time()
                    set_eye_camera(frame_scn, views[i], origin)
                    renderer.render(frame_scn)
                    renderer.overlay(hud_top, hud_bot)
                    prof["render"] += time.time() - _s
                    _s = time.time()
                    renderer.blit_to(fbo, w, h, flip_y=args.flip_y)
                    prof["blit"] += time.time() - _s

                try:
                    layers, keep = xrs.submit(views, blit)
                finally:
                    if scenes is not None:
                        scenes.release()

            xr.end_frame(xrs.session, xr.FrameEndInfo(
                display_time=fs.predicted_display_time,
                environment_blend_mode=xr.EnvironmentBlendMode.OPAQUE,
                layers=layers))
            del keep
            drain_gl()

            # ---- 입력 ----------------------------------------------------
            _t = time.time()
            if xrs.sync_input():
                if not profiles_shown:
                    xrs.show_profiles()
                    profiles_shown = True
                hands = {"left": xrs.hand_state(LEFT, fs.predicted_display_time),
                         "right": xrs.hand_state(RIGHT, fs.predicted_display_time)}
                for side in ("left", "right"):
                    src = hands[side]
                    # 맨손 모드에서는 클러치를 '손이 보이는가' 로 바꾼다.
                    # grasp_ext(주먹)에 걸어 두면 팔을 움직이려고 주먹을 쥐는
                    # 순간 INSPIRE 손도 같이 닫혀서 둘이 서로 싸운다
                    # (2026-09-29 실측). 손을 내려 카메라 밖으로 빼면 풀리고,
                    # 다시 들면 그 자리를 새 기준으로 잡는다.
                    if xrs.bare(side) and xrs.trackers:
                        # 맨손이면 컨트롤러 포즈 대신 손바닥 관절로 팔을 판다.
                        # squeeze(클러치)는 아래 클러치 블록이 따로 정한다.
                        hs = xrs.hand_pose_state(side, fs.predicted_display_time,
                                                 base=hands[side])
                        src = hs if hs is not None else dict(
                            src, valid=False, squeeze=0.0)
                    ctrl_fingers = None
                    live_fingers = False
                    joints_ok = False

                    if side == hand_arm and ee_dof_hand == 6:
                        # 손가락 트래킹이 '살아 있는가' 는 프로파일이 아니라
                        # 손가락이 실제로 움직이는가로 판단한다. 컨트롤러에서
                        # 합성된 손은 손가락이 굳어 있어 보정이 끝나지 않으므로
                        # mapper.calibrated 가 그대로 판별식이 된다. 덕분에
                        # '왼손 컨트롤러 + 오른손 맨손' 같은 조합도 저절로 잡힌다.
                        if mapper is not None:
                            act, arr = xrs.hand_joints(
                                side, fs.predicted_display_time)
                            joints_ok = bool(act)
                            if act:
                                f = mapper.update(arr)
                                if f is not None:
                                    last_finger = f
                            live_fingers = mapper.calibrated
                        if not live_fingers:
                            ctrl_fingers = fingers.update(hands[side])

                    # 클러치. 손가락으로 명령을 만드는 동안에는 손가락을
                    # 클러치로도 쓰면 안 된다 - 데이터가 섞인다.
                    if buttons is not None:
                        # 외부 버튼이 있으면 양팔 모두 그쪽이 우선한다.
                        # 손/컨트롤러와 분리된 채널이라 데이터가 안 섞인다.
                        src = dict(src, squeeze=1.0 if buttons.get(side) else 0.0)
                    elif side == hand_arm:
                        mode = args.clutch
                        if live_fingers and mode != "grip":
                            if mode == "visible":
                                eng_now = joints_ok
                            elif mode == "other-hand":
                                eng_now = hands[other_of[side]]["squeeze"] > 0.5
                            elif mode == "always":
                                eng_now = True
                            else:                      # "fist"
                                eng_now = hands[side]["squeeze"] > 0.5
                            src = dict(src, squeeze=1.0 if eng_now else 0.0)
                        elif ctrl_fingers is not None and args.finger_scheme == "finger6":
                            # 컨트롤러 배치에서는 그립이 중지에 쓰이므로
                            # 클러치는 스틱 클릭 토글이 준다
                            src = dict(src, squeeze=1.0 if fingers.clutch else 0.0)

                    tpos, trot, ee, eng, fresh = arms[side].update(src)

                    if side == hand_arm and ee_dof_hand == 6:
                        ee = last_finger if live_fingers else ctrl_fingers
                    if physics is not None:
                        physics.push(side, tpos, trot, ee, eng, fresh)
                    engaged[side] = eng
                    if eng != prev_eng[side]:
                        print(f"[VR] {arms[side].name} 추종 {'ON' if eng else 'OFF'}")
                        prev_eng[side] = eng

                # 왼손 스틱 = 시점 이동, X 버튼 = 초기화
                lh = hands["left"]
                # 오른손 A/B 는 약지·새끼가 가져갔다. 시점 초기화는 왼손 Y.
                b1 = lh["button"] if xrs.bare(hand_arm) else False
                b2 = lh["button2"]
                if b1 and not prev_btn:
                    hand_follow = not hand_follow
                    print(f"[VR] 맨손 팔 추종 {'ON' if hand_follow else 'OFF'}")
                if b2 and not prev_btn2:
                    if views:
                        _p, _q = views[0].pose.position, views[0].pose.orientation
                        origin = recenter(np.array([_p.x, _p.y, _p.z]),
                                          xr_quat_to_mat(_q), home, args.start_yaw)
                        for _a in arms.values():
                            _a.engaged = False   # 기준 행렬이 바뀌었으니 다시 잡게
                    else:
                        origin = home.copy()
                    print("[VR] 시점 초기화 — 지금 보는 쪽이 로봇 정면이 됩니다")
                prev_btn, prev_btn2 = b1, b2

                sx, sy = lh["stick"]
                sx = 0.0 if abs(sx) < STICK_DEADZONE else sx
                sy = 0.0 if abs(sy) < STICK_DEADZONE else sy
                if (sx or sy) and len(views):
                    axes = VR2ROBOT @ xr_quat_to_mat(views[0].pose.orientation)
                    fwd = -axes[:, 2]; fwd[2] = 0.0
                    rgt = axes[:, 0].copy(); rgt[2] = 0.0
                    n1, n2 = np.linalg.norm(fwd), np.linalg.norm(rgt)
                    if n1 > 1e-6 and n2 > 1e-6:
                        origin += (fwd / n1 * sy + rgt / n2 * sx) * MOVE_SPEED * frame_dt
                        # 셀에서 너무 멀리 못 가게 (길을 잃으면 왼손 X 로 복귀)
                        origin[0] = float(np.clip(origin[0], -4.0, 3.0))
                        origin[1] = float(np.clip(origin[1], -3.0, 3.0))
            prof["input"] += time.time() - _t

            # ---- 물리는 전용 스레드에서 (--inline-physics 는 폐지) -----
            _t = time.time()
            prof["physics"] += time.time() - _t

            # ---- 보고 (누적 평균이 아니라 최근 3초치) -----------------------
            win_frames += 1
            now2 = time.time()
            win = now2 - last_report
            if win > 3.0:
                n = max(win_frames, 1)
                parts = "  ".join(f"{k}={v/n*1000:.1f}" for k, v in prof.items())
                if physics is not None:
                    ds = physics.steps - win_steps
                    db = physics.busy - win_busy
                    win_steps, win_busy = physics.steps, physics.busy
                    ph = (f"  물리 {ds/win:.1f}Hz/{env.control_freq} "
                          f"step={db/max(ds,1)*1000:.0f}ms "
                          f"장면{scenes.published if scenes else 0} "
                          f"오차 L{physics.err['left']*100:.0f}/"
                          f"R{physics.err['right']*100:.0f}cm")
                # 양손이 실제로 잡히고 있는지. 왼손만 안 따라올 때 여기를 본다.
                if xrs.trackers:
                    parts = []
                    for sd, nm in (("left", "왼손"), ("right", "오른손")):
                        act, pos, nj = xrs.hand_probe(
                            sd, fs.predicted_display_time)
                        if pos is None or not act:
                            parts.append(f"{nm}=꺼짐")
                            hand_seen[sd] = None
                            continue
                        prev = hand_seen.get(sd)
                        mv = np.linalg.norm(pos - prev) if prev is not None else 0.0
                        hand_seen[sd] = pos
                        parts.append(f"{nm}: 관절{nj}/26 "
                                     f"손바닥({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f}) "
                                     f"이동{mv*100:.0f}cm")
                    print("     " + "   ".join(parts))
                if args.debug_hand and mapper is not None:
                    print("     손 raw " + "  ".join(
                        f"{f}={mapper.raw[f]:.3f}"
                        f"[{(mapper.lo[f] or 0):.3f}~{(mapper.hi[f] or 0):.3f}]"
                        for f in INSPIRE_ORDER))
                if mapper is not None:
                    if mapper.calibrated:
                        ph += "  손[" + " ".join(f"{v:+.1f}" for v in mapper.last) + "]"
                    else:
                        ph += "  손:보정중(주먹 한번 쥐었다 펴세요)"
                else:
                    ph = ""
                print(f"[VR] {n/win:.1f} fps  (ms) {parts}{ph}  "
                      f"RSS={rss_mb():.0f}MB(+{rss_mb()-rss0:.0f})  "
                      f"origin=({origin[0]:.1f},{origin[1]:.1f},{origin[2]:.1f})")
                if args.debug_head and views:
                    q = views[0].pose.orientation
                    pp = views[0].pose.position
                    # 로봇 좌표계에서의 머리 자세. 고개를 끄덕이면 pitch,
                    # 갸웃하면 roll, 돌리면 yaw 가 움직여야 한다.
                    rpy = R.from_matrix(
                        VR2ROBOT @ xr_quat_to_mat(q) @ VR2ROBOT.T
                    ).as_euler("xyz", degrees=True)
                    tracked = int(vstate.view_state_flags)
                    print(f"     머리 roll={rpy[0]:+6.1f} pitch={rpy[1]:+6.1f} "
                          f"yaw={rpy[2]:+6.1f}   "
                          f"쿼터니언 x={q.x:+.3f} y={q.y:+.3f} z={q.z:+.3f} w={q.w:+.3f}  "
                          f"머리위치=({pp.x:+.2f},{pp.y:+.2f},{pp.z:+.2f})  "
                          f"flags=0x{tracked:x}")
                for k in prof:
                    prof[k] = 0.0
                win_frames = 0
                last_report = now2

    except KeyboardInterrupt:
        print("\n종료합니다.")
    finally:
        el = time.time() - t0
        print(f"프레임 {frames} / {el:.1f}초 = {frames/max(el,1e-6):.1f} fps")
        if buttons is not None:
            buttons.close()
        if physics is not None:
            physics.close()
        xrs.close()
        try: env.close()
        except Exception: pass


if __name__ == "__main__":
    main()