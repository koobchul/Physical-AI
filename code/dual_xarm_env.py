"""
실제 작업 셀을 재현한 robosuite 환경.

    받침대  80(좌우) x 60(전후) x 65(높이) cm
    xArm7 x2  좌우 끝에서 10cm, 전후 30cm(=중앙) 지점, 받침대 윗면에 장착
        로봇을 마주 볼 때 왼쪽 (= 로봇 기준 오른쪽, 월드 -Y)
            : INSPIRE RH56 (오른손 모델, 6 DOF) + 손끝/손바닥 촉각 센서
        로봇을 마주 볼 때 오른쪽 (= 로봇 기준 왼쪽, 월드 +Y)
            : xArm 그리퍼
    작업 테이블: 받침대 앞

좌표계 (robosuite 규약)
    +X 전방(테이블 쪽), +Y 왼쪽, +Z 위. 원점은 받침대 바닥 중앙.

주의: 코드 안의 left/right 는 전부 '로봇 자신' 기준이다.
    로봇 기준 왼쪽  = 받침대 뒤에서 테이블을 볼 때 왼쪽 = 월드 +Y
    로봇 기준 오른쪽 = 로봇을 마주 볼 때 왼쪽        = 월드 -Y
VR 조작자는 받침대 뒤에서 로봇과 같은 방향을 보므로, 조작자의 오른손이
로봇 오른팔(INSPIRE)에, 왼손이 로봇 왼팔(그리퍼)에 대응한다.

로봇 순서는 항상 [로봇 왼쪽, 로봇 오른쪽] 이다.
액션은 인덱스로 다루지 말고 env.make_action(...) 을 쓸 것.
    기본 배치(hand_side="right") 레이아웃, 19차원:
    [0:6]   왼팔(그리퍼) OSC 델타 (x y z, 회전 axis-angle) [-1, 1]
    [6]     xArm 그리퍼 (+1 닫기)
    [7:13]  오른팔(INSPIRE) OSC 델타
    [13:19] INSPIRE 손: 새끼, 약지, 중지, 검지, 엄지굽힘, 엄지회전 (+1 쥐기)
"""

import xml.etree.ElementTree as ET
from functools import lru_cache

import mujoco
import numpy as np

import robosuite
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.robots import XArm7
from robosuite.models.tasks import ManipulationTask
from robosuite.robots import ROBOT_CLASS_MAPPING
from robosuite.robots.fixed_base_robot import FixedBaseRobot
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler


# --------------------------------------------------------------------------
# 셀 치수 (미터)
# --------------------------------------------------------------------------
PEDESTAL_W = 0.80     # 좌우 (Y)
PEDESTAL_D = 0.60     # 전후 (X)
PEDESTAL_H = 0.65     # 높이 (Z)

ARM_EDGE_INSET = 0.10         # 좌우 끝에서 팔 중심까지
ARM_DEPTH_FROM_FRONT = 0.30   # 받침대 앞면에서 팔 중심까지 (60cm 중 30cm = 중앙)

# 팔 베이스 위치 (받침대 윗면)
ARM_Y = PEDESTAL_W / 2 - ARM_EDGE_INSET            # +-0.30
ARM_X = PEDESTAL_D / 2 - ARM_DEPTH_FROM_FRONT      # 0.00
LEFT_BASE = np.array([ARM_X, +ARM_Y, PEDESTAL_H])
RIGHT_BASE = np.array([ARM_X, -ARM_Y, PEDESTAL_H])

# PandaDexRH 와 같은 값. Panda 와 XArm7 은 같은 그리퍼 장착 규약을 쓴다.
INSPIRE_MOUNT_QUAT = [-0.5, 0.5, 0.5, -0.5]


# --------------------------------------------------------------------------
# xArm7 + INSPIRE 오른손
# --------------------------------------------------------------------------
class XArm7DexRH(XArm7):
    """xArm7 에 INSPIRE RH56 오른손을 단 변형. 메타클래스가 자동 등록한다."""

    @property
    def default_gripper(self):
        return {"right": "InspireRightHand"}

    @property
    def gripper_mount_pos_offset(self):
        return {"right": [0.0, 0.0, 0.0]}

    @property
    def gripper_mount_quat_offset(self):
        return {"right": INSPIRE_MOUNT_QUAT}


ROBOT_CLASS_MAPPING["XArm7DexRH"] = FixedBaseRobot


# --------------------------------------------------------------------------
# 촉각 센서
# --------------------------------------------------------------------------
# 실제 RH56E2 의 촉각 배열(손가락마다 수십~수백 taxel)은 MuJoCo 가 제공하지
# 않는다. 여기서는 각 손끝/손바닥 충돌 메시를 감싸는 영역에 MuJoCo touch 센서를
# 두어 '해당 부위에 걸린 법선 접촉력의 합(N)' 하나씩을 얻는다.
TACTILE_LINKS = [
    ("thumb", "r_thumb_distal"),
    ("index", "r_index_distal"),
    ("middle", "r_middle_distal"),
    ("ring", "r_ring_distal"),
    ("pinky", "r_pinky_distal"),
    ("palm", "r_palm"),
]
TACTILE_PAD = 1.15   # 접촉점은 메시 표면에 생기므로 영역을 살짝 키운다


@lru_cache(maxsize=1)
def _measure_tactile_boxes():
    """손 모델을 단독 컴파일해 각 부위 충돌 메시의 경계 상자를 몸체 좌표로 구한다.
    메시라서 XML 에 크기가 없으므로 실측이 유일하게 정확한 방법이다."""
    from robosuite.models.grippers import InspireRightHand

    m = mujoco.MjModel.from_xml_path(InspireRightHand(idn=0).file)
    boxes = {}
    for label, body in TACTILE_LINKS:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
        if bid < 0:
            raise RuntimeError(f"손 모델에 {body} 가 없습니다")
        gids = [g for g in range(m.ngeom)
                if m.geom_bodyid[g] == bid and m.geom_contype[g] != 0]
        if not gids:
            raise RuntimeError(f"{body} 에 충돌 형상이 없습니다")
        g = gids[0]
        c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, m.geom_quat[g])
        pos = m.geom_pos[g] + rot.reshape(3, 3) @ c
        boxes[body] = (label, pos, m.geom_quat[g].copy(), h * TACTILE_PAD)
    return boxes


def _fmt(v):
    return " ".join(f"{x:.6g}" for x in v)


def add_tactile_sensors(model_root, sensor_elem, prefix="tactile_"):
    """병합된 모델 XML 에 촉각 site + touch 센서를 추가한다.
    반환: 센서 이름 리스트 (TACTILE_LINKS 순서)."""
    boxes = _measure_tactile_boxes()
    names = []
    for label, body in TACTILE_LINKS:
        target = None
        for b in model_root.iter("body"):
            n = b.get("name", "")
            if n.endswith(body):
                target = b
                break
        if target is None:
            raise RuntimeError(f"병합된 모델에서 {body} 를 찾지 못했습니다")
        _, pos, quat, half = boxes[body]
        site_name = f"{prefix}{label}_site"
        ET.SubElement(target, "site", {
            "name": site_name, "type": "box",
            "pos": _fmt(pos), "quat": _fmt(quat), "size": _fmt(half),
            "rgba": "1 0.3 0.1 0.25", "group": "3",
        })
        sensor_name = f"{prefix}{label}"
        ET.SubElement(sensor_elem, "touch", {"name": sensor_name, "site": site_name})
        names.append(sensor_name)
    return names


# --------------------------------------------------------------------------
# 환경
# --------------------------------------------------------------------------
class DualXArmPedestal(ManipulationEnv):
    """
    Args:
        table_full_size: (전후 X, 좌우 Y, 상판 두께) m
        table_top_height: 바닥에서 테이블 윗면까지 m
        table_gap: 받침대 앞면과 테이블 사이 간격 m
    """

    def __init__(
        self,
        hand_side="right",                  # INSPIRE 손을 달 팔 (로봇 기준)
        table_full_size=(0.60, 1.20, 0.05),
        table_top_height=0.65,
        table_gap=0.05,
        use_object=True,
        **kwargs,
    ):
        self.table_full_size = np.array(table_full_size)
        self.table_top_height = float(table_top_height)
        self.table_gap = float(table_gap)
        self.use_object = use_object

        table_center_x = PEDESTAL_D / 2 + table_gap + table_full_size[0] / 2
        self.table_offset = np.array([table_center_x, 0.0, table_top_height])

        self.tactile_sensor_names = []
        self._tactile_adr = None

        if hand_side not in ("left", "right"):
            raise ValueError("hand_side 는 'left' 또는 'right'")
        self.hand_side = hand_side
        # 로봇 순서는 항상 [왼쪽, 오른쪽]
        robots = (["XArm7DexRH", "XArm7"] if hand_side == "left"
                  else ["XArm7", "XArm7DexRH"])

        kwargs.setdefault("base_types", "NullMount")      # 기본 원기둥 받침대 제거
        kwargs.setdefault("env_configuration", "default")
        super().__init__(robots=robots, **kwargs)

    # -- 모델 ------------------------------------------------------------
    def _load_model(self):
        super()._load_model()

        # 팔 배치: 받침대 윗면, 둘 다 +X(테이블) 방향
        for robot, pos in zip(self.robots, (LEFT_BASE, RIGHT_BASE)):
            robot.robot_model.set_base_xpos(pos)
            robot.robot_model.set_base_ori([0.0, 0.0, 0.0])

        arena = TableArena(
            table_full_size=self.table_full_size,
            table_offset=self.table_offset,
        )
        arena.set_origin([0, 0, 0])
        self._add_pedestal(arena)
        self._add_cameras(arena)

        objects = []
        if self.use_object:
            self.cube = BoxObject(
                name="cube",
                size_min=[0.025, 0.025, 0.025],
                size_max=[0.025, 0.025, 0.025],
                rgba=[0.9, 0.2, 0.2, 1],
            )
            objects.append(self.cube)
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.cube,
                x_range=[-0.10, 0.10],
                y_range=[-0.15, 0.15],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[r.robot_model for r in self.robots],
            mujoco_objects=objects,
        )

        # INSPIRE 손이 달린 팔에만 촉각 센서 (손은 하나뿐)
        if any(type(r.robot_model).__name__ == "XArm7DexRH" for r in self.robots):
            self.tactile_sensor_names = add_tactile_sensors(
                self.model.root, self.model.sensor)

    def _add_pedestal(self, arena):
        body = ET.SubElement(arena.worldbody, "body", {
            "name": "pedestal", "pos": _fmt([0, 0, PEDESTAL_H / 2])})
        half = [PEDESTAL_D / 2, PEDESTAL_W / 2, PEDESTAL_H / 2]
        ET.SubElement(body, "geom", {
            "name": "pedestal_col", "type": "box", "size": _fmt(half),
            "rgba": "0.35 0.37 0.40 1", "friction": "1 0.005 0.0001",
            "group": "0",
        })
        ET.SubElement(body, "geom", {
            "name": "pedestal_vis", "type": "box", "size": _fmt(half),
            "rgba": "0.35 0.37 0.40 1", "contype": "0", "conaffinity": "0",
            "group": "1",
        })

    def _add_cameras(self, arena):
        # 셀 전체를 보는 정면/측면 카메라 (확인·녹화용)
        ET.SubElement(arena.worldbody, "camera", {
            "name": "cellview", "mode": "fixed",
            "pos": "2.2 0 1.7", "xyaxes": "0 1 0 -0.55 0 0.83"})
        ET.SubElement(arena.worldbody, "camera", {
            "name": "sideview_left", "mode": "fixed",
            "pos": "0.5 1.9 1.4", "xyaxes": "-1 0 0 0 -0.55 0.83"})

    # -- 참조 / 관측 -------------------------------------------------------
    def _setup_references(self):
        super()._setup_references()
        if self.use_object:
            self.cube_body_id = self.sim.model.body_name2id(self.cube.root_body)
        self._tactile_adr = []
        for name in self.tactile_sensor_names:
            sid = self.sim.model.sensor_name2id(name)
            self._tactile_adr.append(self.sim.model.sensor_adr[sid])

    def tactile(self):
        """INSPIRE 촉각값 (N). 순서: thumb, index, middle, ring, pinky, palm"""
        if not self._tactile_adr:
            return np.zeros(0)
        return np.array([self.sim.data.sensordata[a] for a in self._tactile_adr])

    def _setup_observables(self):
        observables = super()._setup_observables()
        if self.tactile_sensor_names:
            @sensor(modality="tactile")
            def left_hand_tactile(obs_cache):
                return self.tactile()
            observables["left_hand_tactile"] = Observable(
                name="left_hand_tactile", sensor=left_hand_tactile,
                sampling_rate=self.control_freq)
        if self.use_object:
            @sensor(modality="object")
            def cube_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cube_body_id])
            observables["cube_pos"] = Observable(
                name="cube_pos", sensor=cube_pos, sampling_rate=self.control_freq)
        return observables

    def _reset_internal(self):
        super()._reset_internal()
        if self.use_object and not self.deterministic_reset:
            for pos, quat, obj in self.placement_initializer.sample().values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0], np.concatenate([np.array(pos), np.array(quat)]))

    # -- 액션 도우미 -------------------------------------------------------
    def ee_dof(self, side):
        """side 팔 끝단(손/그리퍼)의 명령 차원. 손=6, 그리퍼=1"""
        return self.robots[0 if side == "left" else 1].gripper["right"].dof

    def make_action(self, left_arm=None, left_ee=None,
                    right_arm=None, right_ee=None):
        """이름으로 액션을 만든다. 인덱스를 직접 다루지 말 것.
        arm: 6차원 OSC 델타 [-1,1] (None=정지)
        ee : 손이면 6차원, 그리퍼면 1차원. 스칼라를 주면 전부 같은 값.
             +1 쥐기 / -1 펴기 (None=펴기)"""
        parts = []
        for arm, ee, robot in ((left_arm, left_ee, self.robots[0]),
                               (right_arm, right_ee, self.robots[1])):
            dof = robot.gripper["right"].dof
            arm = np.zeros(6) if arm is None else np.asarray(arm, float)
            ee = -1.0 if ee is None else ee
            ee = np.broadcast_to(np.asarray(ee, float), (dof,)).copy()
            parts += [arm, ee]
        return np.concatenate(parts)

    # -- 과제 (자유 조작용 더미) ------------------------------------------
    def reward(self, action=None):
        return 0.0

    def _check_success(self):
        return False


def make_env(**kwargs):
    """suite.make 없이 직접 생성. BASIC 컨트롤러가 기본."""
    from robosuite.controllers import load_composite_controller_config
    kwargs.setdefault("controller_configs",
                      load_composite_controller_config(controller="BASIC"))
    kwargs.setdefault("has_renderer", False)
    kwargs.setdefault("has_offscreen_renderer", False)
    kwargs.setdefault("use_camera_obs", False)
    kwargs.setdefault("control_freq", 20)
    kwargs.setdefault("horizon", 100000)
    kwargs.setdefault("ignore_done", True)
    return DualXArmPedestal(**kwargs)
