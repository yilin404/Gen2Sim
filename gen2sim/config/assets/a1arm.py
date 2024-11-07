import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg

##
# Configuration
##
A1ARM_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/A1Arm",
    spawn=sim_utils.UsdFileCfg(
        usd_path="data/robot_description/a1arm_description/usd/a1arm/a1arm.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0,
            fix_root_link=True,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "arm_joint1": 0.0,
            "arm_joint2": 0.0,
            "arm_joint3": 0.0,
            "arm_joint4": 0.0,
            "arm_joint5": 0.0,
            "arm_joint6": 0.0,
            "gripper1_axis": 0.03,
            "gripper2_axis": 0.03,
        },
    ),
    actuators={
        "a1arm_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["arm_joint[1-2]"],
            effort_limit=40,
            velocity_limit=20.944,
            stiffness=80.0,
            damping=4.0,
        ),
        "a1arm_elbow": ImplicitActuatorCfg(
            joint_names_expr=["arm_joint3"],
            effort_limit=7,
            velocity_limit=25.133,
            stiffness=80.0,
            damping=4.0,
        ),
        "a1arm_forearm": ImplicitActuatorCfg(
            joint_names_expr=["arm_joint[4-6]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        "a1arm_hand": ImplicitActuatorCfg(
            joint_names_expr=["gripper[1-2]_axis"],
            effort_limit=200.0,
            velocity_limit=0.25,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

A1ARM_RGM3_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/A1Arm_RGM3",
    spawn=sim_utils.UsdFileCfg(
        usd_path="data/robot_description/a1arm_description/usd/a1arm_rgm3/a1arm_rgm3.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0,
            fix_root_link=True,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "root_x_axis_joint": 0.0,
            "root_y_axis_joint": 0.0,
            "root_z_rotation_joint": 1.0,
            "arm_joint1": 0.0,
            "arm_joint2": 0.0,
            "arm_joint3": 0.0,
            "arm_joint4": 0.0,
            "arm_joint5": 0.0,
            "arm_joint6": 0.0,
            "gripper1_axis": 0.03,
            "gripper2_axis": 0.03,
        },
        pos=(0., 0., 0.3301)
    ),
    actuators={
        "a1arm_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["arm_joint[1-2]"],
            effort_limit=40,
            velocity_limit=20.944,
            stiffness=80.0,
            damping=4.0,
        ),
        "a1arm_elbow": ImplicitActuatorCfg(
            joint_names_expr=["arm_joint3"],
            effort_limit=7,
            velocity_limit=25.133,
            stiffness=80.0,
            damping=4.0,
        ),
        "a1arm_forearm": ImplicitActuatorCfg(
            joint_names_expr=["arm_joint[4-6]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        "a1arm_hand": ImplicitActuatorCfg(
            joint_names_expr=["gripper[1-2]_axis"],
            effort_limit=200.0,
            velocity_limit=0.25,
            stiffness=2e3,
            damping=1e2,
        ),
        "root_axis": ImplicitActuatorCfg(
            joint_names_expr=["root_x_axis_joint", "root_y_axis_joint"],
            effort_limit=2e6,
            velocity_limit=2.5,
            stiffness=0,
            damping=1e8,
        ),
        "root_rotation": ImplicitActuatorCfg(
            joint_names_expr=["root_z_rotation_joint"],
            effort_limit=2e6,
            velocity_limit=0.5,
            stiffness=0,
            damping=1e8,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)