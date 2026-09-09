"""Frozen rl_sar Go2 policy adapter. No Isaac imports; all joint maps use names.

The observation/action contract follows rl_sar commit
376d42c9b128f963ab08579762d5a216a976ce39 (Apache-2.0).
"""
from pathlib import Path

import torch
import yaml


class ExternalGo2Policy:
    def __init__(self, source: Path, profile: str, joint_names: list[str], device: str, model_path: Path | None = None):
        self.cfg = yaml.safe_load((source / f"policy/go2/{profile}/config.yaml").read_text())[f"go2/{profile}"]
        base = yaml.safe_load((source / "policy/go2/base.yaml").read_text())["go2"]
        self.policy_names = [base["joint_names"][i] for i in self.cfg["joint_mapping"]]
        self.indices = [joint_names.index(n) for n in self.policy_names]
        self.device = device
        self.default = self.tensor(self.cfg["default_dof_pos"])
        self.scale = self.tensor(self.cfg["action_scale"])
        self.model_path = model_path or source / f"policy/go2/{profile}" / self.cfg["model_name"]
        self.model = torch.jit.load(str(self.model_path), map_location=device).eval()
        self.reset()

    def tensor(self, value):
        return torch.tensor(value, dtype=torch.float32, device=self.device)

    def reset(self):
        self.previous = self.tensor([[0.] * 12])
        self.history = None

    def observation(self, q, dq, angular_velocity, gravity, command):
        c = self.cfg
        terms = {
            "commands": command * self.tensor(c["commands_scale"]),
            "ang_vel": angular_velocity * c["ang_vel_scale"],
            "gravity_vec": gravity,
            "dof_pos": (q[:, self.indices] - self.default) * c["dof_pos_scale"],
            "dof_vel": dq[:, self.indices] * c["dof_vel_scale"],
            "actions": self.previous,
        }
        obs = torch.cat([terms[n] for n in c["observations"]], -1).clamp(-c["clip_obs"], c["clip_obs"])
        ids = c["observations_history"]
        if ids:
            # Upstream buffer is initialized with zero past frames; newest first.
            if self.history is None:
                self.history = torch.zeros((1, max(ids) + 1, obs.shape[-1]), device=self.device)
            self.history = torch.cat((obs[:, None], self.history[:, :-1]), dim=1)
            obs = self.history[:, ids].flatten(1)
        return obs

    def act(self, robot, command):
        d = robot.data
        obs = self.observation(d.joint_pos, d.joint_vel, d.root_ang_vel_b, d.projected_gravity_b, command)
        raw = self.model(obs)
        if not torch.isfinite(raw).all():
            raise RuntimeError("Nonfinite policy output")
        self.previous = torch.maximum(torch.minimum(raw, self.tensor(self.cfg["clip_actions_upper"])),
                                      self.tensor(self.cfg["clip_actions_lower"]))
        target = d.default_joint_pos.clone()
        target[:, self.indices] = self.default + self.previous * self.scale
        return target, obs


def configure_external_go2_environment(cfg, source: Path, profile: str, actuator_type=None):
    """Apply the same upstream PD/action/time contract in both composition roots."""
    if actuator_type is None:
        from isaaclab.actuators import IdealPDActuatorCfg
        actuator_type = IdealPDActuatorCfg
    spec = yaml.safe_load((source / f"policy/go2/{profile}/config.yaml").read_text())[f"go2/{profile}"]
    base = yaml.safe_load((source / "policy/go2/base.yaml").read_text())["go2"]
    names = [base["joint_names"][i] for i in spec["joint_mapping"]]
    cfg.scene.robot.init_state.joint_pos = dict(zip(names, spec["default_dof_pos"]))
    cfg.scene.robot.actuators = {"legs": actuator_type(
        joint_names_expr=[".*"], stiffness=dict(zip(names, spec["rl_kp"])),
        damping=dict(zip(names, spec["rl_kd"])), effort_limit=dict(zip(names, spec["torque_limits"])),
        velocity_limit_sim=30.)}
    cfg.actions.joint_pos.joint_names = [".*"]
    cfg.actions.joint_pos.scale = 1.
    cfg.actions.joint_pos.use_default_offset = False
    cfg.sim.dt, cfg.decimation = .005, 4
