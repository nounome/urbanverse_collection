"""Reusable Isaac Sim 4.5 official People, CharacterManager and NavMesh adapter.

This module intentionally does not launch ``SimulationApp``.  The owning
composition root must start Kit with the companion schema-preload experience,
enable the already-installed IRA extension after startup, create the stage and
author a valid NavMesh before constructing :class:`OfficialPeopleRuntime`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import site
from typing import Any, Iterable, Sequence

import numpy as np


RUNTIME_EXTENSIONS = (
    "omni.anim.people",
    "omni.anim.graph.core",
    "omni.anim.navigation.core",
    "omni.anim.navigation.recast",
    "isaacsim.replicator.agent.core",
)


@dataclass(frozen=True)
class OfficialCharacterSpec:
    """One official People character and its ordered high-level route targets."""

    name: str
    asset: Path
    start_xyz: tuple[float, float, float]
    waypoints_xyz: tuple[tuple[float, float, float], ...]
    initial_yaw_deg: float = 0.0
    final_yaw_deg: float | None = None

    def __post_init__(self) -> None:
        if not self.name or any(char.isspace() for char in self.name):
            raise ValueError("character name must be nonempty and contain no whitespace")
        if len(self.start_xyz) != 3:
            raise ValueError("start_xyz must contain x, y and z")
        if not self.waypoints_xyz or any(len(point) != 3 for point in self.waypoints_xyz):
            raise ValueError("waypoints_xyz must contain at least one xyz point")


def local_isaac_extension_folders() -> list[str]:
    """Return installed extension roots needed by the pip Isaac Sim app."""

    site_packages = Path(site.getsitepackages()[0])
    candidates = (
        "isaacsim/exts",
        "isaacsim/extscache",
        "isaacsim/extsPhysics",
        "isaacsim/extsUser",
        "isaacsim/extsDeprecated",
        "exts",
        "extscache",
        "extsPhysics",
        "extsDeprecated",
    )
    return [str(site_packages / item) for item in candidates if (site_packages / item).is_dir()]


def simulation_app_launch_config(
    gpu_index: int,
    *,
    width: int = 640,
    height: int = 480,
) -> dict[str, Any]:
    """Build the headless launch settings proven to avoid the 4.5 viewport wait."""

    extra_args = [
        "--/renderer/multiGpu/enabled=false",
        "--/app/window/hideUi=1",
        "--/renderer/gpuEnumeration/glInterop/enabled=false",
    ]
    for folder in local_isaac_extension_folders():
        extra_args.extend(("--ext-folder", folder))
    return {
        "headless": True,
        "renderer": "RayTracedLighting",
        "width": int(width),
        "height": int(height),
        "active_gpu": int(gpu_index),
        "physics_gpu": int(gpu_index),
        "multi_gpu": False,
        "max_gpu_count": 1,
        # IRA enables viewport.utility in headless mode.  In 4.5 the automatic
        # new-stage path can then wait forever for a nonexistent UI viewport.
        "create_new_stage": False,
        "extra_args": extra_args,
    }


def enable_official_people_runtime(app: Any, *, settle_updates: int = 5) -> dict[str, bool]:
    """Enable the installed IRA closure after schema/core preload startup."""

    import omni.kit.app

    manager = omni.kit.app.get_app().get_extension_manager()
    manager.set_extension_enabled_immediate("isaacsim.replicator.agent.core", True)
    for _ in range(settle_updates):
        app.update()
    enabled = {name: bool(manager.is_extension_enabled(name)) for name in RUNTIME_EXTENSIONS}
    if not all(enabled.values()):
        raise RuntimeError(f"official People runtime extensions are incomplete: {enabled}")
    return enabled


def goto_command(spec: OfficialCharacterSpec) -> str:
    """Serialize an official People multi-point ``GoTo`` command."""

    coordinates = " ".join(
        f"{float(value):.6f}" for point in spec.waypoints_xyz for value in point
    )
    rotation = "_" if spec.final_yaw_deg is None else f"{float(spec.final_yaw_deg):.6f}"
    return f"{spec.name} GoTo {coordinates} {rotation}"


def write_goto_commands(path: Path, specs: Sequence[OfficialCharacterSpec]) -> Path:
    """Write a deterministic command file consumed by ``CharacterBehavior``."""

    if not specs:
        raise ValueError("at least one character is required")
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("character names must be unique")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(goto_command(spec) for spec in specs) + "\n", encoding="utf-8")
    return path


def runtime_character_position(character: Any) -> np.ndarray:
    """Read authoritative live root position from Animation Graph/Fabric."""

    import carb

    position = carb.Float3(0.0, 0.0, 0.0)
    rotation = carb.Float4(0.0, 0.0, 0.0, 0.0)
    # Isaac 4.5 mutates the output objects and commonly returns None.  NVIDIA's
    # own bundled tests ignore that return; only an explicit False is failure.
    result = character.get_world_transform(position, rotation)
    if result is False:
        raise RuntimeError("Animation Graph get_world_transform returned false")
    return np.asarray((position[0], position[1], position[2]), dtype=np.float64)


class OfficialPeopleRuntime:
    """Set up and drive official People characters on an authored NavMesh."""

    def __init__(
        self,
        app: Any,
        stage: Any,
        assets_root: Path,
        *,
        characters_parent: str = "/World/Characters",
    ) -> None:
        self.app = app
        self.stage = stage
        self.assets_root = Path(assets_root).resolve()
        self.characters_parent = characters_parent
        self.specs: tuple[OfficialCharacterSpec, ...] = ()
        self.roots: list[Any] = []
        self.skelroots: list[Any] = []
        self.characters: list[Any] = []
        self.behavior_scripts: dict[str, Any] = {}
        self.behavior_time_s = 0.0
        self.avoidance_radius_m: float | None = None
        self.navmesh = None
        self.timeline = None
        self.navmesh_explicit_start_return: bool | None = None

    def load_characters(
        self,
        specs: Iterable[OfficialCharacterSpec],
        command_path: Path,
    ) -> dict[str, Any]:
        """Load official rigs and bind their command/biped behavior.

        This is the pre-bake half of :meth:`setup`.  It is kept public so the
        Scene09 NavMesh probe can load characters while the approved helper
        controls which geometry Recast sees, then call :meth:`wait_for_navmesh`
        separately.
        """

        import carb
        import omni.kit.commands
        import NavSchema
        from isaacsim.replicator.agent.core.settings import AssetPaths, PrimPaths
        from isaacsim.replicator.agent.core.simulation import SimulationManager
        from isaacsim.replicator.agent.core.stage_util import CharacterUtil
        from pxr import UsdGeom

        self.specs = tuple(specs)
        if not self.specs:
            raise ValueError("at least one character is required")
        missing = [str(spec.asset) for spec in self.specs if not Path(spec.asset).is_file()]
        if missing:
            raise FileNotFoundError(f"People assets are missing: {missing}")

        settings = carb.settings.get_settings()
        settings.set(
            AssetPaths.DEFAULT_BIPED_ASSET_PATH,
            str(self.assets_root / "Characters/Biped_Setup.usd"),
        )
        settings.set(AssetPaths.DEFAULT_CHARACTER_PATH, str(self.assets_root / "Characters"))
        settings.set(PrimPaths.CHARACTERS_PARENT_PATH, self.characters_parent)
        settings.set("/persistent/exts/omni.anim.people/character_prim_path", self.characters_parent)
        settings.set("/exts/omni.anim.people/navigation_settings/navmesh_enabled", True)
        settings.set("/exts/omni.anim.people/navigation_settings/dynamic_avoidance_enabled", True)

        UsdGeom.Xform.Define(self.stage, self.characters_parent)
        self.roots = []
        for index, spec in enumerate(self.specs, start=1):
            root = CharacterUtil.load_character_usd_to_stage(
                str(Path(spec.asset).resolve()),
                tuple(map(float, spec.start_xyz)),
                float(spec.initial_yaw_deg),
                spec.name,
            )
            self.roots.append(root)
            omni.kit.commands.execute(
                "ApplyNavMeshAPICommand",
                prim_path=str(root.GetPath()),
                api=NavSchema.NavMeshExcludeAPI,
            )
            print(
                f"OFFICIAL_CHARACTER_STAGED index={index}/{len(self.specs)} name={spec.name}",
                flush=True,
            )

        command_path = write_goto_commands(Path(command_path).resolve(), self.specs)
        settings.set("/exts/omni.anim.people/command_settings/command_file_path", str(command_path))
        settings.set("/exts/omni.anim.people/command_settings/number_of_loop", "0")

        print("OFFICIAL_CHARACTER_SETUP phase=load_default_skeleton_and_animations", flush=True)
        simulation = SimulationManager()
        simulation.load_default_skeleton_and_animations()
        print("OFFICIAL_CHARACTER_SETUP phase=setup_all_characters", flush=True)
        simulation.setup_all_characters()
        print("OFFICIAL_CHARACTER_SETUP phase=complete", flush=True)
        return {"command_path": str(command_path), "character_count": len(self.specs)}

    def wait_for_navmesh(
        self,
        *,
        navmesh_timeout_updates: int = 1200,
    ) -> dict[str, Any]:
        """Start an explicit Recast bake and wait for queryable routes.

        With no characters loaded (:attr:`specs` empty) the loop accepts the
        first available NavMesh; with characters it additionally requires every
        initial GoTo route to resolve on the baked surface.
        """

        import omni.anim.navigation.core as nav

        interface = nav.acquire_interface()
        self.navmesh_explicit_start_return = bool(interface.start_navmesh_baking())
        route_point_counts: list[list[int]] = []
        navmesh_ready_update: int | None = None
        for update_index in range(1, navmesh_timeout_updates + 1):
            self.app.update()
            candidate = interface.get_navmesh()
            if candidate is None:
                if update_index == 1 or update_index % 120 == 0:
                    print(
                        "OFFICIAL_NAVMESH_WAIT "
                        f"update={update_index}/{navmesh_timeout_updates} available=false",
                        flush=True,
                    )
                continue
            routes = []
            for spec in self.specs:
                targets = (spec.start_xyz, *spec.waypoints_xyz)
                routes.append(
                    [
                        candidate.query_shortest_path(start, goal)
                        for start, goal in zip(targets, targets[1:])
                    ]
                )
            if all(route is not None for character_routes in routes for route in character_routes):
                self.navmesh = candidate
                route_point_counts = [
                    [len(route.get_points()) for route in character_routes]
                    for character_routes in routes
                ]
                navmesh_ready_update = update_index
                print(
                    "OFFICIAL_NAVMESH_READY "
                    f"update={update_index} characters={len(self.specs)}",
                    flush=True,
                )
                break
        if self.navmesh is None:
            raise RuntimeError(
                "NavMesh did not become queryable "
                f"(explicit_start={self.navmesh_explicit_start_return})"
            )
        return {
            "navmesh_queryable": True,
            "navmesh_explicit_start_return": self.navmesh_explicit_start_return,
            "navmesh_ready_update": navmesh_ready_update,
            "navmesh_route_point_counts": route_point_counts,
        }

    def setup(
        self,
        specs: Iterable[OfficialCharacterSpec],
        command_path: Path,
        *,
        navmesh_timeout_updates: int = 1200,
    ) -> dict[str, Any]:
        """Load rigs, bind official behavior and wait for queryable routes."""

        character_metadata = self.load_characters(specs, command_path)
        navmesh_metadata = self.wait_for_navmesh(
            navmesh_timeout_updates=navmesh_timeout_updates
        )
        self.resolve_skelroots()
        return {
            **character_metadata,
            **navmesh_metadata,
            "controlled_skelroots": [str(root.GetPath()) for root in self.skelroots],
        }

    def validate_routes(self) -> list[dict[str, Any]]:
        """Validate every configured leg against the current baked NavMesh."""
        if self.navmesh is None:
            raise RuntimeError("wait_for_navmesh() must complete before route validation")
        rows: list[dict[str, Any]] = []
        for spec in self.specs:
            targets = (tuple(spec.start_xyz), *spec.waypoints_xyz)
            legs: list[bool] = []
            point_counts: list[int] = []
            for leg_index, (start, goal) in enumerate(zip(targets, targets[1:])):
                print(
                    f"OFFICIAL_ROUTE_QUERY name={spec.name} leg={leg_index} start={start} goal={goal}",
                    flush=True,
                )
                route = self.navmesh.query_shortest_path(start, goal)
                count = len(route.get_points()) if route is not None else 0
                print(
                    f"OFFICIAL_ROUTE_RESULT name={spec.name} leg={leg_index} points={count}",
                    flush=True,
                )
                point_counts.append(count)
                legs.append(count >= 2)
            rows.append(
                {
                    "name": spec.name,
                    "resolved": all(legs),
                    "leg_count": len(legs),
                    "legs_resolved": legs,
                    "path_point_counts": point_counts,
                    "path_point_count": sum(point_counts),
                }
            )
        return rows

    def resolve_skelroots(self) -> list[Any]:
        """Resolve every character SkelRoot; ``play`` depends on these handles.

        ``load_characters`` records the character roots; the SkelRoot indirection
        is only produced by ``CharacterUtil`` once the rigs are staged.  This is
        the piece that historically lived only inside :meth:`setup`, so callers
        that drive ``load_characters``/``wait_for_navmesh`` directly must invoke
        it before :meth:`play`.
        """

        from isaacsim.replicator.agent.core.stage_util import CharacterUtil

        self.skelroots = [CharacterUtil.get_character_skelroot_by_root(root) for root in self.roots]
        if any(root is None for root in self.skelroots):
            raise RuntimeError("official Biped setup did not create every character SkelRoot")
        return self.skelroots

    def play(
        self,
        *,
        time_codes_per_second: float = 60.0,
        target_framerate_hz: float | None = None,
        timeout_updates: int = 60,
        restart_timeline: bool = True,
    ) -> int:
        """Enter play state and wait for all native CharacterManager handles."""

        import omni.anim.graph.core as anim_graph
        import carb.settings
        import omni.timeline

        self.resolve_skelroots()
        self.stage.SetTimeCodesPerSecond(float(time_codes_per_second))
        # Several UrbanVerse source stages have a one-frame authored timeline.
        # Kit then wraps ``current_time`` every update even though Animation
        # Graph keeps advancing internally.  Give runtime validation and long
        # captures an explicit 24-hour non-looping interval.
        self.stage.SetStartTimeCode(0.0)
        self.stage.SetEndTimeCode(float(time_codes_per_second) * 24.0 * 60.0 * 60.0)
        settings = carb.settings.get_settings()
        settings.set("/app/player/useFixedTimeStepping", True)
        settings.set("/app/player/useFastMode", True)
        self.timeline = omni.timeline.get_timeline_interface()
        # Standalone official-People tools own the timeline and may restart it.
        # An Isaac Lab environment owns a live SimulationContext; stopping that
        # timeline invokes its render-on-stop callback and can block a headless
        # joint run before Recast is even queryable.  Integrated callers keep
        # the live timeline and reset the RL environment after People setup.
        if restart_timeline:
            self.timeline.stop()
        self.timeline.set_time_codes_per_second(float(time_codes_per_second))
        self.timeline.set_start_time(0.0)
        self.timeline.set_end_time(24.0 * 60.0 * 60.0)
        self.timeline.set_looping(False)
        if restart_timeline:
            self.timeline.set_current_time(0.0)
        if target_framerate_hz is not None:
            if float(target_framerate_hz) <= 0.0:
                raise ValueError("target_framerate_hz must be positive")
            # Standalone Kit updates otherwise inherit the experience default
            # (commonly 24 Hz), while the micromobility controller integrates
            # at its configured simulation dt.  Pinning both clocks prevents
            # People and two-wheelers from evolving on different time bases.
            self.timeline.set_target_framerate(float(target_framerate_hz))
            self.timeline.set_ticks_per_frame(1)
            settings.set("/app/player/targetRunLoopFrequency", float(target_framerate_hz))
        self.timeline.set_play_every_frame(True)
        if not self.timeline.is_playing():
            self.timeline.play()
        self.timeline.commit()
        for update_index in range(1, timeout_updates + 1):
            self.app.update()
            self.characters = [anim_graph.get_character(str(root.GetPath())) for root in self.skelroots]
            if len(self.characters) == len(self.skelroots) and all(
                character is not None for character in self.characters
            ):
                # Native AnimationGraph handles and Python BehaviorScript
                # instances are registered by separate Kit systems.  On a
                # cold RTX/camera launch the handles can become visible one or
                # more updates before OmniFinder publishes CharacterBehavior.
                # Wait for both readiness conditions instead of failing on the
                # first handle-ready frame.
                try:
                    self.resolve_behavior_scripts()
                except RuntimeError as error:
                    if update_index == 1 or update_index % 30 == 0:
                        print(
                            "OFFICIAL_CHARACTER_BEHAVIOR_WAIT "
                            f"update={update_index}/{timeout_updates} error={error}",
                            flush=True,
                        )
                    continue
                return update_index
        raise RuntimeError(
            "CharacterManager handles did not initialize while playing: "
            f"{[character is not None for character in self.characters]}"
        )

    def resolve_behavior_scripts(self) -> dict[str, Any]:
        """Resolve NVIDIA ``CharacterBehavior`` instances for controlled ticks.

        Isaac Sim 4.5 drives :class:`CharacterBehavior` from Kit's stage-update
        callback.  An Isaac Lab tensor-only headless loop does not emit that
        callback, while ``SimulationApp.update()`` also advances PhysX and is
        therefore unsafe to insert after ``env.step()``.  Keep the official
        CharacterBehavior/NavigationManager implementation, but retain its
        already-created instances so an integrated runner can invoke exactly
        their update method once per controlled physics step.

        Isaac People exposes its loaded script instance through
        ``Utils.fetch_target_character_instance_by_name``; using that same
        lookup also avoids Python class-identity differences introduced by
        Kit's OmniFinder loader.  No character pose, path or animation variable
        is replaced by project code.
        """

        from omni.anim.people.scripts.utils import Utils

        resolved: dict[str, Any] = {}
        for spec in self.specs:
            instance = Utils.fetch_target_character_instance_by_name(spec.name)
            if instance is not None and callable(getattr(instance, "on_update", None)):
                resolved[spec.name] = instance
        missing = sorted({spec.name for spec in self.specs} - set(resolved))
        if missing:
            raise RuntimeError(
                "official CharacterBehavior instances are missing: " + ", ".join(missing)
            )
        self.behavior_scripts = resolved
        self.behavior_time_s = float(self.timeline.get_current_time())
        if self.avoidance_radius_m is not None:
            self.configure_avoidance_radius(self.avoidance_radius_m)
        return dict(resolved)

    def configure_avoidance_radius(self, radius_m: float) -> None:
        """Raise the radius published through NVIDIA's NavigationManager.

        Isaac People 4.5 hard-codes a 0.5 m character radius inside
        ``CharacterBehavior.on_update``.  Dense resident roaming needs a
        slightly larger planning envelope to keep animated shoulders apart.
        The adapter leaves the official path planner and avoidance algorithm
        in control: it only replaces the radius argument passed to the
        official ``publish_character_positions`` method.  No pose, velocity or
        Animation Graph variable is written here.
        """

        radius_m = float(radius_m)
        if radius_m < 0.5:
            raise ValueError("official People avoidance radius must be at least 0.5 m")
        self.avoidance_radius_m = radius_m
        if not self.behavior_scripts:
            return
        for behavior in self.behavior_scripts.values():
            manager = getattr(behavior, "navigation_manager", None)
            if manager is None:
                raise RuntimeError("official NavigationManager is unavailable")
            original = getattr(
                manager,
                "_urbanverse_original_publish_character_positions",
                manager.publish_character_positions,
            )
            manager._urbanverse_original_publish_character_positions = original

            def publish(delta_time: float, _radius: float, *, _original=original) -> None:
                _original(delta_time, radius_m)

            manager.publish_character_positions = publish

    def advance_behaviors(self, delta_time_s: float) -> None:
        """Advance only official People behavior/navigation by one fixed tick.

        This deliberately does *not* call ``SimulationApp.update()`` and thus
        cannot introduce a second PhysX step into an Isaac Lab control cycle.
        Animation Graph consumes the variables written here on the following
        controlled environment step.
        """

        delta_time_s = float(delta_time_s)
        if delta_time_s <= 0.0:
            raise ValueError("delta_time_s must be positive")
        if not self.behavior_scripts:
            self.resolve_behavior_scripts()
        self.behavior_time_s += delta_time_s
        for spec in self.specs:
            self.behavior_scripts[spec.name].on_update(
                self.behavior_time_s, delta_time_s
            )
        # ICharacter::update is NVIDIA's public manual-tick API for processing
        # an Animation Graph without running the whole Kit application loop.
        # CharacterBehavior first publishes PathPoints/Walk variables; this
        # call consumes them and advances the root pose and skeleton exactly
        # once, without touching PhysX or the Go2 control clock.
        for character in self.characters:
            character.update(delta_time_s)

    def positions(self) -> np.ndarray:
        """Return current authoritative xyz positions for every character."""

        if not self.characters:
            raise RuntimeError("play() must initialize CharacterManager before reading positions")
        return np.asarray(
            [runtime_character_position(character) for character in self.characters],
            dtype=np.float64,
        )

    def inject_goto(self, character_name: str, target_xyz: Sequence[float]) -> None:
        """Inject one official ``GoTo`` command while the runtime is playing."""

        if self.timeline is None or not self.characters:
            raise RuntimeError("play() must initialize CharacterManager before command injection")
        if len(target_xyz) != 3:
            raise ValueError("target_xyz must contain x, y and z")
        if character_name not in {spec.name for spec in self.specs}:
            raise KeyError(f"unknown official character: {character_name}")
        from omni.anim.people.scripts.utils import Utils

        command = (
            f"{character_name} GoTo "
            + " ".join(f"{float(value):.6f}" for value in target_xyz)
            + " _"
        )
        Utils.runtime_inject_command(
            character_name,
            [command],
            force_inject=True,
            set_status=True,
        )

    def inject_yield_then_goto(
        self,
        character_name: str,
        target_xyz: Sequence[float],
        *,
        yield_duration_s: float,
    ) -> None:
        """Interrupt with an official ``Idle`` yield, then resume via ``GoTo``.

        NVIDIA's local avoidance can become geometrically trapped when two
        characters meet head-on in a narrow admitted strip.  Both commands
        still run through CharacterBehavior and NavigationManager; this helper
        only gives one character deterministic right-of-way and never writes a
        pose or Animation Graph variable directly.
        """

        if self.timeline is None or not self.characters:
            raise RuntimeError("play() must initialize CharacterManager before command injection")
        if len(target_xyz) != 3:
            raise ValueError("target_xyz must contain x, y and z")
        if float(yield_duration_s) <= 0.0:
            raise ValueError("yield_duration_s must be positive")
        if character_name not in {spec.name for spec in self.specs}:
            raise KeyError(f"unknown official character: {character_name}")
        from omni.anim.people.scripts.utils import Utils

        goto = (
            f"{character_name} GoTo "
            + " ".join(f"{float(value):.6f}" for value in target_xyz)
            + " _"
        )
        Utils.runtime_inject_command(
            character_name,
            [f"{character_name} Idle {float(yield_duration_s):.6f}", goto],
            force_inject=True,
            set_status=True,
        )

    def stop(self) -> None:
        if self.timeline is not None:
            self.timeline.stop()
