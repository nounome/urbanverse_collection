#!/usr/bin/env bash
set -Euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODE="${1:?usage: $0 constant|replay|route|three_point GPU_INDEX [LABEL] [TRAFFIC_SCENE_CONFIG]}"
GPU_INDEX="${2:-0}"
LABEL="${3:-scene10_${MODE}_isolation}"
TRAFFIC_SCENE_CONFIG="${TRAFFIC_SCENE_CONFIG:-${4:-}}"
REQUESTED_TRAFFIC_VEHICLE_COUNT="${TRAFFIC_VEHICLE_COUNT:-}"
TRAFFIC_ENABLED="${TRAFFIC_ENABLED:-1}"
REQUESTED_OVERVIEW_FIXED_EYE="${OVERVIEW_FIXED_EYE:-}"
REQUESTED_OVERVIEW_FIXED_TARGET="${OVERVIEW_FIXED_TARGET:-}"
REQUESTED_OVERVIEW_FOCAL_LENGTH="${OVERVIEW_FOCAL_LENGTH:-}"
REQUESTED_OVERVIEW_MOTION_START_EYE="${OVERVIEW_MOTION_START_EYE:-}"
REQUESTED_OVERVIEW_MOTION_END_EYE="${OVERVIEW_MOTION_END_EYE:-}"
REQUESTED_OVERVIEW_MOTION_SPEED_MPS="${OVERVIEW_MOTION_SPEED_MPS:-}"
REQUESTED_OVERVIEW_MOTION_PITCH_DOWN_DEG="${OVERVIEW_MOTION_PITCH_DOWN_DEG:-}"
REQUESTED_OVERVIEW_MOTION_LOOKAHEAD_M="${OVERVIEW_MOTION_LOOKAHEAD_M:-}"
REQUESTED_COLLECTION_LIGHT_PATH_SCALES="${COLLECTION_LIGHT_PATH_SCALES:-}"
REQUESTED_PEDESTRIAN_CONFIG="${PEDESTRIAN_CONFIG:-}"
REQUESTED_REFERENCE_ROUTE="${REFERENCE_ROUTE:-}"
PYTHON="${PROJECT_DIR}/repos/isaac45_probe/.venv/bin/python"
SOURCE_USD="${PROJECT_DIR}/data/urbanverse_craftbench/extracted/scene_10_cbd_cross_intersection_diverse_obstacles/Collected_export_version/export_version.usd"
SOURCE_TAR="${PROJECT_DIR}/data/urbanverse_craftbench/raw/scene_10_cbd_cross_intersection_diverse_obstacles/Collected_export_version.tar"
ROUTE="${REFERENCE_ROUTE:-${PROJECT_DIR}/configs/dynamic_agents/scene_inputs/scene10/local_stall_reference_route.json}"
LOOKAHEAD_DISTANCE="${LOOKAHEAD_DISTANCE:-1.20}"
MINIMUM_TRACKING_SPEED="${MINIMUM_TRACKING_SPEED:-0.24}"
MAX_FORWARD_SPEED="${MAX_FORWARD_SPEED:-0.34}"
MAX_TRACKING_YAW_RATE="${MAX_TRACKING_YAW_RATE:-0.18}"
CURVATURE_SPEED_GAIN="${CURVATURE_SPEED_GAIN:-2.0}"
YAW_COMMAND_GAIN="${YAW_COMMAND_GAIN:-1.0}"
MAXIMUM_CROSS_TRACK_ERROR="${MAXIMUM_CROSS_TRACK_ERROR:-5.0}"
POLICY_KIND="${POLICY_KIND:-robot_lab}"
EXTERNAL_POLICY_SOURCE="${EXTERNAL_POLICY_SOURCE:-${PROJECT_DIR}/data/locomotion_policies/rl_sar/376d42c9b128f963ab08579762d5a216a976ce39}"
GO2Z1_CHECKPOINT="${GO2Z1_CHECKPOINT:-}"
DURATION_S="${DURATION_S:-}"
STOP_AT_ROUTE_GOAL="${STOP_AT_ROUTE_GOAL:-0}"
REQUESTED_SEED="${SEED:-}"
SEED="${SEED:-20260812}"
CONSTANT_VX="${CONSTANT_VX:-0.24}"
CONSTANT_WZ="${CONSTANT_WZ:-0.18}"
MANEUVER_PLAN="${MANEUVER_PLAN:-}"
SCENE10_MULTIVEHICLE_TRAFFIC="${SCENE10_MULTIVEHICLE_TRAFFIC:-0}"
TRAFFIC_REGISTRY="${TRAFFIC_REGISTRY:-${PROJECT_DIR}/configs/dynamic_agents/scene_inputs/scene10/front_validated_vehicle_registry.json}"
TRAFFIC_AUDIT_INVENTORY="${TRAFFIC_AUDIT_INVENTORY:-${PROJECT_DIR}/configs/dynamic_agents/scene_inputs/scene10/vehicle_inventory.json}"
TRAFFIC_VALIDATED_ROUTES="${TRAFFIC_VALIDATED_ROUTES:-${PROJECT_DIR}/configs/dynamic_agents/scene_inputs/scene10/validated_routes.json}"
TRAFFIC_VALIDATED_STATIC_BODIES="${TRAFFIC_VALIDATED_STATIC_BODIES:-${PROJECT_DIR}/configs/dynamic_agents/scene_inputs/scene10/validated_static_vehicle_bodies.json}"
TRAFFIC_CONTINUOUS_LOOPING="${TRAFFIC_CONTINUOUS_LOOPING:-0}"
TRAFFIC_AUTOMOTIVE_ROUTES="${TRAFFIC_AUTOMOTIVE_ROUTES:-${PROJECT_DIR}/configs/dynamic_agents/scene_inputs/scene10/automotive_routes.json}"
TRAFFIC_VEHICLE_COUNT="${TRAFFIC_VEHICLE_COUNT:-5}"
TRAFFIC_INITIAL_FILL="${TRAFFIC_INITIAL_FILL:-0}"
PEDESTRIAN_CONFIG="${PEDESTRIAN_CONFIG:-}"
ROAMING_GHOST_CONFIG="${ROAMING_GHOST_CONFIG:-}"
MIXED_ROAMING_CONFIG="${MIXED_ROAMING_CONFIG:-}"
if [[ -n "${MIXED_ROAMING_CONFIG}" && -z "${REQUESTED_SEED}" ]]; then
    SEED="$("${PYTHON}" - "${PROJECT_DIR}" "${MIXED_ROAMING_CONFIG}" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[2])
if not p.is_absolute():p=Path(sys.argv[1])/p
print(int(json.loads(p.read_text())['fixed_seed']))
PY
    )" || exit 1
fi
ROAMING_GHOST_TWO_PANEL_VIDEO="${ROAMING_GHOST_TWO_PANEL_VIDEO:-0}"
ROAMING_VALIDATION_LEVEL="${ROAMING_VALIDATION_LEVEL:-smoke}"
ROAD_SWEEP_VIDEO="${ROAD_SWEEP_VIDEO:-1}"
GHOST_AB_NORMAL="${GHOST_AB_NORMAL:-0}"
GHOST_AB_HEADLESS="${GHOST_AB_HEADLESS:-0}"
ROAMING_HEADLESS="${ROAMING_HEADLESS:-0}"
PEDESTRIAN_LIFECYCLE_VALIDATION="${PEDESTRIAN_LIFECYCLE_VALIDATION:-0}"
MINIMUM_GO2_VEHICLE_CLEARANCE="${MINIMUM_GO2_VEHICLE_CLEARANCE:-1.0}"
OVERVIEW_VIDEO="${OVERVIEW_VIDEO:-0}"
GO2_FRONT_PINHOLE="${GO2_FRONT_PINHOLE:-0}"
GO2_THREE_CAMERA="${GO2_THREE_CAMERA:-0}"
THREE_CAMERA_WIDTH="${THREE_CAMERA_WIDTH:-480}"
THREE_CAMERA_HEIGHT="${THREE_CAMERA_HEIGHT:-384}"
if [[ "${GO2_FRONT_PINHOLE}" == "1" || "${GO2_THREE_CAMERA}" == "1" ]]; then
    OVERVIEW_VIDEO=1
fi
JOINT_THREE_PANEL_VIDEO="${JOINT_THREE_PANEL_VIDEO:-0}"
JOINT_FOUR_PANEL_VIDEO="${JOINT_FOUR_PANEL_VIDEO:-0}"
PEDESTRIAN_FOLLOW_INDEX="${PEDESTRIAN_FOLLOW_INDEX:-3}"
if [[ "${JOINT_THREE_PANEL_VIDEO}" == "1" && "${JOINT_FOUR_PANEL_VIDEO}" == "1" ]]; then
    echo "JOINT_THREE_PANEL_VIDEO and JOINT_FOUR_PANEL_VIDEO are exclusive" >&2
    exit 2
fi
if [[ "${JOINT_THREE_PANEL_VIDEO}" == "1" || "${JOINT_FOUR_PANEL_VIDEO}" == "1" || "${ROAMING_GHOST_TWO_PANEL_VIDEO}" == "1" ]]; then
    OVERVIEW_VIDEO=1
fi
OVERVIEW_WIDTH="${OVERVIEW_WIDTH:-1280}"
OVERVIEW_HEIGHT="${OVERVIEW_HEIGHT:-720}"
OVERVIEW_FPS="${OVERVIEW_FPS:-10}"
OVERVIEW_VIDEO_CODEC="${OVERVIEW_VIDEO_CODEC:-h264}"
OVERVIEW_H264_CRF="${OVERVIEW_H264_CRF:-18}"
OVERVIEW_H264_PRESET="${OVERVIEW_H264_PRESET:-veryfast}"
OVERVIEW_MOTION_BLUR="${OVERVIEW_MOTION_BLUR:-0}"
OVERVIEW_EXPOSURE_EV="${OVERVIEW_EXPOSURE_EV:-0.0}"
OVERVIEW_OUTPUT_BRIGHTNESS="${OVERVIEW_OUTPUT_BRIGHTNESS:-1.0}"
OVERVIEW_FOCAL_LENGTH="${OVERVIEW_FOCAL_LENGTH:-18.0}"
OVERVIEW_FIXED_EYE="${OVERVIEW_FIXED_EYE:-}"
OVERVIEW_FIXED_TARGET="${OVERVIEW_FIXED_TARGET:-}"
OVERVIEW_MOTION_START_EYE="${OVERVIEW_MOTION_START_EYE:-}"
OVERVIEW_MOTION_END_EYE="${OVERVIEW_MOTION_END_EYE:-}"
OVERVIEW_MOTION_SPEED_MPS="${OVERVIEW_MOTION_SPEED_MPS:-0.55}"
OVERVIEW_MOTION_PITCH_DOWN_DEG="${OVERVIEW_MOTION_PITCH_DOWN_DEG:-5.0}"
OVERVIEW_MOTION_LOOKAHEAD_M="${OVERVIEW_MOTION_LOOKAHEAD_M:-30.0}"
SCENE_GLOBAL_EYE="${SCENE_GLOBAL_EYE:-}"
SCENE_GLOBAL_TARGET="${SCENE_GLOBAL_TARGET:-}"
SCENE_GLOBAL_FOCAL_LENGTH="${SCENE_GLOBAL_FOCAL_LENGTH:-18.0}"
OVERVIEW_PREVIEW_TIME_S="${OVERVIEW_PREVIEW_TIME_S:-}"
GO2_HIGHLIGHT_COLOR="${GO2_HIGHLIGHT_COLOR:-}"
GO2_HIGHLIGHT_EMISSION="${GO2_HIGHLIGHT_EMISSION:-0.10}"
COLLECTION_LIGHT_SCALE="${COLLECTION_LIGHT_SCALE:-0.10}"
COLLECTION_DOME_LIGHT_SCALE="${COLLECTION_DOME_LIGHT_SCALE:-0.10}"
COLLECTION_DISTANT_LIGHT_SCALE="${COLLECTION_DISTANT_LIGHT_SCALE:-0.10}"
COLLECTION_SPHERE_LIGHT_SCALE="${COLLECTION_SPHERE_LIGHT_SCALE:-0.10}"
COLLECTION_LIGHT_PATH_SCALES="${COLLECTION_LIGHT_PATH_SCALES:-}"
SOURCE_DOME_BACKGROUND_VISIBLE="${SOURCE_DOME_BACKGROUND_VISIBLE:-}"
SOURCE_DOME_TEXTURE_ENABLED="${SOURCE_DOME_TEXTURE_ENABLED:-}"
OVERVIEW_CHASE_DISTANCE="${OVERVIEW_CHASE_DISTANCE:-8.0}"
OVERVIEW_LATERAL_OFFSET="${OVERVIEW_LATERAL_OFFSET:-3.0}"
OVERVIEW_CHASE_HEIGHT="${OVERVIEW_CHASE_HEIGHT:-6.5}"
OVERVIEW_TARGET_FORWARD="${OVERVIEW_TARGET_FORWARD:-3.0}"
OVERVIEW_TARGET_LATERAL="${OVERVIEW_TARGET_LATERAL:-0.0}"
MINIMUM_VISIBLE_VEHICLE_PASS_DURATION="${MINIMUM_VISIBLE_VEHICLE_PASS_DURATION:-3.0}"
OPPOSING_SHOWCASE_SPEED="${OPPOSING_SHOWCASE_SPEED:-3.12}"
OPPOSING_SHOWCASE_START_ROUTE_INDEX="${OPPOSING_SHOWCASE_START_ROUTE_INDEX:-0}"
TRAFFIC_VISIBILITY_DIAGNOSTIC_CONVOY="${TRAFFIC_VISIBILITY_DIAGNOSTIC_CONVOY:-0}"
ALLOW_SHARED_GPU="${ALLOW_SHARED_GPU:-0}"
REPLAY="${REPLAY:-}"
ISAAC_EXPERIENCE="${ISAAC_EXPERIENCE:-}"

if [[ -n "${ISAAC_EXPERIENCE}" ]]; then
    [[ "${ISAAC_EXPERIENCE}" = /* ]] || ISAAC_EXPERIENCE="${PROJECT_DIR}/${ISAAC_EXPERIENCE}"
    if [[ ! -f "${ISAAC_EXPERIENCE}" ]]; then
        echo "missing Kit experience: ${ISAAC_EXPERIENCE}" >&2
        exit 2
    fi
    ISAAC_EXPERIENCE="$(realpath "${ISAAC_EXPERIENCE}")"
fi

if [[ -n "${TRAFFIC_SCENE_CONFIG}" ]]; then
    [[ "${TRAFFIC_SCENE_CONFIG}" = /* ]] || TRAFFIC_SCENE_CONFIG="${PROJECT_DIR}/${TRAFFIC_SCENE_CONFIG}"
    mapfile -t scene_values < <(PYTHONPATH="${PROJECT_DIR}/tools" "${PYTHON}" - "${TRAFFIC_SCENE_CONFIG}" <<'PY'
import sys
from urbanverse.dynamic_agents.config import TrafficSceneConfig
c = TrafficSceneConfig.load(sys.argv[1])
for value in (
    c.source_usd,
    c.source_tar or "",
    c.go2_reference_route,
    c.vehicle_catalog,
    c.automotive_routes,
    c.traffic["vehicle_count"],
    " ".join(map(str, c.overview.get("eye", []))),
    " ".join(map(str, c.overview.get("target", []))),
    c.overview.get("focal_length", 18.0),
    c.pedestrian_config or "",
    " ".join(map(str, c.overview.get("motion_start_eye", []))),
    " ".join(map(str, c.overview.get("motion_end_eye", []))),
    c.overview.get("motion_speed_mps", 0.55),
    c.overview.get("motion_pitch_down_deg", 5.0),
    c.overview.get("motion_lookahead_m", 30.0),
):
    print(value)
PY
    )
    SOURCE_USD="${scene_values[0]}"
    SOURCE_TAR="${scene_values[1]}"
    ROUTE="${REQUESTED_REFERENCE_ROUTE:-${scene_values[2]}}"
    TRAFFIC_REGISTRY="${scene_values[3]}"
    TRAFFIC_AUTOMOTIVE_ROUTES="${scene_values[4]}"
    TRAFFIC_VEHICLE_COUNT="${REQUESTED_TRAFFIC_VEHICLE_COUNT:-${scene_values[5]}}"
    if [[ "${scene_values[5]}" == "0" ]]; then
        TRAFFIC_ENABLED=0
    fi
    OVERVIEW_FIXED_EYE="${REQUESTED_OVERVIEW_FIXED_EYE:-${scene_values[6]}}"
    OVERVIEW_FIXED_TARGET="${REQUESTED_OVERVIEW_FIXED_TARGET:-${scene_values[7]}}"
    OVERVIEW_FOCAL_LENGTH="${REQUESTED_OVERVIEW_FOCAL_LENGTH:-${scene_values[8]}}"
    PEDESTRIAN_CONFIG="${REQUESTED_PEDESTRIAN_CONFIG:-${scene_values[9]}}"
    OVERVIEW_MOTION_START_EYE="${REQUESTED_OVERVIEW_MOTION_START_EYE:-${scene_values[10]}}"
    OVERVIEW_MOTION_END_EYE="${REQUESTED_OVERVIEW_MOTION_END_EYE:-${scene_values[11]}}"
    OVERVIEW_MOTION_SPEED_MPS="${REQUESTED_OVERVIEW_MOTION_SPEED_MPS:-${scene_values[12]}}"
    OVERVIEW_MOTION_PITCH_DOWN_DEG="${REQUESTED_OVERVIEW_MOTION_PITCH_DOWN_DEG:-${scene_values[13]}}"
    OVERVIEW_MOTION_LOOKAHEAD_M="${REQUESTED_OVERVIEW_MOTION_LOOKAHEAD_M:-${scene_values[14]}}"
    # DomeLight_04 is a Scene 10 path, not a cross-scene contract.  Portable
    # scenes only receive path-specific light overrides when explicitly asked.
    COLLECTION_LIGHT_PATH_SCALES="${REQUESTED_COLLECTION_LIGHT_PATH_SCALES}"
    if [[ "${TRAFFIC_ENABLED}" == "1" ]]; then
        SCENE10_MULTIVEHICLE_TRAFFIC=1
        TRAFFIC_CONTINUOUS_LOOPING=1
    else
        SCENE10_MULTIVEHICLE_TRAFFIC=0
        TRAFFIC_CONTINUOUS_LOOPING=0
    fi
elif [[ -z "${COLLECTION_LIGHT_PATH_SCALES}" ]]; then
    # Preserve the established Scene10 lighting override only for the legacy
    # default scene.  A path under /UrbanVerseAsset is neither present nor
    # portable when an official-Recast scene keeps its native /World root.
    COLLECTION_LIGHT_PATH_SCALES="/UrbanVerseAsset/DomeLight_04=1.0"
fi
if [[ -n "${ROAMING_GHOST_CONFIG}" || -n "${MIXED_ROAMING_CONFIG}" ]]; then
    # Either resident runtime owns People in this composition. Do not also
    # instantiate the legacy fixed-route manager from the scene config.
    PEDESTRIAN_CONFIG=""
fi

if [[ "${MODE}" != "constant" && "${MODE}" != "replay" && "${MODE}" != "route" && "${MODE}" != "three_point" ]]; then
    echo "mode must be constant, replay, route, or three_point" >&2
    exit 2
fi
cd "${PROJECT_DIR}"
export OMNI_KIT_ACCEPT_EULA=YES
unset CUDA_VISIBLE_DEVICES || true
export OMP_NUM_THREADS="${URBANVERSE_CPU_THREADS:-8}"
export OMP_THREAD_LIMIT="${URBANVERSE_CPU_THREADS:-8}"
export MKL_NUM_THREADS="${URBANVERSE_CPU_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${URBANVERSE_CPU_THREADS:-8}"
export PXR_WORK_THREAD_LIMIT="${URBANVERSE_CPU_THREADS:-8}"

policy_paths="$("${PYTHON}" "${PROJECT_DIR}/tools/urbanverse/dynamic_agents/navigation/policy_selection.py" \
    --kind "${POLICY_KIND}" --source "${EXTERNAL_POLICY_SOURCE}" \
    --checkpoint "${GO2Z1_CHECKPOINT:-${GO2_CHECKPOINT:-}}" --policy "${GO2_POLICY:-}")" || exit 2
mapfile -t resolved_policy_paths <<<"${policy_paths}"
CHECKPOINT="${resolved_policy_paths[0]}"
POLICY="${resolved_policy_paths[1]}"
gpu_state="$(nvidia-smi -i "${GPU_INDEX}" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)"
memory_used="$(awk -F, '{gsub(/ /,"",$1); print $1}' <<<"${gpu_state}")"
utilization="$(awk -F, '{gsub(/ /,"",$2); print $2}' <<<"${gpu_state}")"
if (( memory_used > 1000 || utilization > 20 )) && [[ "${ALLOW_SHARED_GPU}" != "1" ]]; then
    echo "GPU ${GPU_INDEX} is busy: ${gpu_state}" >&2
    exit 2
fi
if (( memory_used > 1000 || utilization > 20 )); then
    echo "WARNING: sharing GPU ${GPU_INDEX} by explicit ALLOW_SHARED_GPU=1: ${gpu_state}" >&2
fi
nvidia-smi -L

timestamp="$(date +%Y%m%d_%H%M%S_%N)"
commit="$(git rev-parse --short HEAD)"
RUN_DIR="${PROJECT_DIR}/outputs/rtx3090_isaac45/go2_policy_isolation/run_${timestamp}_${LABEL}_${commit}_gpu${GPU_INDEX}"
mkdir -p "${RUN_DIR}/controller" "${RUN_DIR}/metadata/preflight" "${RUN_DIR}/captures" "${RUN_DIR}/visualizations"
cp "${CHECKPOINT}" "${RUN_DIR}/controller/checkpoint.pt"
cp "${POLICY}" "${RUN_DIR}/controller/policy.pt"
exec > >(tee -a "${RUN_DIR}/run_log.txt") 2>&1
echo "RUN_DIR=${RUN_DIR} MODE=${MODE} GPU=${GPU_INDEX}"
git status --short | tee "${RUN_DIR}/metadata/preflight/git_status_short.txt"
git rev-parse HEAD | tee "${RUN_DIR}/metadata/preflight/git_commit.txt"
hostname | tee "${RUN_DIR}/metadata/preflight/hostname.txt"
nvidia-smi -L | tee "${RUN_DIR}/metadata/preflight/nvidia_smi_L.txt"
lspci -nn | rg -i 'NVIDIA|VGA|3D controller' | tee "${RUN_DIR}/metadata/preflight/lspci_nvidia.txt"
ls -l /dev/nvidia* | tee "${RUN_DIR}/metadata/preflight/nvidia_device_nodes.txt"
nvidia-smi | tee "${RUN_DIR}/metadata/preflight/nvidia_smi.txt"
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader | tee "${RUN_DIR}/metadata/preflight/nvidia_smi_query.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader | tee "${RUN_DIR}/metadata/preflight/nvidia_smi_processes.csv"
VISIBLE_GPU_COUNT="$(nvidia-smi -L | wc -l)"
PCI_GPU_COUNT="$(lspci -nn | rg -i 'VGA compatible controller|3D controller' | rg -i NVIDIA | wc -l)"
DEVICE_GPU_COUNT="$(find /dev -maxdepth 1 -type c -name 'nvidia[0-9]*' | wc -l)"
if [[ "${VISIBLE_GPU_COUNT}" -ne "${PCI_GPU_COUNT}" || "${VISIBLE_GPU_COUNT}" -ne "${DEVICE_GPU_COUNT}" ]]; then
    echo "GPU inventory inconsistent: nvidia-smi=${VISIBLE_GPU_COUNT} PCI=${PCI_GPU_COUNT} nodes=${DEVICE_GPU_COUNT}" >&2
    exit 3
fi
touch "${RUN_DIR}/metadata/preflight/health_inventory_consistent"
if [[ -n "${ISAAC_EXPERIENCE}" ]]; then
    printf '%s\n' "${ISAAC_EXPERIENCE}" | tee "${RUN_DIR}/metadata/preflight/kit_experience.txt"
    sha256sum "${ISAAC_EXPERIENCE}" | tee "${RUN_DIR}/metadata/preflight/kit_experience_sha256.txt"
fi

command=(
    taskset -c "${URBANVERSE_CPU_AFFINITY:-4-11}" "${PYTHON}"
    "${PROJECT_DIR}/tools/urbanverse/dynamic_agents/integration/go2_traffic_probe.py"
    --mode "${MODE}" --gpu "${GPU_INDEX}" --run-dir "${RUN_DIR}"
    --seed "${SEED}"
    --constant-vx "${CONSTANT_VX}" --constant-wz "${CONSTANT_WZ}"
    --source-usd "${SOURCE_USD}"
    --reference-route "${ROUTE}" --route-config "${PROJECT_DIR}/configs/urbanverse_go2_routes.json"
    --lookahead-distance "${LOOKAHEAD_DISTANCE}"
    --minimum-tracking-speed "${MINIMUM_TRACKING_SPEED}"
    --max-forward-speed "${MAX_FORWARD_SPEED}"
    --max-tracking-yaw-rate "${MAX_TRACKING_YAW_RATE}"
    --curvature-speed-gain "${CURVATURE_SPEED_GAIN}"
    --yaw-command-gain "${YAW_COMMAND_GAIN}"
    --maximum-cross-track-error "${MAXIMUM_CROSS_TRACK_ERROR}"
    --policy-kind "${POLICY_KIND}"
    --external-policy-source "${EXTERNAL_POLICY_SOURCE}"
    --wrapper-template "${PROJECT_DIR}/configs/wrappers/urbanverse_reference_wrapper.usda.in"
    --checkpoint "${RUN_DIR}/controller/checkpoint.pt" --policy "${RUN_DIR}/controller/policy.pt"
    --opposing-showcase-speed "${OPPOSING_SHOWCASE_SPEED}"
    --opposing-showcase-start-route-index "${OPPOSING_SHOWCASE_START_ROUTE_INDEX}"
    --collection-light-scale "${COLLECTION_LIGHT_SCALE}"
    --collection-dome-light-scale "${COLLECTION_DOME_LIGHT_SCALE}"
    --collection-distant-light-scale "${COLLECTION_DISTANT_LIGHT_SCALE}"
    --collection-sphere-light-scale "${COLLECTION_SPHERE_LIGHT_SCALE}"
)
if [[ "${ROUTE_REVIEW_ONLY:-0}" == "1" ]]; then
    command+=(--route-review-only)
fi
if [[ -n "${ISAAC_EXPERIENCE}" ]]; then
    command+=(--experience "${ISAAC_EXPERIENCE}")
fi
if [[ -n "${SOURCE_TAR}" ]]; then
    command+=(--source-tar "${SOURCE_TAR}")
fi
IFS=';' read -r -a collection_light_path_scales <<<"${COLLECTION_LIGHT_PATH_SCALES}"
for light_path_scale in "${collection_light_path_scales[@]}"; do
    [[ -n "${light_path_scale}" ]] && command+=(--collection-light-path-scale "${light_path_scale}")
done
if [[ "${SOURCE_DOME_BACKGROUND_VISIBLE}" == "1" ]]; then
    command+=(--source-dome-background-visible)
elif [[ "${SOURCE_DOME_BACKGROUND_VISIBLE}" == "0" ]]; then
    command+=(--no-source-dome-background-visible)
fi
if [[ "${SOURCE_DOME_TEXTURE_ENABLED}" == "1" ]]; then
    command+=(--source-dome-texture-enabled)
elif [[ "${SOURCE_DOME_TEXTURE_ENABLED}" == "0" ]]; then
    command+=(--no-source-dome-texture-enabled)
fi
if [[ "${MODE}" == "replay" ]]; then
    if [[ -z "${REPLAY}" || ! -f "${REPLAY}" ]]; then
        echo "replay mode requires REPLAY=/absolute/path/to/trajectory.jsonl" >&2
        exit 2
    fi
    command+=(--replay-trajectory "${REPLAY}")
fi
if [[ "${MODE}" == "three_point" ]]; then
    test -n "${MANEUVER_PLAN}"
    command+=(--maneuver-plan "${MANEUVER_PLAN}")
fi
if [[ -n "${DURATION_S}" ]]; then
    command+=(--duration-s "${DURATION_S}")
fi
if [[ "${STOP_AT_ROUTE_GOAL}" == "1" ]]; then
    command+=(--stop-at-route-goal)
fi
if [[ "${SCENE10_MULTIVEHICLE_TRAFFIC}" == "1" ]]; then
    if [[ -n "${TRAFFIC_SCENE_CONFIG}" ]]; then
        command+=(--multivehicle-traffic --traffic-scene-config "${TRAFFIC_SCENE_CONFIG}")
    else
        command+=(
            --scene10-multivehicle-traffic
            --traffic-audit-inventory "${TRAFFIC_AUDIT_INVENTORY}"
            --traffic-validated-routes "${TRAFFIC_VALIDATED_ROUTES}"
            --traffic-validated-static-bodies "${TRAFFIC_VALIDATED_STATIC_BODIES}"
        )
    fi
    command+=(
        --traffic-registry "${TRAFFIC_REGISTRY}"
        --traffic-vehicle-count "${TRAFFIC_VEHICLE_COUNT}"
        --minimum-go2-vehicle-clearance "${MINIMUM_GO2_VEHICLE_CLEARANCE}"
    )
    if [[ "${TRAFFIC_VISIBILITY_DIAGNOSTIC_CONVOY}" == "1" ]]; then
        command+=(--traffic-visibility-diagnostic-convoy)
    fi
    if [[ "${TRAFFIC_CONTINUOUS_LOOPING}" == "1" ]]; then
        command+=(
            --traffic-continuous-looping
            --traffic-automotive-routes "${TRAFFIC_AUTOMOTIVE_ROUTES}"
        )
        if [[ "${TRAFFIC_INITIAL_FILL}" == "1" ]]; then
            command+=(--traffic-initial-fill)
        fi
    fi
fi
if [[ "${SCENE10_MULTIVEHICLE_TRAFFIC}" != "1" && -n "${TRAFFIC_SCENE_CONFIG}" ]]; then
    command+=(--traffic-scene-config "${TRAFFIC_SCENE_CONFIG}" --traffic-vehicle-count "${TRAFFIC_VEHICLE_COUNT}")
fi
if [[ "${OVERVIEW_VIDEO}" == "1" ]]; then
    command+=(
        --overview-video
        --overview-width "${OVERVIEW_WIDTH}"
        --overview-height "${OVERVIEW_HEIGHT}"
        --overview-fps "${OVERVIEW_FPS}"
        --overview-video-codec "${OVERVIEW_VIDEO_CODEC}"
        --overview-h264-crf "${OVERVIEW_H264_CRF}"
        --overview-h264-preset "${OVERVIEW_H264_PRESET}"
        --overview-exposure-ev "${OVERVIEW_EXPOSURE_EV}"
        --overview-output-brightness "${OVERVIEW_OUTPUT_BRIGHTNESS}"
        --overview-focal-length "${OVERVIEW_FOCAL_LENGTH}"
        --overview-chase-distance "${OVERVIEW_CHASE_DISTANCE}"
        --overview-lateral-offset "${OVERVIEW_LATERAL_OFFSET}"
        --overview-chase-height "${OVERVIEW_CHASE_HEIGHT}"
        --overview-target-forward "${OVERVIEW_TARGET_FORWARD}"
        --overview-target-lateral "${OVERVIEW_TARGET_LATERAL}"
        --minimum-visible-vehicle-pass-duration "${MINIMUM_VISIBLE_VEHICLE_PASS_DURATION}"
    )
    if [[ "${OVERVIEW_MOTION_BLUR}" == "1" ]]; then
        command+=(--overview-motion-blur)
    fi
    if [[ -n "${OVERVIEW_FIXED_EYE}" || -n "${OVERVIEW_FIXED_TARGET}" ]]; then
        test -n "${OVERVIEW_FIXED_EYE}"
        test -n "${OVERVIEW_FIXED_TARGET}"
        read -r -a overview_fixed_eye <<<"${OVERVIEW_FIXED_EYE}"
        read -r -a overview_fixed_target <<<"${OVERVIEW_FIXED_TARGET}"
        [[ "${#overview_fixed_eye[@]}" -eq 3 ]]
        [[ "${#overview_fixed_target[@]}" -eq 3 ]]
        command+=(--overview-fixed-eye "${overview_fixed_eye[@]}")
        command+=(--overview-fixed-target "${overview_fixed_target[@]}")
    fi
    if [[ -n "${OVERVIEW_MOTION_START_EYE}" || -n "${OVERVIEW_MOTION_END_EYE}" ]]; then
        test -z "${OVERVIEW_FIXED_EYE}"
        test -z "${OVERVIEW_FIXED_TARGET}"
        test -n "${OVERVIEW_MOTION_START_EYE}"
        test -n "${OVERVIEW_MOTION_END_EYE}"
        read -r -a overview_motion_start_eye <<<"${OVERVIEW_MOTION_START_EYE}"
        read -r -a overview_motion_end_eye <<<"${OVERVIEW_MOTION_END_EYE}"
        [[ "${#overview_motion_start_eye[@]}" -eq 3 ]]
        [[ "${#overview_motion_end_eye[@]}" -eq 3 ]]
        command+=(
            --overview-motion-start-eye "${overview_motion_start_eye[@]}"
            --overview-motion-end-eye "${overview_motion_end_eye[@]}"
            --overview-motion-speed-mps "${OVERVIEW_MOTION_SPEED_MPS}"
            --overview-motion-pitch-down-deg "${OVERVIEW_MOTION_PITCH_DOWN_DEG}"
            --overview-motion-lookahead-m "${OVERVIEW_MOTION_LOOKAHEAD_M}"
        )
    fi
    if [[ -n "${OVERVIEW_PREVIEW_TIME_S}" ]]; then
        command+=(--overview-preview-time-s "${OVERVIEW_PREVIEW_TIME_S}")
    fi
fi
if [[ "${JOINT_THREE_PANEL_VIDEO}" == "1" ]]; then
    command+=(--joint-three-panel-video --pedestrian-follow-index "${PEDESTRIAN_FOLLOW_INDEX}")
fi
if [[ "${JOINT_FOUR_PANEL_VIDEO}" == "1" ]]; then
    test -n "${SCENE_GLOBAL_EYE}"
    test -n "${SCENE_GLOBAL_TARGET}"
    read -r -a scene_global_eye <<<"${SCENE_GLOBAL_EYE}"
    read -r -a scene_global_target <<<"${SCENE_GLOBAL_TARGET}"
    [[ "${#scene_global_eye[@]}" -eq 3 ]]
    [[ "${#scene_global_target[@]}" -eq 3 ]]
    command+=(
        --joint-four-panel-video
        --pedestrian-follow-index "${PEDESTRIAN_FOLLOW_INDEX}"
        --scene-global-eye "${scene_global_eye[@]}"
        --scene-global-target "${scene_global_target[@]}"
        --scene-global-focal-length "${SCENE_GLOBAL_FOCAL_LENGTH}"
    )
fi
if [[ -n "${ROAMING_GHOST_CONFIG}" ]]; then
    command+=(
        --roaming-ghost-config "${ROAMING_GHOST_CONFIG}"
        --roaming-validation-level "${ROAMING_VALIDATION_LEVEL}"
    )
fi
if [[ -n "${MIXED_ROAMING_CONFIG}" ]]; then
    command+=(
        --mixed-roaming-config "${MIXED_ROAMING_CONFIG}"
        --roaming-validation-level "${ROAMING_VALIDATION_LEVEL}"
    )
fi
if [[ "${ROAMING_GHOST_TWO_PANEL_VIDEO}" == "1" ]]; then
    command+=(--roaming-ghost-two-panel-video --pedestrian-follow-index "${PEDESTRIAN_FOLLOW_INDEX}")
elif [[ "${ROAMING_GHOST_TWO_PANEL_VIDEO}" == "0" ]]; then
    command+=(--disable-roaming-two-panel-video)
fi
if [[ "${ROAD_SWEEP_VIDEO}" == "0" ]]; then
    command+=(--disable-road-sweep-video)
fi
if [[ "${GHOST_AB_NORMAL}" == "1" ]]; then
    command+=(--ghost-ab-normal)
fi
if [[ "${ROAMING_HEADLESS}" == "1" ]]; then
    command+=(--roaming-headless)
elif [[ "${GHOST_AB_HEADLESS}" == "1" ]]; then
    command+=(--ghost-ab-headless)
fi
if [[ "${GO2_FRONT_PINHOLE}" == "1" ]]; then
    command+=(--go2-front-pinhole)
fi
if [[ "${GO2_THREE_CAMERA}" == "1" ]]; then
    command+=(--go2-three-camera --three-camera-width "${THREE_CAMERA_WIDTH}" --three-camera-height "${THREE_CAMERA_HEIGHT}")
fi
if [[ -n "${PEDESTRIAN_CONFIG}" ]]; then
    command+=(--pedestrian-config "${PEDESTRIAN_CONFIG}")
fi
if [[ "${PEDESTRIAN_LIFECYCLE_VALIDATION}" == "1" ]]; then
    command+=(--pedestrian-lifecycle-validation)
fi
if [[ -n "${GO2_HIGHLIGHT_COLOR}" ]]; then
    read -r -a go2_highlight_color <<<"${GO2_HIGHLIGHT_COLOR}"
    [[ "${#go2_highlight_color[@]}" -eq 3 ]]
    command+=(--go2-highlight-color "${go2_highlight_color[@]}")
    command+=(--go2-highlight-emission "${GO2_HIGHLIGHT_EMISSION}")
fi
"${command[@]}"
capture_process_exit=$?
PYTHONPATH="${PROJECT_DIR}/tools" "${PYTHON}" -m urbanverse.dynamic_agents.admission.finalize_process "${RUN_DIR}" "${capture_process_exit}"
exit $?
