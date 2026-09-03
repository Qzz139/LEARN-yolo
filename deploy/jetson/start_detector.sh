#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/../.." && pwd)"
config_file="${YOLO_JETSON_CONFIG:-${script_dir}/jetson.env}"

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

is_true() {
    case "${1,,}" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

usage() {
    cat <<'EOF'
Usage: ./deploy/jetson/start_detector.sh [options]

Options:
  --model-id ID          Select a directory under weights/
  --model-file FILE      Select the model artifact in that directory
  --camera-source VALUE  Camera index/path, or auto (default: auto)
  --no-build             Skip colcon build for this start
  --record               Start video recording with the detector
  --view                 Open the detector live OpenCV window
  --check-only           Check environment, model and camera without starting
  -h, --help             Show this help
EOF
}

[[ -f "${config_file}" ]] || fail "Config file not found: ${config_file}"
# shellcheck disable=SC1090
source "${config_file}"

check_only=false
while (($#)); do
    case "$1" in
        --model-id)
            (($# >= 2)) || fail "--model-id requires a value"
            MODEL_ID="$2"
            shift 2
            ;;
        --model-file)
            (($# >= 2)) || fail "--model-file requires a value"
            MODEL_FILE="$2"
            shift 2
            ;;
        --camera-source)
            (($# >= 2)) || fail "--camera-source requires a value"
            CAMERA_SOURCE="$2"
            shift 2
            ;;
        --no-build)
            BUILD_ON_START=false
            shift
            ;;
        --record)
            RECORD_ON_START=true
            shift
            ;;
        --view)
            OPEN_VIEWER=true
            shift
            ;;
        --check-only)
            check_only=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            fail "Unknown option: $1"
            ;;
    esac
done

: "${ROS_DISTRO:=foxy}"
: "${VENV_PATH:=${HOME}/venvs/yolo_ros2}"
: "${MODEL_ID:=yolo26m_tabletop_v1}"
: "${MODEL_FILE:=best.pt}"
: "${CAMERA_SOURCE:=auto}"
: "${CAMERA_BY_ID:=}"
: "${CAMERA_WAIT_SECONDS:=30}"
: "${CAMERA_BACKEND:=auto}"
: "${CAMERA_FPS:=30.0}"
: "${CAMERA_FRAME_ID:=camera_optical_frame}"
: "${CAMERA_RECONNECT_INTERVAL:=2.0}"
: "${CAMERA_READ_FAILURE_THRESHOLD:=3}"
: "${DEVICE:=0}"
: "${IMGSZ:=640}"
: "${CONF_THRESHOLD:=0.25}"
: "${IOU_THRESHOLD:=0.45}"
: "${PUBLISH_ANNOTATED_IMAGE:=true}"
: "${CAPTURE_OUTPUT_DIR:=}"
: "${RECORD_ON_START:=false}"
: "${RECORDING_FPS:=10.0}"
: "${RECORDING_CODEC:=mp4v}"
: "${BUILD_ON_START:=true}"
: "${OPEN_VIEWER:=false}"

if is_true "${OPEN_VIEWER}" \
    && [[ -z "${DISPLAY:-}" ]] \
    && [[ -z "${WAYLAND_DISPLAY:-}" ]]; then
    fail "--view requires a graphical desktop (DISPLAY or WAYLAND_DISPLAY)."
fi

ros_setup="/opt/ros/${ROS_DISTRO}/setup.bash"
venv_activate="${VENV_PATH}/bin/activate"
workspace="${project_root}/ros2_ws"
[[ -n "${MODEL_FILE}" && "${MODEL_FILE}" != */* && "${MODEL_FILE}" != *\\* ]] \
    || fail "MODEL_FILE must be a file name inside weights/${MODEL_ID}."
model_path="${project_root}/weights/${MODEL_ID}/${MODEL_FILE}"
output_dir="${project_root}/outputs/jetson"
pid_file="${output_dir}/yolo_detector.pid"
if [[ -z "${CAPTURE_OUTPUT_DIR}" ]]; then
    CAPTURE_OUTPUT_DIR="${project_root}/outputs/captures"
fi

[[ -f "${ros_setup}" ]] || fail "ROS 2 setup not found: ${ros_setup}"
[[ -f "${venv_activate}" ]] || fail "Python environment not found: ${VENV_PATH}"
[[ -d "${workspace}/src/yolo_detector" ]] || fail "ROS 2 package not found in ${workspace}"
[[ -f "${model_path}" ]] || fail "Model not found: ${model_path}. Run git lfs pull after git pull."

if [[ "$(wc -c < "${model_path}")" -lt 1024 ]] || head -n 1 "${model_path}" | grep -q 'git-lfs.github.com/spec'; then
    fail "${model_path} is a Git LFS pointer, not model data. Run: git lfs pull"
fi

if [[ ! "${CAMERA_WAIT_SECONDS}" =~ ^[0-9]+$ ]]; then
    fail "CAMERA_WAIT_SECONDS must be a non-negative integer."
fi

resolve_camera_device() {
    if [[ "${CAMERA_SOURCE}" != "auto" ]]; then
        if [[ "${CAMERA_SOURCE}" =~ ^[0-9]+$ ]]; then
            printf '/dev/video%s\n' "${CAMERA_SOURCE}"
        else
            printf '%s\n' "${CAMERA_SOURCE}"
        fi
        return 0
    fi

    if [[ -n "${CAMERA_BY_ID}" ]]; then
        preferred="/dev/v4l/by-id/${CAMERA_BY_ID}"
        if [[ -e "${preferred}" && -r "${preferred}" ]]; then
            printf '%s\n' "${preferred}"
            return 0
        fi
    fi

    for candidate in /dev/v4l/by-id/*-video-index0 /dev/video*; do
        if [[ -e "${candidate}" && -r "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    return 1
}

camera_device=""
camera_deadline=$((SECONDS + CAMERA_WAIT_SECONDS))
while true; do
    candidate="$(resolve_camera_device || true)"
    if [[ -n "${candidate}" && -e "${candidate}" && -r "${candidate}" ]]; then
        camera_device="${candidate}"
        break
    fi
    if ((SECONDS >= camera_deadline)); then
        fail "USB camera did not appear within ${CAMERA_WAIT_SECONDS}s. Check lsusb and dmesg."
    fi
    printf 'Waiting for USB camera... (%ss remaining)\n' \
        "$((camera_deadline - SECONDS))"
    sleep 1
done

# ROS and virtualenv setup files may inspect optional unset variables. Suspend
# nounset only while sourcing them, then restore strict mode.
set +u
# shellcheck disable=SC1090
source "${ros_setup}"
# shellcheck disable=SC1090
source "${venv_activate}"
set -u

# JetPack 5 / Ubuntu 20.04 may otherwise fail while importing PyTorch with
# "cannot allocate memory in static TLS block". Use the same workaround that
# has already been validated manually on this Jetson.
for libgomp_path in \
    /lib/aarch64-linux-gnu/libgomp.so.1 \
    /usr/lib/aarch64-linux-gnu/libgomp.so.1; do
    if [[ -f "${libgomp_path}" ]]; then
        case ":${LD_PRELOAD:-}:" in
            *":${libgomp_path}:"*) ;;
            *) export LD_PRELOAD="${libgomp_path}${LD_PRELOAD:+:${LD_PRELOAD}}" ;;
        esac
        break
    fi
done
python3 - <<'PY' || fail "Python environment cannot import required runtime packages."
import cv2
import rclpy
from ultralytics import YOLO
print(f"Runtime imports: PASS (OpenCV {cv2.__version__})")
PY

camera_probe="${camera_device}"
python3 - "${camera_probe}" <<'PY' || fail "Cannot read a frame from USB camera ${camera_device}."
import sys
import time
import cv2

raw = sys.argv[1]
source = int(raw) if raw.isdigit() else raw
for _ in range(10):
    capture = cv2.VideoCapture(source)
    try:
        ok, frame = capture.read()
        if ok and frame is not None:
            print(f"USB camera: PASS ({frame.shape[1]}x{frame.shape[0]})")
            raise SystemExit(0)
    finally:
        capture.release()
    time.sleep(0.5)
raise SystemExit(1)
PY

if is_true "${BUILD_ON_START}"; then
    colcon_bin="$(command -v colcon || true)"
    [[ -n "${colcon_bin}" ]] || fail "colcon is not installed in the active environment."
    printf 'Building ROS 2 package...\n'
    (
        cd "${workspace}"
        python3 "${colcon_bin}" build --symlink-install --packages-select yolo_detector
    )
fi

local_setup="${workspace}/install/local_setup.bash"
[[ -f "${local_setup}" ]] || fail "ROS 2 workspace is not built: ${local_setup}"
set +u
# shellcheck disable=SC1090
source "${local_setup}"
set -u

printf '\nEnvironment check: PASS\n'
printf 'Model   : %s\n' "${model_path}"
printf 'Camera  : %s\n' "${camera_device}"
printf 'ROS 2   : %s\n' "${ROS_DISTRO}"
printf 'Device  : %s\n' "${DEVICE}"
printf 'Captures: %s\n' "${CAPTURE_OUTPUT_DIR}"

if is_true "${check_only}"; then
    exit 0
fi

mkdir -p "${output_dir}"
if [[ -f "${pid_file}" ]]; then
    old_pid="$(<"${pid_file}")"
    if [[ "${old_pid}" =~ ^[0-9]+$ ]] && kill -0 "${old_pid}" 2>/dev/null; then
        fail "Detector is already running with PID ${old_pid}. Use stop_detector.sh first."
    fi
    rm -f -- "${pid_file}"
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
log_file="${output_dir}/detector-${timestamp}.log"
detector_pid=""

cleanup() {
    status=$?
    trap - INT TERM EXIT

    if [[ -n "${detector_pid}" ]] && kill -0 "${detector_pid}" 2>/dev/null; then
        kill -INT "${detector_pid}" 2>/dev/null || true
        wait "${detector_pid}" 2>/dev/null || true
    fi
    if [[ -f "${pid_file}" ]] && [[ "$(<"${pid_file}")" == "${detector_pid}" ]]; then
        rm -f -- "${pid_file}"
    fi
    exit "${status}"
}
trap cleanup INT TERM EXIT

run_args=(
    --ros-args
    -p "model_path:=${model_path}"
    -p "source_mode:=camera"
    -p "camera_source:=${camera_device}"
    -p "camera_backend:=${CAMERA_BACKEND}"
    -p "camera_fps:=${CAMERA_FPS}"
    -p "camera_frame_id:=${CAMERA_FRAME_ID}"
    -p "camera_reconnect_interval:=${CAMERA_RECONNECT_INTERVAL}"
    -p "camera_read_failure_threshold:=${CAMERA_READ_FAILURE_THRESHOLD}"
    -p "device:=${DEVICE}"
    -p "imgsz:=${IMGSZ}"
    -p "conf_threshold:=${CONF_THRESHOLD}"
    -p "iou_threshold:=${IOU_THRESHOLD}"
    -p "publish_annotated_image:=${PUBLISH_ANNOTATED_IMAGE}"
    -p "show_window:=${OPEN_VIEWER}"
    -p "capture_output_dir:=${CAPTURE_OUTPUT_DIR}"
    -p "record_on_start:=${RECORD_ON_START}"
    -p "recording_fps:=${RECORDING_FPS}"
    -p "recording_codec:=${RECORDING_CODEC}"
)

printf '\nStarting YOLO detector. Press Ctrl+C to stop.\n'
printf 'Log: %s\n\n' "${log_file}"
ros2 run yolo_detector yolo_detector_node "${run_args[@]}" \
    > >(tee -a "${log_file}") 2>&1 &
detector_pid=$!
printf '%s\n' "${detector_pid}" > "${pid_file}"

if [[ -t 0 ]]; then
    recording_enabled="${RECORD_ON_START}"
    printf '\nControls: P=photo, R=start/stop recording, Q=quit\n'
    while kill -0 "${detector_pid}" 2>/dev/null; do
        key=""
        if IFS= read -rsn1 -t 0.2 key; then
            case "${key,,}" in
                p)
                    ros2 service call /yolo/save_snapshot \
                        std_srvs/srv/Trigger "{}" || true
                    ;;
                r)
                    if is_true "${recording_enabled}"; then
                        desired=false
                    else
                        desired=true
                    fi
                    if ros2 service call /yolo/set_recording \
                        std_srvs/srv/SetBool "{data: ${desired}}"; then
                        recording_enabled="${desired}"
                    fi
                    ;;
                q)
                    kill -INT "${detector_pid}" 2>/dev/null || true
                    break
                    ;;
            esac
        fi
    done
fi

set +e
wait "${detector_pid}"
status=$?
set -e
detector_pid=""
rm -f -- "${pid_file}"
exit "${status}"
