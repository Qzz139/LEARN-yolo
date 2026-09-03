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
  --model-id ID          Select weights/ID/best.pt
  --camera-source VALUE  USB camera index or /dev/video path (default: 0)
  --no-build             Skip colcon build for this start
  --view                 Open /yolo/annotated_image with rqt_image_view
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
        --camera-source)
            (($# >= 2)) || fail "--camera-source requires a value"
            CAMERA_SOURCE="$2"
            shift 2
            ;;
        --no-build)
            BUILD_ON_START=false
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
: "${MODEL_ID:=yolo26s_baseline_v2}"
: "${CAMERA_SOURCE:=0}"
: "${CAMERA_BACKEND:=auto}"
: "${CAMERA_FPS:=30.0}"
: "${CAMERA_FRAME_ID:=camera_optical_frame}"
: "${DEVICE:=0}"
: "${IMGSZ:=640}"
: "${CONF_THRESHOLD:=0.25}"
: "${IOU_THRESHOLD:=0.45}"
: "${PUBLISH_ANNOTATED_IMAGE:=true}"
: "${BUILD_ON_START:=true}"
: "${OPEN_VIEWER:=false}"

ros_setup="/opt/ros/${ROS_DISTRO}/setup.bash"
venv_activate="${VENV_PATH}/bin/activate"
workspace="${project_root}/ros2_ws"
model_path="${project_root}/weights/${MODEL_ID}/best.pt"
output_dir="${project_root}/outputs/jetson"
pid_file="${output_dir}/yolo_detector.pid"

[[ -f "${ros_setup}" ]] || fail "ROS 2 setup not found: ${ros_setup}"
[[ -f "${venv_activate}" ]] || fail "Python environment not found: ${VENV_PATH}"
[[ -d "${workspace}/src/yolo_detector" ]] || fail "ROS 2 package not found in ${workspace}"
[[ -f "${model_path}" ]] || fail "Model not found: ${model_path}. Run git lfs pull after git pull."

if [[ "$(wc -c < "${model_path}")" -lt 1024 ]] || head -n 1 "${model_path}" | grep -q 'git-lfs.github.com/spec'; then
    fail "${model_path} is a Git LFS pointer, not model data. Run: git lfs pull"
fi

if [[ "${CAMERA_SOURCE}" =~ ^[0-9]+$ ]]; then
    camera_device="/dev/video${CAMERA_SOURCE}"
else
    camera_device="${CAMERA_SOURCE}"
fi
[[ -e "${camera_device}" ]] || fail "USB camera node not found: ${camera_device}"
[[ -r "${camera_device}" ]] || fail "USB camera is not readable: ${camera_device}. Check video group permissions."

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

camera_probe="${CAMERA_SOURCE}"
python3 - "${camera_probe}" <<'PY' || fail "Cannot read a frame from USB camera ${camera_device}. Try --camera-source 1."
import sys
import cv2

raw = sys.argv[1]
source = int(raw) if raw.isdigit() else raw
capture = cv2.VideoCapture(source)
try:
    ok, frame = capture.read()
    if not ok or frame is None:
        raise SystemExit(1)
    print(f"USB camera: PASS ({frame.shape[1]}x{frame.shape[0]})")
finally:
    capture.release()
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
viewer_pid=""

cleanup() {
    status=$?
    trap - INT TERM EXIT
    if [[ -n "${viewer_pid}" ]] && kill -0 "${viewer_pid}" 2>/dev/null; then
        kill "${viewer_pid}" 2>/dev/null || true
    fi
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
    -p "device:=${DEVICE}"
    -p "imgsz:=${IMGSZ}"
    -p "conf_threshold:=${CONF_THRESHOLD}"
    -p "iou_threshold:=${IOU_THRESHOLD}"
    -p "publish_annotated_image:=${PUBLISH_ANNOTATED_IMAGE}"
)

printf '\nStarting YOLO detector. Press Ctrl+C to stop.\n'
printf 'Log: %s\n\n' "${log_file}"
ros2 run yolo_detector yolo_detector_node "${run_args[@]}" \
    > >(tee -a "${log_file}") 2>&1 &
detector_pid=$!
printf '%s\n' "${detector_pid}" > "${pid_file}"

if is_true "${OPEN_VIEWER}"; then
    if command -v rqt_image_view >/dev/null 2>&1 && [[ -n "${DISPLAY:-}" ]]; then
        (sleep 4; rqt_image_view /yolo/annotated_image) &
        viewer_pid=$!
    else
        printf 'Viewer was requested but rqt_image_view or DISPLAY is unavailable.\n' >&2
    fi
fi

set +e
wait "${detector_pid}"
status=$?
set -e
detector_pid=""
rm -f -- "${pid_file}"
exit "${status}"
