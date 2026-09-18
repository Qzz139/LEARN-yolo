#!/usr/bin/env bash
# 依据 PID 文件停止检测进程，核实进程命令行后发送可清理资源的 SIGINT。
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/../.." && pwd)"
pid_file="${project_root}/outputs/jetson/yolo_detector.pid"

if [[ ! -f "${pid_file}" ]]; then
    printf 'YOLO detector is not running (no PID file).\n'
    exit 0
fi

pid="$(<"${pid_file}")"
if [[ ! "${pid}" =~ ^[0-9]+$ ]] || ! kill -0 "${pid}" 2>/dev/null; then
    rm -f -- "${pid_file}"
    printf 'Removed stale detector PID file.\n'
    exit 0
fi

# PID 可能被操作系统复用，发送信号前先核对是否仍是目标 ROS 进程。
cmdline="$(tr '\0' ' ' < "/proc/${pid}/cmdline")"
if [[ "${cmdline}" != *"ros2"* || "${cmdline}" != *"yolo_detector"* ]]; then
    printf 'Refusing to stop PID %s because it is not the YOLO ROS 2 launch process.\n' "${pid}" >&2
    exit 1
fi

kill -INT "${pid}"
# 最多等待约十秒完成清理；超时仅报告失败，不强制杀死进程。
for _ in {1..50}; do
    if ! kill -0 "${pid}" 2>/dev/null; then
        rm -f -- "${pid_file}"
        printf 'YOLO detector stopped.\n'
        exit 0
    fi
    sleep 0.2
done

printf 'Detector did not stop within 10 seconds; PID %s was not force-killed.\n' "${pid}" >&2
exit 1
