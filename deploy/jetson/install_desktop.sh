#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
start_script="${script_dir}/start_detector.sh"
stop_script="${script_dir}/stop_detector.sh"
applications_dir="${XDG_DATA_HOME:-${HOME}/.local/share}/applications"
application_file="${applications_dir}/learn-yolo.desktop"

chmod +x "${start_script}" "${stop_script}" "$0"
mkdir -p "${applications_dir}"

printf '%s\n' \
    '[Desktop Entry]' \
    'Type=Application' \
    'Name=LEARN YOLO Detector' \
    'Comment=Start YOLO USB camera detection and ROS 2 output' \
    "Exec=${start_script} --view" \
    'Icon=camera-video' \
    'Terminal=true' \
    'Categories=Development;Robotics;' \
    > "${application_file}"
chmod +x "${application_file}"

desktop_dir=""
if command -v xdg-user-dir >/dev/null 2>&1; then
    desktop_dir="$(xdg-user-dir DESKTOP 2>/dev/null || true)"
fi
if [[ -z "${desktop_dir}" && -d "${HOME}/Desktop" ]]; then
    desktop_dir="${HOME}/Desktop"
fi

if [[ -n "${desktop_dir}" && -d "${desktop_dir}" ]]; then
    desktop_file="${desktop_dir}/LEARN-YOLO.desktop"
    install -m 755 "${application_file}" "${desktop_file}"
    gio set "${desktop_file}" metadata::trusted true >/dev/null 2>&1 || true
    printf 'Desktop shortcut installed: %s\n' "${desktop_file}"
else
    printf 'Desktop directory was not found; application launcher only was installed.\n'
fi

printf 'Application launcher installed: %s\n' "${application_file}"
printf 'You can also run: %s\n' "${start_script}"
