#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="cv-dp-camera-scheduler.service"
MANIFEST=""
PYTHON_EXECUTABLE=""
START_NOW=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest)
            MANIFEST="${2:-}"
            shift 2
            ;;
        --python-executable)
            PYTHON_EXECUTABLE="${2:-}"
            shift 2
            ;;
        --service-name)
            SERVICE_NAME="${2:-}"
            shift 2
            ;;
        --start-now)
            START_NOW=true
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ ! "$SERVICE_NAME" =~ ^cv-dp[A-Za-z0-9_.@-]*\.service$ ]]; then
    echo "Service name must begin with cv-dp and end with .service: $SERVICE_NAME" >&2
    exit 2
fi

if [[ $EUID -ne 0 ]]; then
    echo "This installer must run as root (use sudo or the GUI authentication prompt)." >&2
    exit 1
fi
if [[ -z "$MANIFEST" || ! -f "$MANIFEST" ]]; then
    echo "Deployment manifest does not exist: $MANIFEST" >&2
    exit 1
fi
if [[ -z "$PYTHON_EXECUTABLE" || ! -x "$PYTHON_EXECUTABLE" ]]; then
    echo "Python executable does not exist or is not executable: $PYTHON_EXECUTABLE" >&2
    exit 1
fi
if ! command -v systemctl >/dev/null 2>&1; then
    echo "systemctl is required to install the Jetson startup service." >&2
    exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/..")"
RUNNER="$(realpath "$PROJECT_ROOT/scripts/scheduled_runner.py")"
MANIFEST="$(realpath "$MANIFEST")"
PYTHON_DIR="$(cd -- "$(dirname -- "$PYTHON_EXECUTABLE")" && pwd -P)"
PYTHON_EXECUTABLE="$PYTHON_DIR/$(basename -- "$PYTHON_EXECUTABLE")"

RUN_UID="${PKEXEC_UID:-}"
if [[ -n "$RUN_UID" ]]; then
    RUN_USER="$(getent passwd "$RUN_UID" | cut -d: -f1)"
elif [[ -n "${SUDO_USER:-}" && "${SUDO_USER}" != "root" ]]; then
    RUN_USER="$SUDO_USER"
else
    RUN_USER="$(stat -c '%U' "$PROJECT_ROOT")"
fi
if [[ -z "$RUN_USER" || "$RUN_USER" == "UNKNOWN" ]]; then
    echo "Could not determine the non-root user that should run the camera service." >&2
    exit 1
fi
RUN_GROUP="$(id -gn "$RUN_USER")"
RUN_HOME="$(getent passwd "$RUN_USER" | cut -d: -f6)"

systemd_quote() {
    local value="$1"
    value="${value//\\/\\\\}"
    value="${value//\"/\\\"}"
    value="${value//%/%%}"
    printf '"%s"' "$value"
}

systemd_path() {
    local value="$1"
    value="${value//\\/\\\\}"
    value="${value//%/%%}"
    value="${value// /\\x20}"
    printf '%s' "$value"
}

UNIT_PATH="/etc/systemd/system/$SERVICE_NAME"
TEMP_UNIT="$(mktemp --suffix=.service)"
trap 'rm -f "$TEMP_UNIT"' EXIT

cat > "$TEMP_UNIT" <<EOF
[Unit]
Description=CV-DP scheduled camera supervisor
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User=$RUN_USER
Group=$RUN_GROUP
WorkingDirectory=$(systemd_path "$PROJECT_ROOT")
Environment=$(systemd_quote "HOME=$RUN_HOME")
Environment=$(systemd_quote "PYTHONUNBUFFERED=1")
ExecStart=$(systemd_quote "$PYTHON_EXECUTABLE") $(systemd_quote "$RUNNER") --manifest $(systemd_quote "$MANIFEST") --headless
Restart=on-failure
RestartSec=15
TimeoutStopSec=45
KillMode=control-group

[Install]
WantedBy=multi-user.target
EOF

if command -v systemd-analyze >/dev/null 2>&1; then
    echo "Validating generated systemd unit..."
    systemd-analyze verify "$TEMP_UNIT"
fi

install -o root -g root -m 0644 "$TEMP_UNIT" "$UNIT_PATH"
systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
systemctl reset-failed "$SERVICE_NAME" 2>/dev/null || true

LOAD_STATE="$(systemctl show "$SERVICE_NAME" --property=LoadState --value)"
if [[ "$LOAD_STATE" != "loaded" ]]; then
    echo "systemd rejected $SERVICE_NAME (LoadState=$LOAD_STATE)." >&2
    systemctl status "$SERVICE_NAME" --no-pager --full >&2 || true
    exit 1
fi
if ! systemctl is-enabled "$SERVICE_NAME" >/dev/null 2>&1; then
    echo "systemd did not enable $SERVICE_NAME for boot." >&2
    exit 1
fi
if [[ "$START_NOW" == "true" ]]; then
    systemctl restart "$SERVICE_NAME"
    if ! systemctl is-active "$SERVICE_NAME" >/dev/null 2>&1; then
        echo "The service was installed but did not remain active." >&2
        systemctl status "$SERVICE_NAME" --no-pager --full >&2 || true
        exit 1
    fi
fi

echo "Installed and enabled systemd service: $SERVICE_NAME"
echo "Run user: $RUN_USER"
echo "Manifest: $MANIFEST"
if [[ "$START_NOW" == "true" ]]; then
    echo "The scheduler service was started."
fi
echo "The scheduler will start on the next boot."
