#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="cv-dp-camera-scheduler.service"
OPERATION="status"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --operation)
            OPERATION="${2:-}"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

case "$OPERATION" in
    status)
        systemctl status "$SERVICE_NAME" --no-pager
        ;;
    stop)
        if [[ $EUID -ne 0 ]]; then
            echo "Stopping must run as root (use sudo or the GUI authentication prompt)." >&2
            exit 1
        fi
        systemctl stop "$SERVICE_NAME"
        echo "Stopped systemd service: $SERVICE_NAME"
        echo "The service remains enabled and will run again after the next device boot."
        ;;
    remove)
        if [[ $EUID -ne 0 ]]; then
            echo "Removal must run as root (use sudo or the GUI authentication prompt)." >&2
            exit 1
        fi
        if systemctl cat "$SERVICE_NAME" >/dev/null 2>&1; then
            systemctl disable --now "$SERVICE_NAME" || systemctl stop "$SERVICE_NAME" || true
        fi
        rm -f "/etc/systemd/system/$SERVICE_NAME"
        systemctl daemon-reload
        systemctl reset-failed "$SERVICE_NAME" 2>/dev/null || true
        echo "Removed systemd service: $SERVICE_NAME"
        echo "Saved manifests, camera configs, and output data were not deleted."
        ;;
    *)
        echo "Operation must be status, stop, or remove." >&2
        exit 2
        ;;
esac
