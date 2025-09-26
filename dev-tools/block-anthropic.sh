#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-api.anthropic.com}"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Call the generic pf-manager script
exec "$SCRIPT_DIR/pf-manager.sh" "$HOST" "block"