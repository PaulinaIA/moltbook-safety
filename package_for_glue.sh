#!/usr/bin/env bash
# Package src/ and config/ into extra_libs.zip for AWS Glue Python Shell.
# Usage: ./package_for_glue.sh [output_dir]
# Output: extra_libs.zip (or <output_dir>/extra_libs.zip)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-$SCRIPT_DIR}"
ZIP_PATH="$OUT_DIR/extra_libs.zip"
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

cd "$SCRIPT_DIR"
mkdir -p "$TMP_DIR/src" "$TMP_DIR/config"
cp -r src/* "$TMP_DIR/src/"
cp -r config/* "$TMP_DIR/config/" 2>/dev/null || true
cp schema_postgres.sql "$TMP_DIR/" 2>/dev/null || true

cd "$TMP_DIR"
zip -rq "$ZIP_PATH" src config schema_postgres.sql 2>/dev/null || zip -rq "$ZIP_PATH" src config
echo "Created $ZIP_PATH"
