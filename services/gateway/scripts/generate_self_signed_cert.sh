#!/usr/bin/env sh
set -eu

CERT_DIR="${1:-/certs}"
CERT_FILE="${CERT_DIR%/}/server.crt"
KEY_FILE="${CERT_DIR%/}/server.key"
COMMON_NAME="${2:-localhost}"

mkdir -p "$CERT_DIR"

openssl req -x509 -nodes -newkey rsa:2048 \
  -keyout "$KEY_FILE" \
  -out "$CERT_FILE" \
  -days 365 \
  -subj "/C=RU/ST=Moscow/L=Moscow/O=AI Platform/OU=Gateway/CN=${COMMON_NAME}"

echo "Generated certificate: $CERT_FILE"
echo "Generated key: $KEY_FILE"

