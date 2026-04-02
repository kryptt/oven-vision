#!/bin/bash
set -euo pipefail

REGISTRY=registry.hr-home.xyz
APP=oven-vision
VERSION=0.10.0

IMG=$REGISTRY/$APP:$VERSION
LATEST=$REGISTRY/$APP:latest

docker buildx build . -t "$IMG"
docker tag "$IMG" "$LATEST"
docker push "$IMG"
docker push "$LATEST"

echo "Pushed $IMG and $LATEST"
