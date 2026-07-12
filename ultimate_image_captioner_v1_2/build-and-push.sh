#!/usr/bin/env bash
set -Eeuo pipefail

IMAGE_REPOSITORY="${IMAGE_REPOSITORY:-markwelshboy/ultimate_image_captioner}"
IMAGE_VERSION="${IMAGE_VERSION:-1.2}"
PLATFORM="${PLATFORM:-linux/amd64}"
PUSH="${PUSH:-true}"

build_args=(
  docker buildx build
  --platform "$PLATFORM"
  --build-arg "IMAGE_VERSION=$IMAGE_VERSION"
  --tag "$IMAGE_REPOSITORY:$IMAGE_VERSION"
  --tag "$IMAGE_REPOSITORY:latest"
)

if [[ "$PUSH" == "true" ]]; then
  build_args+=(--push)
else
  build_args+=(--load)
fi

build_args+=(.)

printf 'Running:'
printf ' %q' "${build_args[@]}"
printf '\n'
exec "${build_args[@]}"
