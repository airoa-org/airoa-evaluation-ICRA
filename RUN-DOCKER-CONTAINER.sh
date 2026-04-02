#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-up}"

is_true() {
  case "${1:-}" in
    1 | true | TRUE | True | yes | YES | on | ON | y | Y)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

resolve_hsr_ip() {
  if [[ -n "${HSR_IP:-}" ]]; then
    echo "${HSR_IP}"
    return 0
  fi

  if [[ -n "${ROBOT_NAME:-}" ]]; then
    local name="${ROBOT_NAME}"
    local ip=""

    ip="$(getent hosts "${name}" | awk '{ print $1 }' || true)"
    if [[ -n "${ip}" ]]; then
      echo "${ip}"
      return 0
    fi

    if command -v avahi-resolve >/dev/null 2>&1; then
      ip="$(avahi-resolve -4 --name "${name}.local" 2>/dev/null | cut -f 2 || true)"
      if [[ -n "${ip}" ]]; then
        echo "${ip}"
        return 0
      fi
    fi
  fi

  return 1
}

ensure_ros_master_uri() {
  if [[ -n "${ROS_MASTER_URI:-}" ]]; then
    return 0
  fi

  local hsr_ip=""
  if hsr_ip="$(resolve_hsr_ip)"; then
    export ROS_MASTER_URI="http://${hsr_ip}:11311"
    echo "[INFO] ROS_MASTER_URI is not set. Using ${ROS_MASTER_URI}"
    return 0
  fi

  echo "[ERROR] ROS_MASTER_URI is not set."
  echo "Set one of the following before running:"
  echo "  1) export ROS_MASTER_URI=http://<HSR_IP>:11311"
  echo "  2) export HSR_IP=<HSR_IP>"
  echo "  3) export ROBOT_NAME=<hsrbxxx>"
  exit 1
}

ensure_ros_ip() {
  if [[ -n "${ROS_IP:-}" ]]; then
    return 0
  fi

  local ips=()
  if command -v ifconfig >/dev/null 2>&1; then
    while IFS= read -r ip; do
      [[ -z "${ip}" ]] && continue
      [[ "${ip}" == 127.* ]] && continue
      ips+=("${ip}")
    done < <(ifconfig | awk '/inet / {print $2}')
  else
    while IFS= read -r ip; do
      [[ -z "${ip}" ]] && continue
      [[ "${ip}" == 127.* ]] && continue
      ips+=("${ip}")
    done < <(ip -4 -o addr show | awk '{print $4}' | cut -d/ -f1)
  fi

  if [[ ${#ips[@]} -eq 0 ]]; then
    echo "[ERROR] ROS_IP is not set and no candidate host IP was found."
    exit 1
  fi

  if [[ ! -t 0 ]]; then
    echo "[ERROR] ROS_IP is not set and this shell is non-interactive."
    echo "Set it explicitly: export ROS_IP=<YOUR_HOST_IP>"
    exit 1
  fi

  echo "ROS_IP is not set. Choose host IP for ROS nodes:"
  select ip in "${ips[@]}"; do
    if [[ -n "${ip:-}" ]]; then
      export ROS_IP="${ip}"
      echo "[INFO] Using ROS_IP=${ROS_IP}"
      break
    fi
  done
}

ensure_paths() {
  : "${POLICY_CACHE_DIR:=${PWD}/.docker_cache/policy_cache}"
  : "${HF_CACHE_DIR:=${PWD}/.docker_cache/hf}"
  : "${ROSBAG_DIR:=${PWD}/datasets/rosbags}"
  : "${POLICY_CHECKPOINT_PATH:?Set POLICY_CHECKPOINT_PATH to your checkpoint full path}"

  export POLICY_CACHE_DIR
  export HF_CACHE_DIR
  export ROSBAG_DIR
  export POLICY_CHECKPOINT_PATH

  if [[ ! -d "${POLICY_CHECKPOINT_PATH}" ]]; then
    echo "[ERROR] POLICY_CHECKPOINT_PATH does not exist: ${POLICY_CHECKPOINT_PATH}"
    exit 1
  fi

  mkdir -p "${POLICY_CACHE_DIR}" "${HF_CACHE_DIR}" "${ROSBAG_DIR}"
}

print_env_summary() {
  echo "[INFO] ROS_MASTER_URI=${ROS_MASTER_URI:-}"
  echo "[INFO] ROS_IP=${ROS_IP:-}"
  echo "[INFO] TEST_MODE=${TEST_MODE:-true}"
  echo "[INFO] POLICY_CONFIG_NAME=${POLICY_CONFIG_NAME:-pi0_hsr_airoa-moma}"
  echo "[INFO] POLICY_PYTORCH_DEVICE=${POLICY_PYTORCH_DEVICE:-}"
  echo "[INFO] POLICY_CHECKPOINT_PATH=${POLICY_CHECKPOINT_PATH:-}"
  echo "[INFO] POLICY_CACHE_DIR=${POLICY_CACHE_DIR:-}"
  echo "[INFO] HF_CACHE_DIR=${HF_CACHE_DIR:-}"
  echo "[INFO] ROSBAG_DIR=${ROSBAG_DIR:-}"
  echo "[INFO] POLICY_SERVER_HOST=${POLICY_SERVER_HOST:-127.0.0.1}"
  echo "[INFO] POLICY_SERVER_PORT=${POLICY_SERVER_PORT:-8000}"
}

cmd_up() {
  if is_true "${TEST_MODE:-true}"; then
    export ROS_MASTER_URI="${ROS_MASTER_URI:-http://127.0.0.1:11311}"
    export ROS_IP="${ROS_IP:-127.0.0.1}"
    echo "[INFO] TEST_MODE=true: skipping HSR network checks."
    echo "[INFO] Using ROS_MASTER_URI=${ROS_MASTER_URI}"
    echo "[INFO] Using ROS_IP=${ROS_IP}"
  else
    ensure_ros_master_uri
    ensure_ros_ip
  fi
  ensure_paths
  print_env_summary
  docker compose up --build -d
  echo "[INFO] Containers started."
  echo "[INFO] Next step: ./RUN-DOCKER-CONTAINER.sh shell"
}

cmd_shell() {
  local container_name="${HSR_CLIENT_CONTAINER_NAME:-airoa_hsr_client}"
  if ! docker ps --format '{{.Names}}' | grep -qx "${container_name}"; then
    echo "[ERROR] Container '${container_name}' is not running."
    echo "Start it first with: ./RUN-DOCKER-CONTAINER.sh up"
    exit 1
  fi
  docker exec -it \
    -e POLICY_SERVER_HOST="${POLICY_SERVER_HOST:-127.0.0.1}" \
    -e POLICY_SERVER_PORT="${POLICY_SERVER_PORT:-8000}" \
    -e POLICY_SERVER_API_KEY="${POLICY_SERVER_API_KEY:-}" \
    -e TEST_MODE="${TEST_MODE:-true}" \
    "${container_name}" bash -lc '
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash
echo "[INFO] ROS environment loaded."
echo "[INFO] Run inside this shell:"
echo "roslaunch hsr_policy_client hsr_policy_client.launch"
exec bash
'
}

cmd_launch() {
  echo "[INFO] Opening hsr_client shell. Run roslaunch inside the container."
  cmd_shell
}

cmd_down() {
  local policy_container="${POLICY_SERVER_CONTAINER_NAME:-airoa_policy_server}"
  local client_container="${HSR_CLIENT_CONTAINER_NAME:-airoa_hsr_client}"
  local found=false

  for container in "${client_container}" "${policy_container}"; do
    if docker ps -a --format '{{.Names}}' | grep -qx "${container}"; then
      found=true
      echo "[INFO] Removing container: ${container}"
      docker rm -f "${container}" >/dev/null
    fi
  done

  if [[ "${found}" == "false" ]]; then
    echo "[INFO] No managed containers found."
  else
    echo "[INFO] Containers stopped and removed."
  fi
}

cmd_logs() {
  local service="${2:-}"
  local policy_container="${POLICY_SERVER_CONTAINER_NAME:-airoa_policy_server}"
  local client_container="${HSR_CLIENT_CONTAINER_NAME:-airoa_hsr_client}"

  if [[ -n "${service}" ]]; then
    local container_name="${service}"
    case "${service}" in
      policy_server)
        container_name="${policy_container}"
        ;;
      hsr_client)
        container_name="${client_container}"
        ;;
    esac
    docker logs -f "${container_name}"
    return
  fi

  echo "[INFO] Showing logs for ${policy_container} and ${client_container} (Ctrl+C to stop)."
  docker logs -f "${policy_container}" &
  local pid1=$!
  docker logs -f "${client_container}" &
  local pid2=$!
  trap 'kill ${pid1} ${pid2} 2>/dev/null || true' INT TERM
  wait
}

case "${MODE}" in
  up)
    cmd_up
    ;;
  shell)
    cmd_shell
    ;;
  launch)
    cmd_launch
    ;;
  down)
    cmd_down
    ;;
  logs)
    cmd_logs "$@"
    ;;
  *)
    echo "Usage: $0 [up|shell|launch|logs [service]|down]"
    exit 1
    ;;
esac
