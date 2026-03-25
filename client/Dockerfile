# Blackwell (RTX 5070 Ti) 対応: CUDA 12.8.1
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu20.04
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git git-lfs curl wget lsb-release ca-certificates \
    python3 python3-venv python3-pip \
    qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools \
    && rm -rf /var/lib/apt/lists/*

# ROS Noetic
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN apt-get update && apt-get install -y --no-install-recommends ros-noetic-desktop-full && \
    rm -rf /var/lib/apt/lists/*

# ROS build tools (install after ROS repo is enabled to avoid package conflicts).
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool && \
    (apt-get install -y --no-install-recommends catkin-tools || \
     apt-get install -y --no-install-recommends python3-catkin-tools || \
     python3 -m pip install --no-cache-dir catkin-tools) && \
    rm -rf /var/lib/apt/lists/*

# HSR/TMC packages
RUN sh -c 'echo "deb [arch=amd64] https://hsr-user:jD3k4G2e@packages.hsr.io/ros/ubuntu `lsb_release -cs` main" > /etc/apt/sources.list.d/tmc.list' && \
    sh -c 'echo "deb [arch=amd64] https://hsr-user:jD3k4G2e@packages.hsr.io/tmc/ubuntu `lsb_release -cs` multiverse main" >> /etc/apt/sources.list.d/tmc.list' && \
    sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list' && \
    wget https://hsr-user:jD3k4G2e@packages.hsr.io/tmc.key -O - | apt-key add - && \
    wget https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc -O - | apt-key add - && \
    wget https://packages.osrfoundation.org/gazebo.key -O - | apt-key add - && \
    mkdir -p /etc/apt/auth.conf.d && \
    /bin/echo -e "machine packages.hsr.io\nlogin hsr-user\npassword jD3k4G2e" >/etc/apt/auth.conf.d/auth.conf && \
    /bin/echo -e "Package: ros-noetic-laser-ortho-projector\nPin: version 0.3.3*\nPin-Priority: 1001\n\nPackage: ros-noetic-laser-scan-matcher\nPin: version 0.3.3*\nPin-Priority: 1001\n\nPackage: ros-noetic-laser-scan-sparsifier\nPin: version 0.3.3*\nPin-Priority: 1001\n\nPackage: ros-noetic-laser-scan-splitter\nPin: version 0.3.3*\nPin-Priority: 1001\n\nPackage: ros-noetic-ncd-parser\nPin: version 0.3.3*\nPin-Priority: 1001\n\nPackage: ros-noetic-polar-scan-matcher\nPin: version 0.3.3*\nPin-Priority: 1001\n\nPackage: ros-noetic-scan-to-cloud-converter\nPin: version 0.3.3*\nPin-Priority: 1001\n\nPackage: ros-noetic-scan-tools\nPin: version 0.3.3*\nPin-Priority: 1001" > /etc/apt/preferences && \
    apt-get update && apt-get install -y --no-install-recommends ros-noetic-tmc-desktop-full && \
    rm -rf /var/lib/apt/lists/*

RUN rosdep init || true && rosdep update || true

WORKDIR /workspace
COPY deploy ./deploy
COPY packages/policy-client ./packages/policy-client

# Build catkin workspace with copied ROS packages.
RUN mkdir -p /root/catkin_ws/src && \
    cp -a /workspace/deploy/hsr_policy_client /root/catkin_ws/src/hsr_policy_client && \
    source /opt/ros/noetic/setup.bash && \
    cd /root/catkin_ws && catkin build

# Python 3.11 venv for hsr_policy.py (shebang points to /home/policy/.venv/bin/python3).
RUN uv venv --python 3.11 --seed /home/policy/.venv && \
    /home/policy/.venv/bin/pip install --no-cache-dir --upgrade pip && \
    /home/policy/.venv/bin/pip install --no-cache-dir numpy==1.26.4 opencv-python pyyaml rospkg catkin_pkg && \
    /home/policy/.venv/bin/pip install --no-cache-dir -e /workspace/packages/policy-client

COPY client/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV ROS_DISTRO=noetic
ENV ROS_HOME=/root/.ros

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash", "-lc", "sleep infinity"]
