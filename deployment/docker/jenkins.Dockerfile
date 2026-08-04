# Jenkins needs a docker CLI to run `docker build`/`docker push` steps (talks
# to the HOST's docker daemon via the socket mounted at container-run time,
# not a nested/dind daemon) and a kubectl CLI to deploy to the host's k3s
# cluster - this image just adds both client binaries on top of upstream
# Jenkins. Debian's `docker.io` apt package on this base image only ships
# dockerd, not the CLI, so the CLI comes from Docker's static binary release
# instead - simpler than pulling in Docker's whole apt repo for one binary.
FROM jenkins/jenkins:lts
USER root
ENV DOCKER_CLI_VERSION=27.3.1
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && curl -fsSL "https://download.docker.com/linux/static/stable/x86_64/docker-${DOCKER_CLI_VERSION}.tgz" -o /tmp/docker.tgz \
    && tar xzf /tmp/docker.tgz --strip-components=1 -C /usr/local/bin docker/docker \
    && rm /tmp/docker.tgz \
    && curl -fsSL -o /usr/local/bin/kubectl "https://dl.k8s.io/release/$(curl -fsSL https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
    && chmod +x /usr/local/bin/kubectl \
    && rm -rf /var/lib/apt/lists/*
USER jenkins
