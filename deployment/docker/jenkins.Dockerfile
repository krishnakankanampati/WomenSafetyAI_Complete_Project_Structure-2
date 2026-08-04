# Jenkins needs a docker CLI to run `docker build`/`docker push` steps (talks
# to the HOST's docker daemon via the socket mounted at container-run time,
# not a nested/dind daemon) and a kubectl CLI to deploy to the host's k3s
# cluster - this image just adds both client binaries on top of upstream
# Jenkins.
FROM jenkins/jenkins:lts
USER root
RUN apt-get update && apt-get install -y --no-install-recommends docker.io curl \
    && curl -fsSL -o /usr/local/bin/kubectl "https://dl.k8s.io/release/$(curl -fsSL https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
    && chmod +x /usr/local/bin/kubectl \
    && rm -rf /var/lib/apt/lists/*
USER jenkins
