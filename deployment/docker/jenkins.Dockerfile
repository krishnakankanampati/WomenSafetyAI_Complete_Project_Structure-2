# Jenkins needs a docker CLI to run `docker build`/`docker push` steps in the
# pipeline. It talks to the HOST's docker daemon via the socket mounted at
# container-run time (not a nested/dind daemon) - this image just adds the
# client binary on top of upstream Jenkins.
FROM jenkins/jenkins:lts
USER root
RUN apt-get update && apt-get install -y --no-install-recommends docker.io \
    && rm -rf /var/lib/apt/lists/*
USER jenkins
