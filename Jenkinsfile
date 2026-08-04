pipeline {
    agent any

    environment {
        DOCKERHUB_CREDS = credentials('dockerhub-creds')
        KUBECONFIG      = credentials('k3s-kubeconfig')
        IMAGE_BACKEND   = 'krishna3242/wsai-backend'
        IMAGE_FRONTEND  = 'krishna3242/wsai-frontend'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Build images') {
            steps {
                sh "docker build -t ${IMAGE_BACKEND}:${BUILD_NUMBER} -t ${IMAGE_BACKEND}:latest -f deployment/docker/backend.Dockerfile ."
                sh "docker build -t ${IMAGE_FRONTEND}:${BUILD_NUMBER} -t ${IMAGE_FRONTEND}:latest -f deployment/docker/frontend.Dockerfile ."
            }
        }

        stage('Push images') {
            steps {
                sh 'echo "$DOCKERHUB_CREDS_PSW" | docker login -u "$DOCKERHUB_CREDS_USR" --password-stdin'
                sh "docker push ${IMAGE_BACKEND}:${BUILD_NUMBER}"
                sh "docker push ${IMAGE_BACKEND}:latest"
                sh "docker push ${IMAGE_FRONTEND}:${BUILD_NUMBER}"
                sh "docker push ${IMAGE_FRONTEND}:latest"
            }
        }

        stage('Deploy to k3s') {
            steps {
                // Tag with BUILD_NUMBER (not :latest) so the rollout is tied to
                // this exact build and `kubectl rollout undo` can step back to
                // a specific prior image if a deploy misbehaves.
                sh "kubectl -n wsai set image deployment/wsai-backend backend=${IMAGE_BACKEND}:${BUILD_NUMBER}"
                sh "kubectl -n wsai set image deployment/wsai-frontend frontend=${IMAGE_FRONTEND}:${BUILD_NUMBER}"
                sh 'kubectl -n wsai rollout status deployment/wsai-backend --timeout=280s'
                sh 'kubectl -n wsai rollout status deployment/wsai-frontend --timeout=90s'
            }
        }
    }

    post {
        always {
            sh 'docker logout || true'
        }
    }
}
