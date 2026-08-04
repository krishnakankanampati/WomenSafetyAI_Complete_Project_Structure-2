# Build stage
FROM node:20-alpine AS build
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
# VITE_BACKEND_URL is deliberately left unset - the app then defaults to
# same-origin ("") in production, which is correct here since the
# host-level nginx in front of every deployment (Docker Compose or k8s)
# reverse-proxies /api and /auth to the backend on that same origin.
RUN npm run build

# Serve stage - small, no Node/npm in the final image
FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY deployment/docker/frontend.nginx.conf.template /etc/nginx/templates/default.conf.template
ENV BACKEND_UPSTREAM=backend:8000
EXPOSE 80
