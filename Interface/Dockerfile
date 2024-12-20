# Stage 1: Builder
FROM node:21-bookworm AS builder
WORKDIR /app

# Copy the package.json, package-lock.json and next.config.mjs files from the 'Interface' directory to the working directory
COPY ./Interface/package.json ./package.json
COPY ./Interface/package-lock.json ./package-lock.json
COPY ./Interface/next.config.mjs ./

# Install dependencies
RUN npm install && npm cache clean --force

# Copy all contents of interface folder into container
COPY ./Interface .

# Builds the Next.js application
RUN npm run build

# Stage 2: Runner
FROM node:21-bookworm AS runner
WORKDIR /app

# Install necessary tools for SSH
RUN apt update && apt install -y vim inetutils-ping inetutils-telnet openssh-client

# Configure the known_hosts file
ARG NEXT_PUBLIC_SSH_SERVER
RUN mkdir -p /root/.ssh && ssh-keyscan -H $NEXT_PUBLIC_SSH_SERVER >> /root/.ssh/known_hosts

# Copy the necessary files from the build phase to the working directory of the container in which the app will run
COPY --from=builder /app .

# Exposes port 3000 for the app
EXPOSE 3000

# The application will be started with the 'entrypoint' statement in "docker-compose.yml".
