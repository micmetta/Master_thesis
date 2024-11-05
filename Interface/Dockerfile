# Stage 1: Builder
FROM node:21-bookworm AS builder
WORKDIR /app

# Copia i file package.json, package-lock.json e next.config.mjs dalla directory 'Interface' alla directory di lavoro
COPY ./Interface/package.json ./package.json
COPY ./Interface/package-lock.json ./package-lock.json
COPY ./Interface/next.config.mjs ./

# Installa le dipendenze
RUN npm install && npm cache clean --force

# Copia tutto il contenuto della cartella interface nel container
COPY ./Interface .

# Costruisce l'applicazione Next.js
RUN npm run build

# Stage 2: Runner
FROM node:21-bookworm AS runner
WORKDIR /app

# Installa strumenti necessari per SSH
RUN apt update && apt install -y vim inetutils-ping inetutils-telnet openssh-client

# Copia la chiave privata SSH dalla root del progetto
COPY .ssh/id_rsa /root/.ssh/id_rsa
RUN chmod 600 /root/.ssh/id_rsa

# Configura il file known_hosts
ARG NEXT_PUBLIC_SSH_SERVER
RUN mkdir -p /root/.ssh && ssh-keyscan -H $NEXT_PUBLIC_SSH_SERVER >> /root/.ssh/known_hosts

# Copia i file necessari dalla fase di build alla directory di lavoro del container nel quale girerà l'app
#COPY --from=builder /app/node_modules ./node_modules
#COPY --from=builder /app/.next ./.next
#COPY --from=builder /app/package.json ./package.json
#COPY --from=builder /app/next.config.mjs ./next.config.mjs
COPY --from=builder /app .

# Espone la porta 3000 per l'app
EXPOSE 3000

# Avvia l'applicazione
CMD ["npm", "start"]
