# Stage 1: Build the React UI
FROM node:24-alpine AS builder

WORKDIR /app

# Install dependencies using the lockfile
COPY package.json package-lock.json ./
RUN npm ci

# Copy the application source and build it
COPY . .
RUN npm run build

# Stage 2: Serve the compiled UI
FROM nginx:alpine

# Copy the production build into Nginx
COPY --from=builder /app/dist /usr/share/nginx/html

# Expose the web server port
EXPOSE 80

# Start Nginx
CMD ["nginx", "-g", "daemon off;"]
