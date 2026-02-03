FROM node:22.19-slim

# Set working directory
WORKDIR /app

# Copy only package files first (better caching)
COPY package.json package-lock.json* ./

RUN npm install react-scripts
RUN npm install web-vitals


# Install dependencies
RUN npm i

# Copy rest of the app
COPY . .

# Expose app port (change if needed)
ENV PUBLIC_URL=/genbi

EXPOSE 3000

# Start the app
CMD ["npm", "start","--","--host","0.0.0.0"]
