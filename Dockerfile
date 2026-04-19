FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy all project files into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8080
EXPOSE 8080

# Grant execution permissions to start.sh
RUN chmod +x start.sh

# Set the CMD to run the startup script
CMD ["./start.sh"]
