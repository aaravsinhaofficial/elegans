# Use the official Python slim image as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install uv and other dependencies
RUN pip install --no-cache-dir uv

# Copy the requirements file into the container at /app
COPY pyproject.toml uv.lock /app/
COPY packages/elegans/pyproject.toml /app/packages/elegans/pyproject.toml

# Install project dependencies
RUN uv sync --no-install-project --extra torch --extra pixel

# Copy local project
COPY packages/elegans /app/packages/elegans
COPY scripts /app/scripts

# Install project
RUN uv sync --extra torch --extra pixel
