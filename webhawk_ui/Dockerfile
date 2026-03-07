# Use a Ruby base image
FROM ruby:3.2

# Install system dependencies
RUN apt-get update -qq && apt-get install -y \
  build-essential \
  libpq-dev \
  nodejs \
  curl \
  git \
  sqlite3

# Set working directory
WORKDIR /app

# Copy Gem files and install dependencies
COPY Gemfile Gemfile.lock ./
RUN gem install bundler && bundle install

# Copy rest of the app
COPY . .

# Expose port
EXPOSE 8080

# Default command (overridden in docker-compose)
CMD ["bin/rails", "server", "-b", "0.0.0.0", "-p", "8080"]
