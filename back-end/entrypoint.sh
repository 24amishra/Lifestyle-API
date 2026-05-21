#!/bin/sh

echo "Starting Flask app..."
exec gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 120 app:app
