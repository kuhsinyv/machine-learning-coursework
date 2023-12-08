#!/bin/bash

if [ -z "$(find "$1" -type d -name "__pycache__")" ]; then
  echo "No __pycache__ folders found."
else
  echo "All __pycache__ folders found are as follow: "
  find "$1" -type d -name "__pycache__"
  read -p "Do you want to DELETE these __pycache__ folders? (y/n): " confirm
  if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
    find "$1" -type d -name "__pycache__" | xargs rm -r
    echo "Deleted __pycache__ folders."
  fi
fi