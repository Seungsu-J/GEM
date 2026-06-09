#!/bin/bash
# Automated build script for both Debug and Release versions

set -e  # Exit on error

echo "Building GEM in both Debug and Release modes..."

# Build Debug version
echo ""
echo "=== Building Debug version ==="
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build/debug -j8

# Build Release version
echo ""
echo "=== Building Release version ==="
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release -j8

echo ""
echo "=== Copying debug executable to build/ ==="
cp build/debug/GEM build/GEM

echo ""
echo "=== Build Complete ==="
echo "Debug executable:   ./build/GEM (and ./build/debug/GEM)"
echo "Release executable: ./build/release/GEM"