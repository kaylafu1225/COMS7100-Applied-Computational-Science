#!/bin/bash

echo "Loading Project 1a..."

if [[ "$OSTYPE" == "darwin"* ]]; then
    SDK="$(xcrun --show-sdk-path)"
    clang++ -std=c++17 \
        -isysroot "$SDK" \
        -isystem "$SDK/usr/include/c++/v1" \
        P1a.cpp -o project1a
elif command -v g++ >/dev/null 2>&1; then
    g++ -std=c++17 P1a.cpp -o project1a
elif command -v clang++ >/dev/null 2>&1; then
    clang++ -std=c++17 P1a.cpp -o project1a
else
    echo "Error: No C++17 compiler found."
    exit 1
fi

if [ $? -ne 0 ]; then
    echo "Compilation failed."
    exit 1
fi

./project1a