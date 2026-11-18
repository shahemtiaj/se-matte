#!/bin/zsh
# Builds the native Fast-mode helper into engine/bin/mac/semfast (needs Xcode Command Line Tools).
set -e
cd "${0:A:h}"
mkdir -p ../bin/mac
swiftc -O -target arm64-apple-macos14 semfast/main.swift -o ../bin/mac/semfast-arm64
swiftc -O -target x86_64-apple-macos14 semfast/main.swift -o ../bin/mac/semfast-x86_64 2>/dev/null && \
  lipo -create ../bin/mac/semfast-arm64 ../bin/mac/semfast-x86_64 -output ../bin/mac/semfast.new && \
  mv -f ../bin/mac/semfast.new ../bin/mac/semfast && rm -f ../bin/mac/semfast-arm64 ../bin/mac/semfast-x86_64 || \
  mv -f ../bin/mac/semfast-arm64 ../bin/mac/semfast
echo "built $(file ../bin/mac/semfast)"
