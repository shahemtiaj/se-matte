#!/bin/zsh
# Builds dist/SE-Matte-<version>-mac.zip from the committed sources plus the prebuilt native helper.
#   scripts/package_release.sh            version from panel/CSXS/manifest.xml
set -e
cd "${0:A:h}/.."
VER=$(sed -n 's/.*ExtensionBundleVersion="\([^"]*\)".*/\1/p' panel/CSXS/manifest.xml)
NAME="SE-Matte-$VER-mac"
OUT="dist/$NAME"
rm -rf "$OUT" "dist/$NAME.zip"
mkdir -p "$OUT"

./engine/native/build_mac.sh >/dev/null 2>&1                     # universal arm64 + x86_64 helper
git archive HEAD | tar -x -C "$OUT"                            # committed files only
rm -f "$OUT/panel/.debug" "$OUT/.gitignore"                     # no DevTools port in release builds
rm -rf "$OUT/scripts"
mkdir -p "$OUT/engine/bin/mac"
cp engine/bin/mac/semfast "$OUT/engine/bin/mac/semfast"

(cd dist && ditto -c -k --norsrc --noextattr --keepParent "$NAME" "$NAME.zip")
rm -rf "$OUT"
shasum -a 256 "dist/$NAME.zip"
