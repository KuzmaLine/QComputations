#!/usr/bin/env bash
# ============================================================================
# QComputations — генератор Doxygen-документации
# ----------------------------------------------------------------------------
# Использование:
#   ./generate.sh                 # вариант single (CPU, Intel oneAPI)
#   ./generate.sh cluster         # + ENABLE_CLUSTER ENABLE_MPI
#   ./generate.sh cuda            # CUDA-ядро (__CUDACC__ + .cu)
#   ./generate.sh blocked_cuda    # blocked CUDA
#   ./generate.sh single --no-open
#
# Результат: ../doxygen_html/index.html
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
VARIANT="${1:-single}"
OPEN="${2:-}"   # "--no-open" чтобы не открывать браузер

case "$VARIANT" in
  single)
     PREDEF="ENABLE_ONEAPI"
     FILEPAT="*.cpp *.hpp *.h" ;;
  cluster)
     PREDEF="ENABLE_ONEAPI ENABLE_CLUSTER ENABLE_MPI WITHOUT_NUMPY"
     FILEPAT="*.cpp *.hpp *.h" ;;
  cuda)
     PREDEF="__CUDACC__ ENABLE_CLUSTER"
     FILEPAT="*.cpp *.hpp *.h *.cu" ;;
  blocked_cuda)
     PREDEF="__CUDACC__ ENABLE_CLUSTER"
     FILEPAT="*.cpp *.hpp *.h *.cu" ;;
  -h|--help|help)
     grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  *)
     echo "[error] неизвестный вариант '$VARIANT'. Варианты: single, cluster, cuda, blocked_cuda"
     exit 1 ;;
esac

# --- зависимость: doxygen ---
if ! command -v doxygen >/dev/null 2>&1; then
  echo "[error] doxygen не установлен."
  echo "  Ubuntu/WSL: sudo apt-get install -y doxygen"
  echo "  macOS:      brew install doxygen"
  exit 1
fi

# --- graphviz (опционально; без него диаграммы выключаются) ---
if command -v dot >/dev/null 2>&1; then
  HAVE_DOT="YES"
else
  HAVE_DOT="NO"
  echo "[warn] graphviz 'dot' не найден -> класс-/call-диаграммы отключены."
  echo "       Ubuntu/WSL: sudo apt-get install -y graphviz"
fi

# --- версия (git describe -> PROJECT_NUMBER) ---
if command -v git >/dev/null 2>&1 && git -C "$HERE/.." rev-parse --is-inside-work-tree 2>/dev/null; then
  VER="$(git -C "$HERE/.." describe --tags --always 2>/dev/null || echo dev)"
else
  VER="dev"
fi

# --- подстановка плейсхолдеров -> Doxyfile.generated ---
# ВАЖНО: список FILE_PATTERNS в Doxygen разделяется ПРОБЕЛАМИ (не ";").
GEN="$HERE/Doxyfile.generated"
sed -e "s/@QCOMP_VERSION@/${VER}/" \
    -e "s/@QCOMP_PREDEFINED@/${PREDEF}/" \
    -e "s/@QCOMP_FILE_PATTERNS@/${FILEPAT}/" \
    -e "s/^HAVE_DOT[[:space:]]*=[[:space:]]*\(.*\)/HAVE_DOT = ${HAVE_DOT}/" \
    "$HERE/Doxyfile" > "$GEN"

echo "[info] вариант: ${VARIANT}   version: ${VER}"
echo "[info] PREDEFINED=${PREDEF}"
echo "[info] FILE_PATTERNS=${FILEPAT}"
echo "[info] HAVE_DOT=${HAVE_DOT}"
echo

# --- сборка ---
cd "$HERE"
doxygen "$GEN"

OUT="$HERE/../doxygen_html/index.html"
echo
echo "============================================================"
echo " Готово. HTML: ${OUT}"
echo " Открой:  xdg-open \"${OUT}\""
echo "============================================================"

if [ "$OPEN" != "--no-open" ] && command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$OUT" || true
fi
