#!/bin/bash
# Fast-Calib 启动脚本
# 用法: bash start.sh [端口号]

set -e

PORT=${1:-8500}
ADDR=0.0.0.0
MAX_UPLOAD=1024  # MB

echo "=========================================="
echo "  Fast-Calib 雷达-相机外参标定"
echo "=========================================="
echo "  地址: http://localhost:${PORT}"
echo "  上传限制: ${MAX_UPLOAD}MB"
echo "=========================================="

# 检查权重文件
WEIGHT_PATH="/tmp/best_model.pth"
if [ ! -f "${WEIGHT_PATH}" ]; then
    echo "[WARN] SECOND 权重不存在: ${WEIGHT_PATH}"
    echo "       请将 best_model.pth 放到 ${WEIGHT_PATH}"
    echo "       否则将使用手动 ROI 模式"
fi

# 启动 Streamlit
streamlit run app.py \
    --server.address ${ADDR} \
    --server.port ${PORT} \
    --server.maxUploadSize ${MAX_UPLOAD} \
    --server.headless true
