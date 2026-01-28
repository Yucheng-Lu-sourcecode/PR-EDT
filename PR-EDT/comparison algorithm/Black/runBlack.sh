#!/bin/bash

# ===================== 核心配置（满足需求：尺寸[1024,2048,4096,8192]，占比0~100%）=====================
SIZES=(1024 2048 4096 8192)  # 固定尺寸列表
MIN_PERCENT=0                 # 最小占比
MAX_PERCENT=100               # 最大占比
N=30                          # 每个任务循环次数
MAX_RETRY=2                   # 8192尺寸失败重试次数
CSV_FILE="result_full_0_to_100.csv"  # 最终汇总的单个CSV文件
PYTHON_SCRIPT="Black.py"

# ===================== 预处理参数（用于计算进度和循环）=====================
TOTAL_SIZE_COUNT=${#SIZES[@]}                   # 总尺寸数（4个）
TOTAL_PERCENT_COUNT=$((MAX_PERCENT - MIN_PERCENT + 1))  # 总占比数（101个）
# 生成占比数组（0,1,2,...,100）
BLACK_PERCENT_ARRAY=()
for ((i=MIN_PERCENT; i<=MAX_PERCENT; i++)); do
    BLACK_PERCENT_ARRAY+=($i)
done

# ===================== 初始化CSV文件（表头：black_percent + 所有尺寸）=====================
echo -n "black_percent" > $CSV_FILE
for size in "${SIZES[@]}"; do
    echo -n ",$size" >> $CSV_FILE
done
echo "" >> $CSV_FILE

# ===================== 辅助函数：打印格式化进度条 =====================
print_overall_progress() {
    local current_percent_idx=$1
    local total_percents=$2
    # 计算整体进度百分比（保留1位小数，依赖bc命令）
    if [ $total_percents -gt 0 ]; then
        local overall_progress=$(echo "scale=1; ($current_percent_idx / $total_percents) * 100" | bc)
    else
        local overall_progress=0.0
    fi
    # 覆盖当前行打印，保持整洁（蓝色字体）
    echo -ne "\033[1;34m📈 整体任务进度：$current_percent_idx/$total_percents ($overall_progress%) | 已处理：占比${BLACK_PERCENT_ARRAY[$((current_percent_idx-1))]}%\033[0m\r"
}

print_size_progress() {
    local current_size_idx=$1
    local total_sizes=$2
    local percent=$3
    # 计算当前占比下的尺寸测试进度
    local size_progress=$(echo "scale=1; ($current_size_idx / $total_sizes) * 100" | bc)
    echo -e "\033[1;36m🔍 占比$percent% 尺寸测试进度：$current_size_idx/$total_sizes ($size_progress%)\033[0m"
}

# ===================== 辅助函数：运行单个任务（独立进程，支持重试）=====================
run_single_task() {
    local size=$1
    local percent=$2
    local n=$3
    local retry_count=0
    local result="0.0"

    while [ $retry_count -le $MAX_RETRY ]; do
        echo -e "  🔄 运行任务（重试$retry_count/$MAX_RETRY）：尺寸 $size，占比 $percent%，循环 $n 次"
        result=$(python3 $PYTHON_SCRIPT $size $percent $n)

        # 判断结果是否有效
        if [ "$result" != "OOM" ] && [ "$result" != "0.0" ]; then
            echo -e "  ✅ 任务成功：尺寸 $size，占比 $percent%，平均耗时 \033[1;32m$result\033[0m 毫秒"
            echo "$result"
            return 0
        elif [ "$result" == "OOM" ]; then
            echo -e "  ❌ 任务OOM失败：尺寸 $size，占比 $percent%（显存不足/无法运行）"
            retry_count=$((retry_count + 1))
            if [ $retry_count -le $MAX_RETRY ]; then
                echo -e "  💤 等待5秒，释放显存后重试..."
                sleep 5
                # 强制清理残留进程（保险）
                pkill -f "python3 $PYTHON_SCRIPT $size $percent $n" 2>/dev/null
            fi
        else
            echo -e "  ✅ 任务完成（无有效数据/全0矩阵）：尺寸 $size，占比 $percent%，结果 $result"
            echo "$result"
            return 0
        fi
    done

    # 多次重试失败，返回OOM标记
    echo "OOM"
    return 1
}

# ===================== 主循环：遍历所有占比+尺寸，独立运行，汇总到单个CSV =====================
echo -e "\033[1;33m=================================================="
echo -e "🚀 开始执行全量测试任务（尺寸：${SIZES[*]}，占比：$MIN_PERCENT~$MAX_PERCENT%）"
echo -e "==================================================\033[0m"
echo -e "📋 任务配置：总占比数=$TOTAL_PERCENT_COUNT，总尺寸数=$TOTAL_SIZE_COUNT，总任务数=$((TOTAL_PERCENT_COUNT*TOTAL_SIZE_COUNT))"
echo -e "📁 结果将汇总写入：$CSV_FILE"
echo -e "--------------------------------------------------\n"

# 遍历每个占比（0~100%）
for percent_idx in "${!BLACK_PERCENT_ARRAY[@]}"; do
    # 计算当前进度索引（从1开始，符合用户习惯）
    local current_percent_idx=$((percent_idx + 1))
    local percent=${BLACK_PERCENT_ARRAY[$percent_idx]}

    # 1. 打印实时整体进度（覆盖当前行，不刷屏）
    print_overall_progress $current_percent_idx $TOTAL_PERCENT_COUNT

    # 2. 打印当前占比的开始提示（分隔符，清晰区分批次）
    echo -e "\n\033[1;35m=================================================="
    echo -e "📌 开始处理第$current_percent_idx/$TOTAL_PERCENT_COUNT 批：占比 $percent%"
    echo -e "==================================================\033[0m"

    # 3. 初始化当前行结果（先写入占比，后续追加各尺寸结果）
    current_row="$percent"
    local current_size_idx=0

    # 4. 遍历每个尺寸，独立运行Python进程（每个组合一个独立进程）
    for size in "${SIZES[@]}"; do
        current_size_idx=$((current_size_idx + 1))
        # 打印当前占比下的尺寸进度
        print_size_progress $current_size_idx $TOTAL_SIZE_COUNT $percent

        # 运行单个任务，捕获结果
        avg_time=$(run_single_task $size $percent $N)
        
        # 5. 将尺寸结果追加到当前行
        if [ "$avg_time" == "OOM" ]; then
            current_row="$current_row,OOM"
        else
            current_row="$current_row,$avg_time"
        fi
        echo -e "  --------------------------------------------------\n"
    done

    # 6. 将当前占比的所有尺寸结果写入CSV（实时保存，避免数据丢失）
    echo "$current_row" >> $CSV_FILE
    echo -e "\033[1;32m📊 占比 $percent% 结果已写入CSV（当前整体进度：$current_percent_idx/$TOTAL_PERCENT_COUNT）\033[0m"
    echo -e "--------------------------------------------------\n"
done

# ===================== 任务完成：打印总结信息 =====================
echo -e "\n\033[1;33m=================================================="
echo -e "🎉 所有全量测试任务执行完毕！"
echo -e "==================================================\033[0m"
echo -e "📁 最终汇总结果文件：\033[1;36m$CSV_FILE\033[0m"
echo -e "📋 任务汇总：共处理 $TOTAL_PERCENT_COUNT 个占比（$MIN_PERCENT~$MAX_PERCENT%），每个占比 $TOTAL_SIZE_COUNT 个尺寸"
echo -e "⚠️  备注1：CSV中标记为'OOM'的项表示该尺寸+占比组合无法运行（多为8192尺寸显存不足）"
echo -e "⚠️  备注2：占比0%时结果为0.0是正常现象（全0矩阵无有效EDM计算）"
echo -e "✅ 任务完成，可直接用Excel/记事本打开$CSV_FILE查看完整结果"
