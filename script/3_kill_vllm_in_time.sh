#!/bin/bash

# 等待3小时后杀死多个进程的脚本

# 设置要杀死的进程ID列表（用空格分隔）
PIDS="2905245 2905246 2905247 2905248 2905249 2905250 2905251 2905252"

echo "脚本已启动，将在3小时后杀死以下进程: $PIDS"
echo "开始时间: $(date)"

# 等待3小时（10800秒）
sleep 10800

echo "时间到！开始处理进程..."
echo "结束时间: $(date)"

# 遍历每个进程ID
for PID in $PIDS
do
    echo "处理进程 $PID ..."
    
    # 检查进程是否存在
    if ps -p $PID > /dev/null 2>&1
    then
        echo "正在杀死进程 $PID ..."
        kill $PID
        
        # 等待一下然后检查
        sleep 1
        if ps -p $PID > /dev/null 2>&1
        then
            echo "进程 $PID 仍然存在，使用强制杀死..."
            kill -9 $PID
        fi
        
        echo "✓ 进程 $PID 已被杀死"
    else
        echo "进程 $PID 已不存在，跳过"
    fi
    echo "------------------------"
done

echo "所有进程处理完成！"