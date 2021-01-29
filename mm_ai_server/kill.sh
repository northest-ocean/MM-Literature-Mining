ps -ef | grep cat | grep -v grep | awk '{print }' | xargs kill -9
