import subprocess
import time

# 需要传递的命令行参数列表
args_list = [
    ['arg1', 'arg2'],
    ['arg3', 'arg4'],
    ['arg5', 'arg6'],
    # 添加更多的参数列表
]

# 要调用的脚本路径
script_path = 'D:/SourceCode/sdp/vault/args.py'

# 循环调用脚本
for args in args_list:
    # 构造命令行
    command = ['python', script_path] + args
    print(f'Executing: {" ".join(command)}')

    # 调用脚本并等待完成
    result = subprocess.run(command, capture_output=True, text=True)

    # 打印输出和错误信息（如果有）
    print('Output:', result.stdout)
    if result.stderr:
        print('Error:', result.stderr)

    # 可选：添加延时以确保资源释放
    time.sleep(10)  # 根据需要调整时间

print('All scripts have been executed.')
