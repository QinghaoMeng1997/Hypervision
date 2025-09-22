import torch
import torch.nn.functional as F
import numpy as np
import time
import os


# input、outputs、weights 存放目录
root_path      = ('./normalize/')
if not os.path.exists(root_path):
    os.makedirs(root_path)
    print(f"Folder '{root_path}' created successfully.")
else:
    print(f"Folder '{root_path}' already exists.")

# weight_path    = root_path + "weight.txt"
# bias_path      = root_path + "bias.txt"
inputs_path    = root_path + "inputs.txt"
results_path   = root_path + "results.txt"
hls_path       = root_path + "hls.txt"
diff_path      = root_path + "diff.txt"
#==========================================
# 全连接层层相关参数定义
C            = 256
HEAD         = 8
C_pre        = C//HEAD
IN_R         = 16
IN_C         = 16
SIZE         = IN_C*IN_R
BATCH_SIZE = 1
#==========================================

def generate_and_compute():
    # (1) generate input
    # 取值 0~1 之间
    inputs = np.random.uniform(-1,1,(BATCH_SIZE, HEAD, C_pre, SIZE))
    inputs = inputs.astype(np.float32) # float64 to float32
    inputs.tofile("test/norm/in.bin")
    with open(inputs_path, 'w') as file:
        for n in range(BATCH_SIZE):
            for h in range(HEAD):
                for c in range(C_pre):
                    for s in  range(SIZE):
                        value = inputs[n][h][c][s]
                        file.write(str(value)+"\n")

    input_lines = BATCH_SIZE * C * SIZE
    print("共有{}行 input".format(input_lines))

    inputs  = torch.tensor(inputs)
    print("inputs.shape:", inputs.shape)
    outputs = F.normalize(inputs, dim=-1, p=2)
    print(outputs.dtype, outputs.shape)

    # (4) record results,保存结果（10进制）
    N_outputs = outputs.shape[0]
    H_outputs = outputs.shape[1]
    C_outputs = outputs.shape[2]
    S_outputs = outputs.shape[3]

    with open(results_path,'w') as file:
        for n in range(N_outputs):
            for h in range(H_outputs):
                for c in range(C_outputs):
                    for s in range(S_outputs):
                        value = outputs[n][h][c][s].numpy()
                    # 写入文件
                        file.write(str(value) + "\n")

    # print(outputs)
    output_lines = N_outputs*H_outputs*C_outputs*S_outputs
    print("共有{}行 output".format(output_lines))
    # 由于数据较多，执行后等待30s左右文件会更新完毕
    print(time.strftime("%Y-%m-%d %H:%M:%S") + "| success！") # 打印时间戳


precent_limit = 2 # 误差允许低于 limit %
abs_limit     = 0.01 # 绝对值差值允许低于 abs_limit
def check_diff():
    with open(hls_path, 'r') as file1:
        with open(results_path, 'r') as file2:
            with open(diff_path, 'w') as file3:
                for r in range(IN_R):
                    for c in range(IN_C):
                        value_pytorch = float(file2.readline())
                        value_hls     = float(file1.readline().strip("\n"))
                        diff          = value_pytorch - value_hls
                        # 差值/原值，百分比
                        precent = (diff * 100) / value_pytorch
                        if ((abs(diff) > abs_limit) or (precent > precent_limit)):
                            file3.write(str(r * IN_R + c + 1) + str(" : ") + str(
                                diff) + "  " + str(precent) + "% \n")

    print(time.strftime("%Y-%m-%d %H:%M:%S") + "| success！")  # 打印时间戳


if __name__ == "__main__":
    mode = input("请输入要进行的操作:\n "
                 "0:生成激励并计算参考输出\n "
                 "1:进行结果对比\n"
                 )
    if mode=="0":
        generate_and_compute()
    elif mode=="1":
        check_diff()
    else:
        print("输出的值错误，请重新输入")