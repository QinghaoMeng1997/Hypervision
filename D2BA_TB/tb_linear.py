import torch
import torch.nn.functional as F
import numpy as np
import time
import os


# input、outputs、weights 存放目录
root_path      = ('./headlinear/')
if not os.path.exists(root_path):
    os.makedirs(root_path)
    print(f"Folder '{root_path}' created successfully.")
else:
    print(f"Folder '{root_path}' already exists.")

weight_path    = root_path + "weight.txt"
bias_path      = root_path + "bias.txt"
inputs_path    = root_path + "inputs.txt"
results_path   = root_path + "results.txt"
hls_path       = root_path + "hls.txt"
diff_path      = root_path + "diff.txt"
rescale_path   = root_path + "rescale.txt"
#==========================================
# 全连接层层相关参数定义
head         = 4
pre_c        = 32
size         = 1024
feature      = 32

"""
# 权重位宽,例如 位宽为3，则取值范围为(-3, 3)
weight_bit   = 3
weight_max   = 2**(weight_bit-1) - 1
weight_min   = -weight_max
"""
BATCH_SIZE = 1
#==========================================

def generate_and_compute():
    # (1) generate input
    # 取值 0~1 之间
    inputs = np.random.random((BATCH_SIZE, head, pre_c, size))
    inputs = inputs.astype(np.float32) # float64 to float32
    inputs.tofile("test/lin/in.bin")
    with open(inputs_path, 'w') as file:
        for n in range(BATCH_SIZE):
            for h in range(head):
                for c in range(pre_c):
                    for s in range(size):
                        value = inputs[n][h][c][s]
                        file.write(str(value)+"\n")

    input_lines = BATCH_SIZE * head * pre_c * size
    print("共有{}行 input".format(input_lines))

    # (2) generate weight and bias
    weights = np.random.rand(BATCH_SIZE, head, feature, size) # 生成浮点数 0~1
    weights = weights*2 - 1 #-1~1
    weights = weights.astype(np.float32)# float64 to float32
    weights.tofile("test/lin/w.bin")
    with open(weight_path, 'w') as file:
        for n in range(BATCH_SIZE):
            for h in range(head):
                for f in range(feature):
                    for s in range(size):
                        value = weights[n][h][f][s]
                        file.write(str(value)+"\n")

    weight_lines = BATCH_SIZE*head*feature*pre_c
    print("共有{}行 weight".format(weight_lines))

    # generate bias
    biases = np.zeros(head*feature) # 0
    biases = biases.astype(np.float32)  # float64 to float32
    biases.tofile("test/lin/b.bin")
    with open(bias_path, 'w') as file:
        for n in range(head*feature):
            value = biases[n]
            file.write(str(value)+"\n")

    # print(biases)
    print("共有{}行 bias".format(head*feature))

    rescale = np.random.rand(head, 1, 1)
    rescale = rescale.astype(np.float32)
    rescale.tofile("test/lin/r.bin")
    with open(rescale_path, 'w') as file:
        for h in range(head):
            for i in range(1):
                for j in range(1):
                    value = rescale[h][i][j]
                    file.write(str(value)+"\n")

    # (3) compute results

    inputs  = torch.tensor(inputs)
    weights = torch.tensor(weights)
    biases  = torch.tensor(biases)
    rescale = torch.tensor(rescale)
    print("inputs.shape:", inputs.shape)
    print("weights.shape:",weights.shape)
    print("biases.shape:", biases.shape)
    print("rescale shape:",rescale.shape)
    outputs = (inputs @ weights.transpose(-2,-1))*rescale
    # outputs = (weights @ inputs)
    print(outputs.dtype, outputs.shape)


    # (4) record results,保存结果（10进制）
    N_outputs = outputs.shape[0]
    H_outputs = outputs.shape[1]
    C_outputs = outputs.shape[2]
    F_outputs = outputs.shape[3]
    with open(results_path,'w') as file:
        for n in range(N_outputs):
            for h in range(H_outputs):
                for c in range(C_outputs):
                    for f in range(F_outputs):
                        value = outputs[n][h][c][f].numpy()
                    # 写入文件
                        file.write(str(value) + "\n")

    # print(outputs)
    output_lines = N_outputs*H_outputs*C_outputs*F_outputs
    print("共有{}行 output".format(output_lines))
    # 由于数据较多，执行后等待30s左右文件会更新完毕
    print(time.strftime("%Y-%m-%d %H:%M:%S") + "| success！") # 打印时间戳


precent_limit = 1 # 误差允许低于 limit %
abs_limit     = 0.1 # 绝对值差值允许低于 abs_limit
def check_diff():
    with open(hls_path, 'r') as file1:
        with open(results_path, 'r') as file2:
            with open(diff_path, 'w') as file3:
                for r in range(head):
                    for c in range(pre_c):
                        for f in range(feature):
                            value_pytorch = float(file2.readline())
                            value_hls     = float(file1.readline().strip("\n"))
                            diff          = value_pytorch - value_hls
                        # 差值/原值，百分比
                            precent = (diff * 100) / value_pytorch
                            if ((abs(diff) > abs_limit) or (precent > precent_limit)):
                                file3.write(str(r * pre_c + c + 1) + str(" : ") + str(
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