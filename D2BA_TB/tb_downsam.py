import torch
import torch.nn.functional as F
import numpy as np
import time
import os


# input、outputs、weights 存放目录
root_path      = './downsam/'
#  check if the folder './conv2d/' exists and create it if it doesn't
if not os.path.exists(root_path):
    os.makedirs(root_path)
    print(f"Folder '{root_path}' created successfully.")
else:
    print(f"Folder '{root_path}' already exists.")

weight_path    = root_path + "weight.txt"
bias_path      = root_path + "bias.txt"
inputs_path    = root_path + "inputs.txt"
results_path   = root_path + "results.txt"
hls_path       = root_path + "vitis.txt"
diff_path      = root_path + "diff.txt"
#==========================================
# 卷积层相关参数定义
IN_C         = 128
IN_H         = 32
IN_W         = 32

# k3 s1 p1
FLITER_H     = 4
FLITER_w     = FLITER_H
PADDING      = 1
STRIDE       = 2

OUT_C        = 256
OUT_H        = 16
OUT_W        = 16

BATCH_SIZE = 1
#==========================================

def generate_and_compute():
    # (1) generate input
    # 取值 0~1 之间
    inputs = np.random.random((BATCH_SIZE, IN_C, IN_H, IN_W))
    inputs = inputs.astype(np.float32) # float64 to float32
    #print(inputs)
    inputs.tofile("test/ds_conv/in.bin")
    with open(inputs_path, 'w') as file:
        for n in range(BATCH_SIZE):
            for c in range(IN_C):
                for h in range(IN_H):
                    for w in range(IN_W):
                        value = inputs[n][c][h][w]
                        file.write(str(value)+"\n")

    input_lines = BATCH_SIZE * IN_C * IN_H * IN_W
    print("共有{}行 input".format(input_lines))

    # (2) generate weight and bias
    weights = np.random.rand(OUT_C, IN_C, FLITER_H, FLITER_w) # 生成浮点数 0~1
    weights = weights*2 - 1 #-1~1
    weights = weights.astype(np.float32)# float64 to float32
    weights.tofile("test/ds_conv/w.bin")
    with open(weight_path, 'w') as file:
        for n in range(OUT_C):
            for c in range(IN_C):
                for h in range(FLITER_H):
                    for w in range(FLITER_w):
                        value = weights[n][c][h][w]
                        #value = value.astype(np.int8) # 以 int8 格式写入
                        file.write(str(value)+"\n")

    weight_lines = OUT_C*IN_C*FLITER_H*FLITER_w
    print("共有{}行 weight".format(weight_lines))
    # generate bias
    biases = np.zeros(OUT_C) # 0~1 float
    biases = biases.astype(np.float32)  # float64 to float32
    biases.tofile('test/ds_conv/b.bin')
    with open(bias_path, 'w') as file:
        for n in range(OUT_C):
            value = biases[n]
            file.write(str(value)+"\n")

    print("共有{}行 bias".format(OUT_C))
    # (3) compute results
    inputs  = torch.tensor(inputs)
    weights = torch.tensor(weights)
    biases  = torch.tensor(biases)
    print("inputs.shape:", inputs.shape)
    print("weights.shape:",weights.shape)
    print("biases.shape:", biases.shape)
    outputs = F.conv2d(input=inputs, weight=weights, bias=biases, stride=STRIDE, padding=PADDING, groups=1)
    # outputs = F.conv_transpose2d(input=inputs, weight=weights, bias=biases, stride=STRIDE, padding=PADDING)
    print(outputs.dtype, outputs.shape)

    # (4) record results,保存结果（10进制）
    N_outputs = outputs.shape[0]
    C_outputs = outputs.shape[1]
    H_outputs = outputs.shape[2]
    W_outputs = outputs.shape[3]
    with open(results_path,'w') as file:
        for n in range(N_outputs):
            for c in range(C_outputs):
                for h in range(H_outputs):
                    for w in range(W_outputs):
                        value = outputs[n][c][h][w].numpy()
                        # print(value)
                        # 写入文件
                        file.write(str(value) + "\n")

    output_lines = N_outputs*C_outputs*H_outputs*W_outputs
    print("共有{}行 output".format(output_lines))
    # 由于数据较多，执行后等待30s左右文件会更新完毕
    print(time.strftime("%Y-%m-%d %H:%M:%S") + "| success！") # 打印时间戳


precent_limit = 1 # 误差允许低于 limit %
abs_limit     = 0.01 # 绝对值差值允许低于 abs_limit
def check_diff():
    with open(hls_path, 'r') as file1:
        with open(results_path, 'r') as file2:
            with open(diff_path, 'w') as file3:
                for c in range(OUT_C):
                    for h in range(OUT_H):
                        for w in range(OUT_W):
                            value_pytorch = float(file2.readline())
                            value_hls     = float(file1.readline().strip("\n"))
                            diff          = value_pytorch - value_hls
                            # 差值/原值，百分比
                            precent = (diff * 100) / value_pytorch
                            if ((abs(diff) > abs_limit) or (precent > precent_limit)):
                                file3.write(str(c * OUT_H * OUT_W + h * OUT_W + w + 1) + str(" : ") + str(
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