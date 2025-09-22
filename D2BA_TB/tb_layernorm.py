import torch
import numpy as np
import time
import os
import torch.nn as nn
import numbers
from einops import rearrange

# input、outputs、weights 存放目录
root_path      = ('./layernorm/')
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
#==========================================
C            = 128
IN_R         = 32
IN_C         = 32
BATCH_SIZE = 1
#==========================================
#测试代码
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
#===========================================================

def generate_and_compute():
    # (1) generate input
    # 取值 0~1 之间
    inputs = np.random.random((BATCH_SIZE, C, IN_R, IN_C))
    inputs = inputs.astype(np.float32) # float64 to float32
    inputs.tofile("test/layernorm/in.bin")
    with open(inputs_path, 'w') as file:
        for n in range(BATCH_SIZE):
           for c in range(C):
                for h in range(IN_R):
                    for w in range(IN_C):
                        value = inputs[n][c][h][w]
                        file.write(str(value)+"\n")

    input_lines = BATCH_SIZE * C * IN_R * IN_C
    print("共有{}行 input".format(input_lines))

    # (2) generate weight and bias
    weights = np.random.rand(C) # 生成浮点数 0~1
    weights = weights*2 - 1 #-1~1
    weights = weights.astype(np.float32)# float64 to float32
    weights.tofile("test/layernorm/w.bin")
    with open(weight_path, 'w') as file:
        for c in range(C):
            value = weights[c]
            file.write(str(value)+"\n")

    weight_lines = C
    print("共有{}行 weight".format(weight_lines))

    # generate bias
    biases = np.random.rand(C) # 0~1 float
    biases = biases.astype(np.float32)  # float64 to float32
    biases.tofile("test/layernorm/b.bin")
    with open(bias_path, 'w') as file:
        for c in range(C):
            value = biases[c]
            file.write(str(value)+"\n")

    print(biases)
    print("共有{}行 bias".format(C))
    # (3) compute results
    inputs  = torch.tensor(inputs)
    weights = torch.tensor(weights)
    biases  = torch.tensor(biases)
    print("inputs.shape:", inputs.shape)
    print("weights.shape:",weights.shape)
    print("biases.shape:", biases.shape)

    # 初始化 LayerNorm 模块
    layer_norm = LayerNorm(C)
    # 替换模块中的 weight 和 bias
    with torch.no_grad():  # 不记录梯度信息，避免对参数进行反向传播
        layer_norm.body.weight.copy_(weights)
        layer_norm.body.bias.copy_(biases)

    # 运行 LayerNorm 模块并获取输出
    outputs = layer_norm(inputs)

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
                        value = outputs[n][c][h][w].detach().numpy()
                    # 写入文件
                        file.write(str(value) + "\n")

    # print(outputs)
    output_lines = N_outputs*C_outputs*H_outputs*W_outputs
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