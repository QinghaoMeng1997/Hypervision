import torch
import torch.nn.functional as F
import numpy as np
import time
import os


# input、outputs、weights 存放目录
root_path      = ('./softmax/')
if not os.path.exists(root_path):
    os.makedirs(root_path)
    print(f"Folder '{root_path}' created successfully.")
else:
    print(f"Folder '{root_path}' already exists.")

inputs_path    = root_path + "inputs.txt"
results_path   = root_path + "results.txt"
hls_path       = root_path + "hls.txt"
diff_path      = root_path + "diff.txt"
#==========================================
head            = 4
pre_channel     = 32
size            = 32
folder          = 16
item            = head*pre_channel*size
BATCH_SIZE = 1
#==========================================

class ApproxSoftmax(torch.nn.Module):

    def __init__(self, n=64, dim=-1):
        """
        自定义的 Softmax 函数，使用 (1 + x/n)^n 近似 e^x
        :param n: 使用 (1 + x/n)^n 的 n 值，默认是 64
        :param dim: 计算 Softmax 的维度
        """
        super(ApproxSoftmax, self).__init__()
        self.n = n
        self.dim = dim

    def approx_exp(self, x):
        """
        使用 (1 + x/n)^n 近似 e^x
        :param x: 输入张量
        :return: 近似后的结果
        """
        return (1 + x / self.n) ** self.n

    def forward(self, x):
        """
        使用近似的指数函数计算 Softmax
        :param x: 输入张量
        :return: 计算后的 Softmax 结果
        """
        # 找到每个组的最大值，沿着指定维度进行
        max_val = torch.max(x, dim=self.dim, keepdim=True)[0]

        # 每个数减去该维度的最大值
        x_stable = x - max_val

        # 使用 (1 + x/n)^n 近似 e^x
        approx_exp_x = self.approx_exp(x_stable)

        # 归一化 Softmax 结果
        return approx_exp_x / torch.sum(approx_exp_x, dim=self.dim, keepdim=True)

def generate_and_compute():
    # (1) generate input
    # 取值 0~1 之间
    inputs = np.random.random((BATCH_SIZE, item, folder))
    inputs = inputs.astype(np.float32) # float64 to float32
    inputs.tofile("test/soft/in.bin")
    with open(inputs_path, 'w') as file:
        for n in range(BATCH_SIZE):
            for c in range(item):
                for r in range(folder):
                    value = inputs[n][c][r]
                    file.write(str(value)+"\n")

    input_lines = BATCH_SIZE * item * folder
    print("共有{}行 input".format(input_lines))
    inputs = np.sum(inputs,axis=-1)
    inputs = inputs.reshape(head, pre_channel, size)
    print(inputs.shape)
    inputs  = torch.tensor(inputs)
    print("inputs.shape:", inputs.shape)
    approx_softmax = ApproxSoftmax(n=64)
    outputs = approx_softmax(inputs)
    print(outputs.dtype, outputs.shape)

    # (4) record results,保存结果（10进制）
    N_outputs = outputs.shape[0]
    C_outputs = outputs.shape[1]
    S_outputs = outputs.shape[2]
    with open(results_path,'w') as file:
        for n in range(N_outputs):
            for r in range(C_outputs):
                for s in range(S_outputs):
                    value = outputs[n][r][s].numpy().astype(np.float16)
                    # 写入文件
                    file.write(str(value) + "\n")

    output_lines = N_outputs*C_outputs*S_outputs
    print("共有{}行 output".format(output_lines))
    # 由于数据较多，执行后等待30s左右文件会更新完毕
    print(time.strftime("%Y-%m-%d %H:%M:%S") + "| success！") # 打印时间戳


precent_limit = 1 # 误差允许低于 limit %
abs_limit     = 0.01 # 绝对值差值允许低于 abs_limit
def check_diff():
    with open(hls_path, 'r') as file1:
        with open(results_path, 'r') as file2:
            with open(diff_path, 'w') as file3:
                for c in range(channel):
                    for r in range(size):
                        value_pytorch = float(file2.readline())
                        value_hls     = float(file1.readline().strip("\n"))
                        diff          = value_pytorch - value_hls
                        # 差值/原值，百分比
                        precent = (diff * 100) / value_pytorch
                        if ((abs(diff) > abs_limit) or (precent > precent_limit)):
                            file3.write(str(c *size+ r) + str(" : ") + str(
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