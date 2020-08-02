# coding:UTF-8

import numpy as np
# from FM_train import getPrediction
def loadModel(model_file):
    '''导入FM模型
    input:  model_file(string)FM模型
    output: w0, np.mat(w).T, np.mat(v)FM模型的参数
    '''
    f = open(model_file)
    line_index = 0
    w0 = 0.0
    w = []
    v = []
    for line in f.readlines():
        lines = line.strip().split("\t")
        if line_index == 0:  # w0
            w0 = float(lines[0].strip())
        elif line_index == 1:  # w
            for x in lines:
                w.append(float(x.strip()))
        else:
            v_tmp = []
            for x in lines:
                v_tmp.append(float(x.strip()))
            v.append(v_tmp)
        line_index += 1
    f.close()
    return w0, np.mat(w).T, np.mat(v)

if __name__ == "__main__":
    # 1、导入测试数据
    # dataTest = loadDataSet("test_data.txt")
    # 2、导入FM模型
    w0, w , v = loadModel('data/weights')

