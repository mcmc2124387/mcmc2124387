import torch
import torch.nn as nn
import openpyxl



# 打开文件，获取excel文件的workbook（工作簿）对象
wb = openpyxl.load_workbook("data/2021MCMProblemC_DataSet_plus.xlsx", read_only=True)  # 文件路径
sheet = wb.worksheets[0]

def generate():
    flag = 0
    for row in sheet.rows:
        if flag == 0:  # 排除表头
            flag = 1
            continue
        row_value = [cell.value for cell in row]
        if (row_value[2] == ' ' or row_value[8] == None) and (row_value[8] == '' or row_value[8] == None):
            continue
        item = (row_value[2] if row_value[2] else ' ', row_value[4] if row_value[4] else ' ', 
                row_value[6], row_value[7], row_value[8] if row_value[8] else '',
                1 if row_value[3] == 'Positive ID' else 0 if row_value[3] == 'Negative ID' else 2)
        yield item

g = list(generate())
dataloader = torch.utils.data.DataLoader(g, batch_size=16, shuffle=False, num_workers=2)


