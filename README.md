## Dependencies

- python3.7
- torch1.2
- numpy
- tqdm
- sklearn
- pickle


## 运行方式

python example.py 0 0 --num 3

第一个参数user type，0表示模拟用户，1表示命令行输入

第二个参数agent type，0表示基于规则的系统，1表示基于强化学习的系统

第三个参数（可选），对话次数，默认为1

note：选择命令行输入时会提供目标餐厅信息，根据提供的信息回答系统提出的问题


## 参考论文
Conversational Recommender System

**<https://arxiv.org/abs/1806.03277>**


## 生成训练数据（可选）

使用脚本make_data.sh

在每个模块的子文件夹中，包含各个模块训练使用的代码