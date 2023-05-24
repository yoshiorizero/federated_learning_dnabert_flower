# federated_learning_dnabert_flower
基于联邦学习框架Flower构建的DNABERT预训练模型，3个Client和1个Server，各自拥有一个完整DNABERT模型

联邦聚合算法采用FedAvg

启动Server:
> python server.py --do_train --do_eval --mlm --line_by_line --overwrite_output_dir --evaluate_during_training

待Server向Client请求初始参数时，启动Client：
> python client1.py --do_train --do_eval --mlm --line_by_line --overwrite_output_dir --evaluate_during_training
> python client2.py --do_train --do_eval --mlm --line_by_line --overwrite_output_dir --evaluate_during_training
> python client3.py --do_train --do_eval --mlm --line_by_line --overwrite_output_dir --evaluate_during_training

其余args参考DNABERT原模型介绍
