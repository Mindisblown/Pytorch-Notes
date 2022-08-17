**get_multi_dataloaders.py**

​		加载多个数据，一般用于半监督学习引入unlabel数据

**stratified_sampler.py**

​		分层采样，每种类别采样的均匀性。在分层采样的前提下使用fisherYatesShuffle进行打乱，然后再随机打乱

​		A:100 B:200 C:300，随机采样60份C的概率要大于其他两种，从A中取10，B中20，C中取30,即为分层采样

**cuda_data_prefetch.py**

​		数据加载加速，可将数据直接加载至cuda stream中

