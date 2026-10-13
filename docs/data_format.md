# PDW数据格式

UTF-8 CSV，首行 TOA,RF,PW,DOA,PA,Label。每行一个脉冲，按TOA递增。前五列为有限数值；Label为整数发射源ID，只用于评分。自定义标签列用 --label-column 指定。

仿真TOA为任意时间单位，RF对应MHz范围、PW对应微秒范围、DOA为度、PA为任意幅度。范围用于程序演示，不表示真实设备参数。自有记录须保持单位一致。

按行号分窗，不按时间分窗；窗口不足会报错。--count 2 --window 10000 默认读取起点0和10000的两个窗口。缺列、NaN/Inf或非数值必须先清理。

ACC使用Hungarian一对一匹配，F1是对应Macro-F1，K_abs_error是当前窗口预测源数与真实源数的绝对差。
