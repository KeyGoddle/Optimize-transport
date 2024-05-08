# Optimize-transport
Транспортная задача c использованием оптимизации
Minimize: $$Z= \sum_{j=1}^n\ \sum_{i=1}^n (x_{i,j} * c_{i,j}) \$$

Где:
    $x_{i,j}$ - кол-во груза грузовка j доставленное в точку i     
    $c_{i,j}$ - стоимость

**Ограничения:**
1) $$\sum_{i=1}^m\ x_{i,j} * d_{i} >= d_i $$
2) $$\sum_{j=1}^n\ x_{i,j}  =< q_{i} $$
3) $$\sum_{j=1}^n\ x_{i,j}*p_{i,j}  =< 24 $$

Где:
    $$d_{i} - вес
    p_{i,j} - время$$
    
