# 字节
## 猴子吃香蕉:给猴子发香蕉，猴子排队依次取食，会多拿食物但是不会超过自身食量的两倍，且不超过当前剩余香蕉的一半，而最后一只可拿完剩余的所有，问最少需要准备多少香蕉？
### input：数组，每个猴子的食量
```py
class Monkey():
    def monkeyEatBanana(monkey):
        n = len(monkey)
        dp = [0] * (n+1)    #dp[i]代表后i只猴子吃饱所需最少香蕉数
        dp[1] = monkey[n - 1];
        for i in range(n-2, -1, -1):
            if monkey[i] < dp[n-i-1]: dp[n-i] = min(2*monkey[i]+dp[n-i-1],2*dp[n-i-1])
            else: dp[n-i] = 2*monkey[i]
        return dp[n]
```
## 有N块蛋糕排成一列，最多可选取其中连续M块，问可收获蛋糕的美味值最大是多少？
### 最大连续子串+窗口，维护一个最大值
```py
class DeliciousVal():
    def sol_1(self,N,M,alist):
        max_ = []
        for i in range(1,M+1):
            if i == 1: max_.append(max(max(alist),0))           #如果美味值全为负数，则不选
            else:
                max_i = sum(alist[0:i])
                for j in range(1,len(alist)-i+1):
                    if sum(alist[j:j+i]) > max_i: max_i = sum(alist[j:j+i])
                max_.append(max(max_[-1],max_i))
        return max_[-1]
```
