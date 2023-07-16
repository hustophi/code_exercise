# 字节
## 猴子吃香蕉:给猴子发香蕉，猴子排队依次取食，会多拿食物但是不会超过自身食量的两倍，且不超过当前剩余香蕉的一半，而最后一只可拿完剩余的所有，问最少需要准备多少香蕉？
### input：数组，每个猴子的食量
```py
import time
class Monkey():
	def violentSol(self,alist):
		N = len(alist)
		minB = max(2*alist[0],sum(alist))
		enough = False
		if N > 1:
			while not enough:
				resB = minB
				for i in range(N-1):
					if resB//2 >= alist[i]:
						take = min(resB//2,2*alist[i])
						resB = resB - take
						if resB < sum(alist[i+1:]):
							minB = minB + 1
							break
					else:
						minB = minB + 1
				else:
					if resB >= alist[-1]:
						enough = True
					else:
						minB = minB + 1
		else:
			minB = 2 * alist[0]
		return minB
```
## 有N块蛋糕排成一列，最多可选取其中连续M块，问可收获蛋糕的美味值最大是多少？
### 最大连续子串+窗口，维护一个最大值
```py
class DeliciousVal():
	def sol_1(self,N,M,alist):
		max_ = []
		for i in range(1,M+1):
			if i == 1:
				max_.append(max(max(alist),0))           #如果美味值全为负数，则不选
			else:
				max_i = sum(alist[0:i])
				for j in range(1,len(alist)-i+1):
					if sum(alist[j:j+i]) > max_i:
						max_i = sum(alist[j:j+i])
				max_.append(max(max_[-1],max_i))
		return max_[-1]
```
