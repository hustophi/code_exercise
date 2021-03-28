#判断一个链表是否有环，如果采用暴力求解，时间复杂度为O(n^2)
#此题可考虑采用快慢指针，快指针faST每次前进两步，而慢指针slow前进一步
#若无环，那么fast或fast.next必然先到达链表末尾None;
#否则,slow进入环前,fast已进入环，当slow遍历环中所有节点时,fast与slow必然会相遇
#时间复杂度O(n),空间复杂度O(1),following is code:
###################################################
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
#
# @param head ListNode类 
# @return bool布尔型
#
class Solution:
    def hasCycle(self , head ):
        if head:
            slow = head
            fast = head
            flag = False
            while not flag and fast and fast.next:
                    slow = slow.next
                    fast = fast.next.next
                    if slow == fast:
                        flag = True
        else:
            flag = False
        return flag
###################################################
#Fibinacci数列非递归
#每次记录并更新两个值(当前值及其前一个,类似于动态规划),following is code:
###################################################
class Solution:
    def Fibonacci(self, n):
        zero,one=0,1
        for _ in range(n):
            zero,one=one,zero+one
        return zero
###################################################
#给定一个m x n大小的矩阵（m行，n列），按螺旋的顺序返回矩阵中的所有元素。
# @param matrix int整型二维数组 
# @return int整型一维数组
#可利用递归，重点在于判断矩阵行列的奇偶性
class Solution:
    def spiralOrder(self , matrix ):
        if not matrix:
            return []
        else:
            return spiralHelp(matrix,0,len(matrix)-1,len(matrix[0])-1,[])
def spiralHelp(matrix,start,m,n,li):
    if start <= n and start <= m:
        for col in range(start,n+1):
            li.append(matrix[start][col])
        for row in range(start+1,m+1):
            li.append(matrix[row][n])
        for col in range(n-1,start-1,-1):
            if start != m:                     #若无此判断,奇*奇阶矩阵不满足,即会重复便利,下同
                li.append(matrix[m][col])
        for row in range(m-1,start,-1):
            if start != n:                     #若无此判断,奇*偶阶矩阵不满足
                li.append(matrix[row][start])
        spiralHelp(matrix,start+1,m-1,n-1,li)
    return li
###################################################
#假设你有一个数组，其中第 i 个元素是股票在第 i 天的价格。
#你有一次买入和卖出的机会。（只有买入了股票以后才能卖出）,计算可以获得的最大收益。 
# @param prices int整型一维数组 
# @return int整型
class Solution:
    def maxProfit(self , prices ):
        maxprofit = 0
        dp = 0                                       #dp记为第i天卖出所获最大收益,因为dp[i]只可能使用了上一时刻的dp[i-1],而且ans可以在dp生成过程中算出来,所以没必要把dp开成数组
        for i in range(1,len(prices)):
            dp = max(dp, 0) + prices[i]-prices[i-1]  #则dp[i+1]=max(prices[i+1]-prices[i]+dp[i],prices[i+1]-prices[i])
            if dp > maxprofit:
                maxprofit = dp
        return maxprofit
###################################################    
