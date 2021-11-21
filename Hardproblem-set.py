#合并k个已排序的链表并将其作为一个已排序的链表返回。分析并描述其复杂度(O(nlogk),)
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
# @param lists ListNode类一维数组 
# @return ListNode类
#
import heapq
class Solution:
    def mergeKLists(self , lists ):
        tmpNode = ListNode(-1)
        cur = tmpNode
        tmpList = [(lists[i].val, i) for i in range(len(lists)) if lists[i]]
        heapq.heapify(tmpList)      #trick:以(节点值,节点索引)建堆, 从而避免值相同时无法比较而无法建堆的问题
        while tmpList:
            val, idx = heapq.heappop(tmpList)
            cur.next = lists[idx]
            cur = cur.next
            if cur.next: 
                heapq.heappush(tmpList,(cur.next.val, idx))
                lists[idx] = cur.next      #更新lists第idx个节点
        return tmpNode.next
        # write code here
################################################
#假设你有一个数组prices，长度为n，其中prices[i]是某只股票在第i天的价格，请根据这个价格数组，返回买卖股票能获得的最大收益
#1. 你最多可以对该股票有两笔交易操作，一笔交易代表着一次买入与一次卖出，但是再次购买前必须卖出之前的股票
#2. 如果不能获取收益，请返回0
#3. 假设买入卖出均无手续费
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
#
# 两次交易所能获得的最大收益
# @param prices int整型一维数组 股票每一天的价格
# @return int整型
#key:遍历数组，以当前天作为两次交易的分隔点求最大收益，两次的最大收益可分别用动态规划求
class Solution:
    def maxProfit(self , prices: List[int]) -> int:
        res = 0
        if not prices: return res
        lenth = len(prices)
        first = [0] * lenth     #first[i]为[0,i]内只进行一次交易的最大收益
        minimum = prices[0]
        for i in range(1,lenth):
            minimum = min(prices[i-1], minimum)  #[0,i-1]内的最低价格
            first[i] = max(prices[i] - minimum, first[i-1])  #分在第i天卖出和第i天前卖出讨论
        second = [0] * lenth    #second[i]为[i,lenth-1]内只进行一次交易的最大收益
        maximum = prices[-1]
        for i in range(lenth-2, -1, -1):
            maximum = max(maximum, prices[i+1]) #[i+1, lenth-1]的最高价格
            second[i] = max(maximum - prices[i], second[i+1]) #分在第i天买入和第i天后买入讨论
        for i in range(lenth):
            res = max(first[i] + second[i], res)
        return res
        # write code here
################################################
#二叉树里面的路径被定义为:从该树的任意节点出发，经过父=>子或者子=>父的连接，达到任意节点的序列。
#注意: 1.同一个节点在一条二叉树路径里中最多出现一次; 2.一条路径至少包含一个节点，且不一定经过根节点
#给定一个二叉树的根节点root，请你计算它的最大路径和
#key: 递归计算经过每个节点的最大路径和,维护一个最大值

# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

# @param root TreeNode类 
# @return int整型
#
class Solution:
    def maxPathSum(self , root ):
        res = -float('inf')  #初始化
        def help(root):      #IMPORTANT: help函数递归计算以该节点为根节点的子树中寻找以该节点为起点的一条路径,使得该路径上的节点值之和最大
            nonlocal res     #维护一个全局变量, 记录最大值
            if root:
                l = help(root.left)
                r = help(root.right)
                val = max(l, 0) + max(r, 0) + root.val  #val为经过该节点的最大路径和 (NOTES:此步可直接在help递归过程中得到,而无需在help外单独使用递归,可减少空间复杂度)
                if val > res:
                    res = val 
                return max(l, r, 0) + root.val
            else:
                return 0
        help(root)
        return res 
        # write code here        

