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
        
        
