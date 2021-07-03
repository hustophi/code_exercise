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
